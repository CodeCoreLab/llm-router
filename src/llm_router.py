#!/usr/bin/env python3
"""
llm_router.py — 可移植的多端点 LLM 路由客户端

单一文件、仅依赖 requests，提供 OpenAI chat.completions 兼容端点的
多级候选路由：优先级遍历、熔断、多 key 轮换、瞬错重试、推理预算治理、
Prompt 缓存与强制 JSON Schema 结构化输出。

快速上手：
    from llm_router import call_llm

    text = call_llm("帮我总结这段日志……")              # 按优先级顺序路由
    text = call_llm("……", model="my-model")            # 按模型名精确路由
    text = call_llm("看这张图", image_path="shot.png")  # 多模态（可选 Pillow 压缩）

配置优先级（从高到低）：
    1. llm_endpoints.yaml（LLM_ENDPOINTS_YAML 可指定路径；api_key 支持列表 → 多 key 轮换）
    2. 环境变量：LLM_P{n}_URL / LLM_P{n}_KEY(或 KEY1..KEY9) / LLM_P{n}_MODEL，n = 1..12
       （P1 另兼容 LLM_API_*，P2 另兼容 LLM_FALLBACK_* 旧命名）
    3. env 文件（LLM_ENV_FILE 指定，解析 export K="V" 形式；不设置则不加载）

环境变量总览（全部可选）：
    LLM_ENDPOINTS_YAML       YAML 配置路径
    LLM_ENV_FILE             env 兜底文件路径
    LLM_RETRY_MAX            单端点重试次数（默认 3）
    LLM_RETRY_BACKOFF        重试退避秒数，逗号分隔（默认 1,3,5）
    LLM_HEALTH_ENABLED       熔断开关（默认 true）
    LLM_CIRCUIT_COOLDOWN     熔断冷却秒数（默认 300）
    LLM_PROMPT_CACHE_TTL     Prompt 缓存 TTL 秒（默认 300，0 关闭）
    LLM_STRUCTURED_LOGGING   结构化 JSON 日志（默认关闭）
    SSL_NO_VERIFY            全局禁用证书校验（自签场景；默认关闭）
    LLM_REASONING_BUDGET_CEILING      推理预算放大上限 tokens（默认 16000）
    LLM_REASONING_BUDGET_MIN_TIMEOUT  放大重试最小超时秒（默认 180）

公开 API：
    call_llm(prompt, ...) -> str | None
    call_llm_with_tools(messages, tools, ...) -> dict
    call_llm_json(prompt, schema, ...) -> dict | None
    list_endpoints() -> list[dict]           # 脱敏诊断视图
    LLMDeadlineExceeded                      # deadline 预算耗尽异常
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import requests

__all__ = [
    "SSL_NO_VERIFY",
    "call_llm",
    "call_llm_with_tools",
    "call_llm_json",
    "list_endpoints",
    "LLMDeadlineExceeded",
]

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 数值解析助手（env/YAML 来的值可能是脏字符串，fail-fast 而非静默 NaN）
# ════════════════════════════════════════════════════════════════════════════

def _to_int(value: Any, ctx: str = "") -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid int{(' ' + ctx) if ctx else ''}: {value!r}") from exc


def _to_float(value: Any, ctx: str = "") -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float{(' ' + ctx) if ctx else ''}: {value!r}") from exc


_MILLISECONDS_PER_SECOND = 1000
_BYTES_PER_KIB = 1024

# 在途图片读入上限（32MB）：
# 超过上限的图片直接拒绝，避免一次性把大文件读进内存（OOM 风险）。
# 2MB 以上的图片会走 Pillow 压缩通路，但压缩前仍需先读入原始字节。
_MAX_IMAGE_READ_BYTES = 32 * 1024 * 1024  # 32MB

# ════════════════════════════════════════════════════════════════════════════
# 默认参数（均可被 YAML defaults 段覆盖）
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 120

# ════════════════════════════════════════════════════════════════════════════
# 重试 / 候选参数
# ════════════════════════════════════════════════════════════════════════════
LLM_RETRY_MAX = _to_int(os.getenv("LLM_RETRY_MAX", "3"))
LLM_RETRY_BACKOFF = [
    _to_int(x) for x in os.getenv("LLM_RETRY_BACKOFF", "1,3,5").split(",") if x.strip()
]
LLM_CANDIDATE_MAX = 12  # 允许 P1..P12

# ════════════════════════════════════════════════════════════════════════════
# 推理预算治理
#
# 推理型模型会先把 token 花在 reasoning 上：finish_reason=length 且
# reasoning_tokens ≈ completion_tokens 时 content 为空。实测：max_tokens=2000
# → reasoning 烧满 2000、content=0；同一 prompt 放大到 8000 → reasoning 3162
# + content 203，正常返回。
#
# 处理：预算耗尽 → max_tokens × BUMP 重试一次，上限 CEILING（防单请求吃掉
# 外层预算）。放大后沿用原 timeout 会读超时（思考链越长生成越久，实测 141s），
# 故设最小超时。
# ════════════════════════════════════════════════════════════════════════════
REASONING_BUDGET_BUMP = 4
REASONING_BUDGET_CEILING = _to_int(os.getenv("LLM_REASONING_BUDGET_CEILING", "16000"))
REASONING_BUDGET_MIN_TIMEOUT = _to_int(os.getenv("LLM_REASONING_BUDGET_MIN_TIMEOUT", "180"))


class LLMDeadlineExceeded(RuntimeError):
    """调用方总预算耗尽（非端点故障）。

    外层任务超时是硬约束：内层重试/候选切换不得跨越它，
    否则外部 SIGKILL 会让调用方的兜底逻辑永远执行不到。
    """


# ════════════════════════════════════════════════════════════════════════════
# 熔断 / 健康
# ════════════════════════════════════════════════════════════════════════════
LLM_HEALTH_ENABLED = os.getenv("LLM_HEALTH_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_CIRCUIT_COOLDOWN = _to_int(os.getenv("LLM_CIRCUIT_COOLDOWN", "300"))

# ════════════════════════════════════════════════════════════════════════════
# Prompt 缓存（进程内 TTL 缓存；仅无 image/无显式 override 的调用参与）
# ════════════════════════════════════════════════════════════════════════════
LLM_PROMPT_CACHE_TTL = _to_int(os.getenv("LLM_PROMPT_CACHE_TTL", "300"))
_PROMPT_CACHE: Dict[str, Tuple[str, float]] = {}
_PROMPT_CACHE_LOCK = threading.Lock()

# ════════════════════════════════════════════════════════════════════════════
# 结构化日志（默认关闭）/ SSL 校验开关（自签证书场景）
# ════════════════════════════════════════════════════════════════════════════
LLM_STRUCTURED_LOGGING = os.getenv("LLM_STRUCTURED_LOGGING", "").lower() in ("1", "true", "yes")
SSL_NO_VERIFY = os.getenv("SSL_NO_VERIFY", "").lower() in ("1", "true", "yes")


# ════════════════════════════════════════════════════════════════════════════
# HTTP 会话（连接池复用；重试由上层路由逻辑控制，adapter 层不做重试）
# ════════════════════════════════════════════════════════════════════════════
_SESSION: Optional["requests.Session"] = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=0,
            )
            _SESSION.mount("http://", adapter)
            _SESSION.mount("https://", adapter)
        return _SESSION


# ════════════════════════════════════════════════════════════════════════════
# 熔断状态
# ════════════════════════════════════════════════════════════════════════════
class _EndpointHealth:
    __slots__ = ("dead_until", "last_latency", "fail_count", "latency_samples", "key_idx")

    def __init__(self) -> None:
        self.dead_until = 0.0
        self.last_latency: Optional[int] = None
        self.fail_count = 0
        self.latency_samples: List[int] = []
        self.key_idx = 0  # 当前 key 索引（多 key 轮换用）


_HEALTH: Dict[str, _EndpointHealth] = {}
_HEALTH_LOCK = threading.Lock()


def _h(url: str) -> _EndpointHealth:
    with _HEALTH_LOCK:
        return _HEALTH.setdefault(url, _EndpointHealth())


def _is_dead(url: str) -> bool:
    if not LLM_HEALTH_ENABLED:
        return False
    return time.time() < _h(url).dead_until


def _mark_dead(url: str) -> None:
    if not LLM_HEALTH_ENABLED:
        return
    # 读-改-写必须整段持锁：多线程并发 _mark_dead 时，锁外写入会互相覆盖
    # dead_until 与 fail_count（竞态窗口内第二轮覆盖第一轮）。
    with _HEALTH_LOCK:
        h = _HEALTH.setdefault(url, _EndpointHealth())
        now = time.time()
        if h.dead_until <= now:
            h.dead_until = now + LLM_CIRCUIT_COOLDOWN
        h.fail_count += 1


def _mark_good(url: str, latency_ms: Optional[int] = None) -> None:
    if not LLM_HEALTH_ENABLED:
        return
    # 与 _mark_dead 同源：写段整体持锁，避免并发读改写竞态。
    with _HEALTH_LOCK:
        h = _HEALTH.setdefault(url, _EndpointHealth())
        h.dead_until = 0.0
        h.fail_count = 0
        if latency_ms is not None:
            h.last_latency = latency_ms
            h.latency_samples.append(latency_ms)
            if len(h.latency_samples) > 5:
                h.latency_samples.pop(0)


def _is_hard_error(exc: BaseException) -> bool:
    """硬错 = 立即切下一候选（无退避）。连接错 / 鉴权错(401/403/404)。"""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in (401, 403, 404):
            return True
    s = str(exc).lower()
    return any(m in s for m in (
        "no route to host", "connection refused", "name or service not known",
        "failed to resolve", "getaddrinfo", "connection error",
        "unauthorized", "incorrect api key", "invalid api key",
    ))


def _is_rate_limited(exc: BaseException) -> bool:
    """瞬错 = 429 / 5xx / 超时；429 在多 key 场景下触发 key 轮换。"""
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in (429, 500, 502, 503, 504):
            return True
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    s = str(exc).lower()
    return any(m in s for m in ("429", "500", "502", "503", "504", "timeout", "rate limit"))


# ════════════════════════════════════════════════════════════════════════════
# env 文件兜底加载（可选：LLM_ENV_FILE 指定路径，解析 export K="V" 形式）
# ════════════════════════════════════════════════════════════════════════════
_ENV_SH_CACHE: Dict[str, str] = {}


def _load_env_sh() -> Dict[str, str]:
    """解析 env 文件到缓存（支持 export 前缀、双引号、CRLF/LF）。"""
    if _ENV_SH_CACHE:
        return _ENV_SH_CACHE
    path = os.getenv("LLM_ENV_FILE", "")
    if not path or not os.path.isfile(path):
        return _ENV_SH_CACHE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip().rstrip("\r")  # 兼容 CRLF 混合
                if not line or line.startswith("#") or "export " not in line:
                    continue
                kv = line[7:].strip()
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                v = v.strip().strip('"').strip("'").rstrip("\r")
                if not v:
                    continue  # 空值不进缓存：语义等同未设置，_eg 会继续走默认值
                _ENV_SH_CACHE[k.strip()] = v
    except OSError as e:
        logger.warning("LLM env 文件读取失败: %s", e)
    return _ENV_SH_CACHE


def _eg(name: str, default: str = "") -> str:
    """优先 os.getenv，空则兜底 env 文件（防御未 source 场景）。"""
    v = os.getenv(name)
    if v:
        return v.rstrip("\r\n ")
    return _load_env_sh().get(name, default).rstrip("\r\n ")


# ════════════════════════════════════════════════════════════════════════════
# YAML 配置加载（可选依赖：缺 PyYAML 自动回退环境变量）
# ════════════════════════════════════════════════════════════════════════════
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # 可选依赖：缺失时自动回退 env.sh

LLM_YAML_PATHS = [
    os.getenv("LLM_ENDPOINTS_YAML"),
    # 与本模块同目录的 llm_endpoints.yaml 自动生效
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_endpoints.yaml"),
]
_YAML_CACHE_TTL = 30
_YAML_CACHE: Dict[str, Any] = {"path": None, "data": None, "ts": 0.0}


def _interp_env(v: Any) -> Any:
    """${VAR} / ${VAR:default} 环境变量插值（配置文件里不写死密钥）。"""
    if not isinstance(v, str):
        return v
    m = re.match(r"^\$\{([^}:]+)(?::([^}]*))?\}$", v.strip())
    if not m:
        return v
    return _eg(m.group(1), m.group(2) or "")


def _resolve_yaml_path() -> Optional[str]:
    for p in LLM_YAML_PATHS:
        if p and os.path.isfile(p):
            return p
    return None


def _load_yaml_config() -> Optional[Dict[str, Any]]:
    """读取 llm_endpoints.yaml（带 30s 缓存）。"""
    if yaml is None:
        return None
    now = time.time()
    path = _resolve_yaml_path()
    if not path:
        return None
    if _YAML_CACHE["path"] == path and now - _YAML_CACHE["ts"] < _YAML_CACHE_TTL:
        return _YAML_CACHE["data"]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _YAML_CACHE.update({"path": path, "data": data, "ts": now})
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning("llm_endpoints.yaml 解析失败: %s", e)
        return None


def _apply_yaml_globals(data: Dict[str, Any]) -> None:
    """YAML defaults / circuit_breaker 段覆盖模块级默认值。"""
    global LLM_CIRCUIT_COOLDOWN, LLM_HEALTH_ENABLED, LLM_RETRY_MAX, LLM_RETRY_BACKOFF
    global DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT, DEFAULT_MODEL
    cb = data.get("circuit_breaker") or {}
    if isinstance(cb, dict):
        if "enabled" in cb:
            LLM_HEALTH_ENABLED = str(cb["enabled"]).lower() in ("1", "true", "yes")
        if "cooldown_seconds" in cb:
            LLM_CIRCUIT_COOLDOWN = _to_int(cb["cooldown_seconds"])
    d = data.get("defaults") or {}
    if isinstance(d, dict):
        for k, v in (
            ("max_tokens", d.get("max_tokens")),
            ("temperature", d.get("temperature")),
            ("timeout", d.get("timeout")),
        ):
            if v is None:
                continue
            try:
                if k == "max_tokens":
                    DEFAULT_MAX_TOKENS = int(v)
                elif k == "temperature":
                    DEFAULT_TEMPERATURE = float(v)
                elif k == "timeout":
                    DEFAULT_TIMEOUT = int(v)
            except (ValueError, TypeError):
                logger.warning("忽略非法 defaults.%s=%r", k, v)
        try:
            if d.get("retry_max") is not None:
                LLM_RETRY_MAX = int(d["retry_max"])
        except (ValueError, TypeError):
            logger.warning("忽略非法 defaults.retry_max=%r", d.get("retry_max"))
        _rb = d.get("retry_backoff")
        if isinstance(_rb, list) and _rb:
            try:
                LLM_RETRY_BACKOFF = [int(x) for x in _rb]
            except (ValueError, TypeError):
                logger.warning("忽略非法 defaults.retry_backoff=%r", _rb)
        if d.get("model"):
            DEFAULT_MODEL = d["model"]


# ════════════════════════════════════════════════════════════════════════════
# 候选构造：YAML 优先，环境变量兜底
# ════════════════════════════════════════════════════════════════════════════
class Candidate(NamedTuple):
    url: str
    key: str
    model: str
    tag: str
    ssl_verify: bool = True


def _build_candidates_from_yaml(data: Dict[str, Any]) -> List[Candidate]:
    """从 YAML endpoints 列表构造候选。

    支持两种 api_key 格式：
      - str:  单 key（1 endpoint → 1 candidate）
      - list: 多 key（1 endpoint → N candidates，429/403 时按序轮换）
    """
    endpoints = data.get("endpoints") or []
    out: List[Candidate] = []
    seen: set = set()
    for ep in sorted(endpoints, key=lambda e: e.get("priority", 999) if isinstance(e, dict) else 999):
        if not isinstance(ep, dict):
            continue
        url = _interp_env(ep.get("api_url"))
        model = _interp_env(ep.get("model")) or DEFAULT_MODEL
        prio = ep.get("priority", 999)
        base_tag = ("P%d" % prio) if isinstance(prio, int) else str(ep.get("name", "X"))
        if not url:
            continue
        ssl_verify = bool(ep.get("ssl_verify", True))
        raw_key = ep.get("api_key")
        if isinstance(raw_key, (list, tuple)):
            keys = [k for k in (_interp_env(x) for x in raw_key) if k]
        else:
            k = _interp_env(raw_key)
            keys = [k] if k else []
        for i, key in enumerate(keys):
            tag = base_tag if len(keys) == 1 else f"{base_tag}-k{i + 1}"
            # 去重：同 URL+MODEL+KEY 三元组只保留第一条
            dedup_key = (url, model, key)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            out.append(Candidate(url, key, model, tag, ssl_verify))
    return out


def _eg_keys_pool(var_base: str) -> List[str]:
    """多 key 池：优先编号池（``{var_base}1`` … ``{var_base}11``），兜底单 KEY。

    参数 ``var_base`` 是**完整变量名的公共基名**（不含编号），
    例如 ``LLM_P1_KEY``（对应 ``LLM_P1_KEY1``…``LLM_P1_KEY11`` 与 ``LLM_P1_KEY``）。

    约定（契约，写死在此以免调用方误解）：

    1. 编号池：依次读 ``{var_base}1`` … ``{var_base}11``，遇首个空值即停
       （编号必须连续，空号后面不再探测）。
    2. 单 KEY 兜底：编号池全空时读 ``{var_base}`` 本身。

    ⚠️ **不接受以 ``_`` 结尾的前缀**（如 ``LLM_P1_``）。前缀形式无法唯一确定
    key 变量的完整名——旧实现用 ``var_prefix[:-1]`` 截断取 ``LLM_P1``，
    静默读到错误变量名 → 兜底失效且无任何报错。当前所有调用点都传完整基名
    （``prefix + "KEY"`` / ``f"LLM_P{n}_KEY"``），故该隐患已消除：截断分支
    直接删除，按前缀语义传参会立即读不到值而不是静默取错变量。
    """
    pool: List[str] = []
    for n in range(1, 12):
        v = (_eg(f"{var_base}{n}") or "").strip()
        if not v:
            break
        pool.append(v)
    if pool:
        return pool
    single = (_eg(var_base) or "").strip()
    return [single] if single else []


def _build_candidates_env() -> List[Candidate]:
    """环境变量兜底：统一 LLM_P{n}_URL/KEY/MODEL 命名，n = 1..12。

    P1 另兼容旧 LLM_API_*，P2 另兼容旧 LLM_FALLBACK_*。
    某槽缺 KEY 跳过该槽继续，不中断后续候选。
    """
    candidates: List[Candidate] = []
    seen: set = set()

    def _push(url: str, key: str, model: str, tag: str) -> None:
        url = url.strip()
        key = key.strip()
        if not (url and key):
            return
        dedup = (url, model, key)
        if dedup in seen:
            return
        seen.add(dedup)
        candidates.append(Candidate(url, key, model or DEFAULT_MODEL, tag, ssl_verify=not SSL_NO_VERIFY))

    # P1: 优先 LLM_P1_*，兼容旧 LLM_API_*
    for prefix, tag in (("LLM_P1_", "P1"), ("LLM_API_", "P1")):
        url = (_eg(prefix + "URL") or "").strip()
        if not url:
            continue
        model = (_eg(prefix + "MODEL") or "").strip()
        for i, key in enumerate(_eg_keys_pool(prefix + "KEY")):
            _push(url, key, model, tag if i == 0 else f"{tag}-k{i + 1}")
        if candidates:
            break

    # P2: 优先 LLM_P2_*，兼容旧 LLM_FALLBACK_*
    for prefix, tag in (("LLM_P2_", "P2"), ("LLM_FALLBACK_", "P2")):
        url = (_eg(prefix + "URL") or "").strip()
        if not url:
            continue
        model = (_eg(prefix + "MODEL") or "").strip()
        for i, key in enumerate(_eg_keys_pool(prefix + "KEY")):
            _push(url, key, model, tag if i == 0 else f"{tag}-k{i + 1}")

    # P3..P{N}: 统一 LLM_P{n}_URL/KEY（多 key 池）。缺 KEY 跳过该槽继续。
    for n in range(3, LLM_CANDIDATE_MAX + 1):
        url = (_eg(f"LLM_P{n}_URL") or "").strip()
        if not url:
            continue
        model = (_eg(f"LLM_P{n}_MODEL") or "").strip()
        keys = _eg_keys_pool(f"LLM_P{n}_KEY")
        if not keys:
            logger.warning("LLM_P%d_URL exists but no KEY, skip", n)
            continue
        tag = f"P{n}"
        for i, key in enumerate(keys):
            _push(url, key, model, tag if i == 0 else f"{tag}-k{i + 1}")

    return candidates


def _build_candidates() -> List[Candidate]:
    """按 P1→P{N} 优先级构造候选端点。YAML > 环境变量。"""
    ycfg = _load_yaml_config()
    if ycfg:
        _apply_yaml_globals(ycfg)
        yc = _build_candidates_from_yaml(ycfg)
        if yc:
            return yc
        logger.debug("YAML 无可用端点，回退环境变量")
    return _build_candidates_env()


def _backoff_wait(attempt: int) -> int:
    """取第 attempt 次重试的退避秒数，**绝不越界**。

    旧实现直接 `LLM_RETRY_BACKOFF[attempt]`：当 LLM_RETRY_MAX 大于退避数组长度
    （如 LLM_RETRY_MAX=5 而退避只有 1,3,5 三项）时抛 IndexError，
    在 except 分支里再抛会打断整条重试链——一次配置错误即让所有候选直接失败。
    越界时退化为「用数组最后一项」，不足时用 0（即不额外等待）。
    """
    if not LLM_RETRY_BACKOFF:
        return 0
    if attempt < 0:
        attempt = 0
    if attempt >= len(LLM_RETRY_BACKOFF):
        return LLM_RETRY_BACKOFF[-1]
    return LLM_RETRY_BACKOFF[attempt]


def _mask_key(key: str) -> str:
    """脱敏密钥：任何长度下**都不暴露完整明文**，且统一用 '...' 标记遮蔽段。

    长 key（>16）露头 8 尾 4；短 key 只露头部、其余整体遮蔽 ——
    原实现对 ≤4 字符的 key 取 key[:4]，等于把明文原样吐出（脱敏形同虚设）；
    对 5~16 字符的 key 只留 4 位头 + '***'，看不出被遮蔽了多少位。
    """
    n = len(key)
    if n == 0:
        return ""
    if n > 16:
        return key[:8] + "..." + key[-4:]
    # 短 key：留出的头部必须严格短于 key 本身，否则仍会泄露全文
    head = min(2, n - 1) if n > 1 else 0
    return key[:head] + "..." + "*" * max(3, n - head)


def list_endpoints() -> List[Dict[str, Any]]:
    """列出当前所有可用候选端点（诊断用）。返回脱敏视图（key 只露头尾）。"""
    cands = _build_candidates()
    return [
        {
            "tag": c.tag, "url": c.url, "model": c.model,
            "key": _mask_key(c.key),
        }
        for c in cands
    ]


# ════════════════════════════════════════════════════════════════════════════
# 多模态图片处理（Pillow 可选：>2MB 自动缩到 1600px + JPEG 85%）
# ════════════════════════════════════════════════════════════════════════════
def _detect_mime(image_path: str) -> str:
    ext = os.path.splitext(str(image_path))[1].lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/png")


def _detect_mime_from_bytes(raw: bytes) -> Optional[str]:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"GIF8":
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _read_image_bytes(image_path: str) -> Optional[bytes]:
    """读图片；>2MB 自动 Pillow 缩到 1600px + JPEG85%。

    读入上限 ``_MAX_IMAGE_READ_BYTES``：先取文件大小，超过上限直接拒绝，
    避免 ``read()`` 一次性把超大文件读进内存（调用方传大文件即 OOM）。
    """
    try:
        size = os.path.getsize(image_path)
    except OSError as e:
        logger.error("读图失败: %s", e)
        return None
    if size > _MAX_IMAGE_READ_BYTES:
        logger.error(
            "图片过大: %s (%d 字节 > 上限 %d 字节)",
            image_path, size, _MAX_IMAGE_READ_BYTES,
        )
        return None
    try:
        with open(image_path, "rb") as fh:
            raw = fh.read(_MAX_IMAGE_READ_BYTES + 1)
    except OSError as e:
        logger.error("读图失败: %s", e)
        return None
    # 兜底：读入期间文件可能变大（TOCTOU），超出上限即丢弃
    if len(raw) > _MAX_IMAGE_READ_BYTES:
        logger.error("图片过大（读入期间增长）: %s", image_path)
        return None
    if len(raw) <= 2 * _BYTES_PER_KIB * _BYTES_PER_KIB:
        return raw
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        img.thumbnail((1600, 1600))
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning("图片压缩失败，用原始 bytes: %s", e)
        return raw


# ════════════════════════════════════════════════════════════════════════════
# 拒答检测
# ════════════════════════════════════════════════════════════════════════════
_REFUSAL_PREFIXES = ("[advisor", "<yao_advisor>")
_REFUSAL_PHRASES = ("as an ai", "i cannot", "i can't", "i'm sorry", "i am unable")


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(t.startswith(p) for p in _REFUSAL_PREFIXES):
        return True
    return any(p in t[:400] for p in _REFUSAL_PHRASES)


# ════════════════════════════════════════════════════════════════════════════
# 结构化日志 + Prompt 缓存
# ════════════════════════════════════════════════════════════════════════════
def _log_structured(event: str, **kwargs: Any) -> None:
    if not LLM_STRUCTURED_LOGGING:
        return
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": "INFO",
        "event": event,
        **kwargs,
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


def _prompt_cache_key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    if LLM_PROMPT_CACHE_TTL <= 0:
        return None
    with _PROMPT_CACHE_LOCK:
        entry = _PROMPT_CACHE.get(key)
        if not entry:
            return None
        content, ts = entry
        if time.time() - ts > LLM_PROMPT_CACHE_TTL:
            _PROMPT_CACHE.pop(key, None)
            return None
        return content


def _cache_set(key: str, content: str) -> None:
    if LLM_PROMPT_CACHE_TTL <= 0:
        return
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE[key] = (content, time.time())
        # 简单上限（>500 条清最老 100 条）
        if len(_PROMPT_CACHE) > 500:
            oldest = sorted(_PROMPT_CACHE.items(), key=lambda x: x[1][1])[:100]
            for k, _ in oldest:
                _PROMPT_CACHE.pop(k, None)


# ════════════════════════════════════════════════════════════════════════════
# 核心请求：POST + 推理预算耗尽重试（单一事实源）
# ════════════════════════════════════════════════════════════════════════════
def _post_with_reasoning_budget(
    url: str, headers: Dict[str, str], payload: Dict[str, Any],
    timeout: int, ssl_verify: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """POST + 推理预算耗尽重试。

    reasoning / reasoning_content 是**思考过程**不是答案 —— 把它当答案返回，
    调用方会拿到 "Thinking Process: 1. …" 这种半成品。预算耗尽时直接把
    max_tokens 提到 CEILING 再试**一次**（「小预算逐级放大」会连续发多次
    请求，单次调用实测被拖到 90~141s，远超自动化可接受延迟）；仍空则返回
    空 content，由调用方判「无答案」降级。

    返回 (choice, msg, content)。
    """
    resp = _get_session().post(url, headers=headers, json=payload, timeout=timeout, verify=ssl_verify)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        usage = data.get("usage") or {}
        ct = usage.get("completion_tokens")
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        exhausted = (
            choice.get("finish_reason") == "length"
            and isinstance(ct, int) and isinstance(rt, int) and rt >= ct - 8
        )
        if exhausted and payload.get("max_tokens", 0) < REASONING_BUDGET_CEILING:
            remain = max(timeout, REASONING_BUDGET_MIN_TIMEOUT)
            logger.warning(
                "LLM reasoning budget exhausted (max_tokens=%d, reasoning_tokens=%s/%s), retry with %d",
                payload.get("max_tokens"), rt, ct, REASONING_BUDGET_CEILING,
            )
            payload["max_tokens"] = REASONING_BUDGET_CEILING
            resp = _get_session().post(url, headers=headers, json=payload,
                                       timeout=remain, verify=ssl_verify)
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
    return choice, msg, content


def _do_request(
    api_url: str, api_key: str, model: str, prompt: str,
    system_prompt: Optional[str], max_tokens: int, temperature: float,
    timeout: int, image_path: Optional[str] = None,
    ssl_verify: bool = True,
) -> str:
    """单次 LLM 请求（chat.completions 格式）。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_path:
        raw = _read_image_bytes(image_path)
        if raw is None:
            raise ValueError("image read failed: %s" % image_path)
        b64 = base64.b64encode(raw).decode("ascii")
        mime = _detect_mime_from_bytes(raw) or _detect_mime(image_path)
        user_content: Any = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    else:
        user_content = prompt
    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = api_url
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    _choice, msg, content = _post_with_reasoning_budget(url, headers, payload, timeout, ssl_verify)
    if not content:
        logger.warning(
            "LLM 无有效答案（content 为空；reasoning 有 %d 字但属思考过程，不当作答案）",
            len((msg.get("reasoning_content") or msg.get("reasoning") or "").strip()),
        )
        return ""
    return content


def _do_request_messages(
    api_url: str, api_key: str, model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    timeout: int, max_tokens: int,
    ssl_verify: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """单次 LLM 请求（messages + tools 格式，支持 tool_calls）。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    url = api_url
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    _choice, msg, content = _post_with_reasoning_budget(url, headers, payload, timeout, ssl_verify)
    # 有 tool_calls 时 content 本就为空（结构化输出正常形态），不告警
    if not content and not (msg.get("tool_calls") or []):
        logger.warning(
            "LLM 无有效答案（content 为空；reasoning 有 %d 字但属思考过程，不当作答案）",
            len((msg.get("reasoning_content") or msg.get("reasoning") or "").strip()),
        )
    tool_calls: List[Dict[str, Any]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        tool_calls.append({
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            },
        })
    return content, tool_calls


# ════════════════════════════════════════════════════════════════════════════
# 路由排序：健康端点优先，按平均延迟升序
# ════════════════════════════════════════════════════════════════════════════
def _sort_by_health(candidates: List[Candidate]) -> List[Candidate]:
    if not LLM_HEALTH_ENABLED or len(candidates) <= 1:
        return list(candidates)
    live, dead = [], []
    for c in candidates:
        (dead if _is_dead(c[0]) else live).append(c)

    def _avg_latency(c: Candidate) -> float:
        samples = _h(c[0]).latency_samples
        if samples:
            return sum(samples) / len(samples)
        last = _h(c[0]).last_latency
        return last if last is not None else float("inf")

    live.sort(key=_avg_latency)
    dead.sort(key=lambda c: _h(c[0]).fail_count)
    if dead:
        logger.info("LLM 路由(latency): 活 %d / 死 %d", len(live), len(dead))
    return live + dead


def _resolve_candidates(
    model: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    skip_primary: bool = False,
) -> List[Candidate]:
    """共享候选解析（call_llm / call_llm_with_tools 复用）。

    规则：显式覆盖 > 按模型名过滤（未命中退回全量）> 全量候选；
    skip_primary 时跳过 P1（首候选 tag 以 "P1" 开头，含多 key 展开 P1-k2）。
    返回空列表表示未配置任何端点。
    """
    if api_url and api_key:
        candidates = [Candidate(api_url, api_key, model or DEFAULT_MODEL, "override", ssl_verify=not SSL_NO_VERIFY)]
    elif model:
        all_c = _build_candidates()
        candidates = [c for c in all_c if c[2] == model]
        if not candidates:
            # 模型未找到，退回默认链
            logger.warning("model=%s 未在候选中找到，退回全量候选", model)
            candidates = all_c
    else:
        candidates = _build_candidates()

    if skip_primary and candidates and candidates[0][3].startswith("P1"):
        candidates = candidates[1:]
    return candidates


# ════════════════════════════════════════════════════════════════════════════
# 公开 API：call_llm
# ════════════════════════════════════════════════════════════════════════════
def call_llm(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    system_prompt: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    image_path: Optional[str] = None,
    model: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    deadline: Optional[float] = None,
) -> Optional[str]:
    """统一 LLM 调用（P1→P{N} 多级候选 + 多 key 轮换 + 熔断 + 硬错快跳）。

    deadline: 可选，time.monotonic() 截止时刻。给定后：
      * 每次尝试前检查剩余预算，不足抛 LLMDeadlineExceeded（调用方走兜底）
      * 单次 timeout 收敛为 min(timeout, 剩余预算)
      * 预算耗尽后不再切换候选
    这是「外层超时是硬约束、内层重试不得跨越」的落地。

    Args:
        prompt: 用户 prompt
        timeout: 单次请求超时（秒）
        system_prompt: 系统提示（None = 不发送 system 消息）
        max_tokens / temperature: 生成参数
        image_path: 可选，图片路径 → 构造多模态请求
        model: 可选，按模型名精确路由
        api_url / api_key: 可选，显式覆盖单端点（禁用 fallback）

    Returns:
        str: 模型返回内容；全部失败返回 None
    """
    if not prompt:
        return None

    # 候选构造：共享解析（显式覆盖 > 模型名 > 全量）
    candidates = _resolve_candidates(model=model, api_url=api_url, api_key=api_key)

    if not candidates:
        logger.warning("LLM 未配置（llm_endpoints.yaml 或 LLM_P1_URL/KEY 缺失）")
        return None

    # Prompt 缓存（有图 / 显式 override 不缓存）
    cache_key = None
    if not image_path and not api_url and not api_key:
        ycfg = _load_yaml_config() or {}
        effective_model = model or (ycfg.get("defaults") or {}).get("model") or DEFAULT_MODEL
        cache_key = _prompt_cache_key(effective_model, system_prompt or "", prompt)
        cached = _cache_get(cache_key)
        if cached:
            _log_structured("llm.cache.hit", key=cache_key[:16])
            return cached

    order = _sort_by_health(candidates)
    last_err: Optional[BaseException] = None

    for url, key, mdl, tag, ssl_verify in order:
        if _is_dead(url):
            logger.warning("LLM %s 熔断中（%ds 内曾失败），跳过", tag, LLM_CIRCUIT_COOLDOWN)
            continue
        for attempt in range(LLM_RETRY_MAX):
            # 总预算闸：每次尝试前校验收敛后的单次 timeout，
            # 预算不足立即向上抛，由调用方走确定性兜底。
            eff_timeout = timeout
            if deadline is not None:
                remain = deadline - time.monotonic()
                if remain <= 1:
                    logger.warning(
                        "LLM(%s) 总预算耗尽（剩余 %.1fs），停止重试与候选切换", tag, remain)
                    raise LLMDeadlineExceeded(
                        "LLM 总预算耗尽（剩余 %.1fs），调用方应走兜底" % remain)
                eff_timeout = _to_int(max(1, min(timeout, remain)))
            try:
                t0 = time.time()
                content = _do_request(
                    url, key, mdl, prompt, system_prompt,
                    max_tokens, temperature, eff_timeout, image_path,
                    ssl_verify=ssl_verify,
                )
                dt = time.time() - t0
                if content and _looks_like_refusal(content):
                    last_err = RuntimeError("refusal: %.80s" % content)
                    if attempt < LLM_RETRY_MAX - 1:
                        wait = _backoff_wait(attempt)
                        logger.warning("LLM(%s) 拒答重试 %d/%d: %ds", tag, attempt + 1, LLM_RETRY_MAX, wait)
                        time.sleep(wait)
                        continue
                    logger.warning("LLM(%s) 拒答耗尽重试: %.80s", tag, content)
                    break
                if content:
                    _mark_good(url, round(dt * _MILLISECONDS_PER_SECOND))
                    _log_structured("llm.call.success", tag=tag, latency_ms=round(dt * _MILLISECONDS_PER_SECOND), model=mdl)
                    logger.info("LLM 命中 %s (%.0fms, model=%s)", tag, dt * _MILLISECONDS_PER_SECOND, mdl)
                    if cache_key:
                        _cache_set(cache_key, content)
                    return content
                # 空响应：推理模型(content 在 reasoning_content)或瞬时空，
                # 不熔断端点（避免污染熔断计数器），直接切下一候选。
                logger.warning("LLM %s 返回空内容，切下一候选（不熔断）", tag)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_rate_limited(e) and "429" in str(e).upper():
                    # 429 → 多 key 场景下轮换 key（同 URL，下一把）
                    logger.info("LLM(%s) 429/限流 → 切下一候选（可能是下一把 key）", tag)
                    _mark_good(url)  # 端点还活着，只是这把 key 被限，不断链
                    break
                if _is_hard_error(e):
                    _mark_dead(url)
                    logger.warning("LLM(%s) 硬错误→熔断跳过: %s", tag, e)
                    break
                if attempt < LLM_RETRY_MAX - 1:
                    wait = _backoff_wait(attempt)
                    logger.warning("LLM(%s) 瞬错重试 %d/%d: %s, 等待 %ds", tag, attempt + 1, LLM_RETRY_MAX, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("LLM(%s) 耗尽重试: %s", tag, e)
        logger.warning("LLM 候选 %s 失败，切换下一候选", tag)

    logger.error("所有 LLM 候选均失败 (last_err=%s)", last_err)
    return None


# ════════════════════════════════════════════════════════════════════════════
# 公开 API：call_llm_with_tools
# ════════════════════════════════════════════════════════════════════════════
def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = _BYTES_PER_KIB,
    model: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    skip_primary: bool = False,
) -> Dict[str, Any]:
    """统一 LLM 调用（支持 tool_calls），多级候选 + 熔断 + 硬错快跳。

    Returns:
        {"content": str, "tool_calls": [...], "used_fallback": bool, "model": str}
    """
    if not messages:
        return {"content": "", "tool_calls": [], "used_fallback": False, "model": ""}

    candidates = _resolve_candidates(model=model, api_url=api_url, api_key=api_key,
                                     skip_primary=skip_primary)

    if not candidates:
        logger.warning("LLM 未配置")
        return {"content": "", "tool_calls": [], "used_fallback": False, "model": ""}

    cache_key = None
    if not api_url and not api_key:
        ycfg = _load_yaml_config() or {}
        effective_model = model or (ycfg.get("defaults") or {}).get("model") or DEFAULT_MODEL
        raw = json.dumps({"model": effective_model, "messages": messages, "tools": tools or []},
                         ensure_ascii=False, sort_keys=True)
        cache_key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        cached = _cache_get(cache_key)
        if cached:
            return json.loads(cached)

    order = _sort_by_health(candidates)
    for idx, (url, key, mdl, tag, ssl_verify) in enumerate(order):
        if _is_dead(url):
            continue
        try:
            t0 = time.time()
            content, tool_calls = _do_request_messages(url, key, mdl, messages, tools, timeout, max_tokens, ssl_verify=ssl_verify)
            dt = time.time() - t0
        except Exception as e:  # noqa: BLE001
            if _is_hard_error(e):
                _mark_dead(url)
                logger.warning("LLM(tools)(%s) 硬错误→熔断跳过: %s", tag, e)
            else:
                logger.warning("LLM(tools)(%s) 瞬错→切下一候选: %s", tag, e)
            continue

        need_tools = bool(tools)
        has_tc = bool(tool_calls)
        is_refusal = _looks_like_refusal(content)
        if need_tools:
            if not has_tc and (is_refusal or not content.strip()):
                logger.warning("LLM(tools)(%s) 未返回工具调用，切换下一候选", tag)
                continue
            _mark_good(url, round(dt * _MILLISECONDS_PER_SECOND))
            result = {"content": content, "tool_calls": tool_calls,
                      "used_fallback": idx > 0, "model": mdl}
            if cache_key:
                _cache_set(cache_key, json.dumps(result, ensure_ascii=False))
            return result
        if content and not is_refusal:
            _mark_good(url, round(dt * _MILLISECONDS_PER_SECOND))
            result = {"content": content, "tool_calls": [],
                      "used_fallback": idx > 0, "model": mdl}
            if cache_key:
                _cache_set(cache_key, json.dumps(result, ensure_ascii=False))
            return result
        logger.warning("LLM(tools)(%s) 拒答/空，切换下一候选", tag)

    logger.error("所有 LLM(tools) 候选均失败")
    return {"content": "", "tool_calls": [], "used_fallback": False, "model": ""}


# ════════════════════════════════════════════════════════════════════════════
# 公开 API：call_llm_json（强制 JSON Schema 结构化输出）
# ════════════════════════════════════════════════════════════════════════════
def call_llm_json(
    prompt: str,
    schema: Dict[str, Any],
    system_prompt: Optional[str] = None,
    schema_name: str = "emit",
    schema_description: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = 2000,
    temperature: float = DEFAULT_TEMPERATURE,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """按 JSON Schema 取结构化结果（用 tool calling 强制约束，不用提示词求模型自觉）。

    为什么用 tool calling 而不是 response_format：
      * response_format=json_object 在推理型端点上实测 18.8s 后
        finish_reason=length、content 为空 —— 推理 token 把预算吃光，
        对 JSON 输出无任何约束力。
      * response_format=json_schema 部分上游直接拒：HTTP 400。
      * tool_choice 强制单函数**实测可用**：1.5s / 94 completion tokens /
        arguments 为合法 JSON（同一任务 response_format 要 18.8s 且失败）。
    因此本函数把 schema 包成唯一一个 function，强制模型必须调用它。

    Args:
        prompt: 用户输入
        schema: 目标 JSON Schema（object 类型）
        system_prompt: 可选的 system 指令
        schema_name / schema_description: 工具名与说明（仅影响提示，不改变解析）

    Returns:
        解析后的 dict；失败（无 tool_call / arguments 非 JSON / 全部候选失败）
        返回 None。调用方应自行降级（如走规则兜底），本函数不抛异常。
    """
    if not prompt:
        return None
    if not isinstance(schema, dict) or not schema:
        raise ValueError("schema 必须是非空 dict（JSON Schema object）")

    tool = {
        "type": "function",
        "function": {
            "name": schema_name,
            "description": schema_description or "按给定 schema 输出结构化结果",
            "parameters": schema,
        },
    }
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    res = call_llm_with_tools(
        messages=messages,
        tools=[tool],
        timeout=timeout,
        max_tokens=max_tokens,
        model=model,
        # call_llm_with_tools 只在「有 tool_call」时返回，无需额外 force；
        # 若上游忽略 tools 则返回 content，会被下面判为失败。
    )
    calls = (res or {}).get("tool_calls") or []
    for tc in calls:
        fn = (tc or {}).get("function") or {}
        if fn.get("name") != schema_name:
            continue
        raw = fn.get("arguments") or ""
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as e:
            logger.warning("call_llm_json: tool arguments 非合法 JSON: %s", e)
            continue
        if isinstance(parsed, dict):
            return parsed
        logger.warning("call_llm_json: tool arguments 非 object（%s）", type(parsed).__name__)
    if res and res.get("content"):
        logger.warning("call_llm_json: 模型未按 schema 调用工具（返回文本 %.120s）",
                       str(res["content"]))
    return None
