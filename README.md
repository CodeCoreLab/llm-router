# llm-router

> 单文件、可移植的多端点 LLM 路由客户端。P1→P12 优先级候选、熔断、多 key
> 轮换、推理预算治理、强制 JSON Schema 结构化输出——全部在一个 py 文件里。

产品介绍页（含架构图与特性总览）：[`docs/llm-router-产品页.html`](docs/llm-router-产品页.html)

## 为什么需要它

接多个 LLM 供应商是常态：主力模型限流要切兜底、自建网关要排最后、
推理型模型把 token 烧在思考链上返回空内容、`response_format=json_object`
在很多上游根本不生效。这些问题每个接入方都要重踩一遍。

llm-router 把这些实战沉淀收敛成**一个可复用的单文件模块**：

- **候选路由**：P1→P12 按优先级遍历，硬错（连接/401/403/404）立即切下家，瞬错（超时/429/5xx）单端指数退避重试
- **熔断**：判死端点 300s 冷却自动跳过；恢复后按**实测平均延迟**重排健康端点
- **多 key 轮换**：一个 endpoint 配一列 key，429/403 自动切下一把，端点不掉链
- **推理预算治理**：推理模型烧光 budget 返回空 content 时，自动放大 max_tokens 重试一次（上限可配）
- **结构化输出**：`call_llm_json()` 用强制 tool calling 约束 JSON Schema——不靠提示词求模型自觉
- **deadline 预算闸**：外层超时是硬约束，内层重试不得跨越——预算耗尽抛 `LLMDeadlineExceeded` 让调用方走兜底
- **Prompt 缓存**：进程内 TTL 缓存，重复分析任务不重复付费

## 设计底线

| 底线 | 说明 |
|---|---|
| 单文件 | `llm_router.py` 拷走即用，无包结构要求 |
| 依赖最少 | 仅 `requests`（PyYAML 可选、Pillow 可选、缺了自动降级） |
| 密钥不落盘 | 配置里用 `${VAR}` 插值，key 走环境变量；`list_endpoints()` 诊断视图自动脱敏 |
| 纯客户端 | 无服务端、无守护进程、不联网探测，失败只影响当次调用 |
| 离线可测 | 单元测试全部离线跑，不打真实端点 |

## 快速开始

```bash
pip install requests          # 唯一必装依赖
```

### 1. 环境变量方式（最快）

```bash
export LLM_P1_URL="https://api.example-llm.com/v1"
export LLM_P1_KEY="sk-xxx"
export LLM_P1_MODEL="my-model"
export LLM_P2_URL="https://api.backup-llm.com/v1"   # 可选兜底
export LLM_P2_KEY="sk-yyy"
export LLM_P2_MODEL="my-model-lite"
```

### 2. YAML 方式（推荐，支持多 key 池）

```bash
cp config/llm_endpoints.example.yaml config/llm_endpoints.yaml
# 编辑后放到 llm_router.py 同目录，或设 LLM_ENDPOINTS_YAML 指向它
```

### 3. 调用

```python
from llm_router import call_llm, call_llm_json, list_endpoints

# 按优先级顺序路由
text = call_llm("帮我总结这段日志的异常点", system_prompt="你是运维专家")

# 按模型名精确路由
text = call_llm("翻译成英文", model="my-model")

# 多模态（可选 Pillow 压缩 >2MB 图片）
text = call_llm("这张监控图里有什么异常", image_path="shot.png")

# 强制 JSON Schema 结构化输出
result = call_llm_json(
    "提取告警的严重等级与影响面",
    schema={
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
            "impact": {"type": "string"},
        },
        "required": ["severity", "impact"],
    },
)

# 诊断：查看当前候选链（key 自动脱敏）
print(list_endpoints())
```

### 4. deadline 预算闸（长任务防拖死）

```python
import time
from llm_router import call_llm, LLMDeadlineExceeded

deadline = time.monotonic() + 60  # 外层只给 60s
try:
    text = call_llm("……", deadline=deadline)
except LLMDeadlineExceeded:
    text = "（LLM 超时，走规则兜底）"  # 调用方确定性兜底
```

## 配置参考

完整示例见 [`config/llm_endpoints.example.yaml`](config/llm_endpoints.example.yaml)。环境变量总览：

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_ENDPOINTS_YAML` | YAML 配置路径 | 模块同目录自动发现 |
| `LLM_ENV_FILE` | env 兜底文件（解析 `export K="V"`） | 不加载 |
| `LLM_RETRY_MAX` | 单端点重试次数 | 3 |
| `LLM_RETRY_BACKOFF` | 退避秒数（逗号分隔） | 1,3,5 |
| `LLM_HEALTH_ENABLED` | 熔断开关 | true |
| `LLM_CIRCUIT_COOLDOWN` | 熔断冷却秒数 | 300 |
| `LLM_PROMPT_CACHE_TTL` | 缓存 TTL 秒（0 关闭） | 300 |
| `LLM_STRUCTURED_LOGGING` | 结构化 JSON 日志 | 关 |
| `SSL_NO_VERIFY` | 全局禁证书校验（自签场景） | 关 |
| `LLM_REASONING_BUDGET_CEILING` | 推理预算放大上限 tokens | 16000 |
| `LLM_REASONING_BUDGET_MIN_TIMEOUT` | 放大重试最小超时秒 | 180 |

## 测试

```bash
cd llm-router && python -m pytest tests/ -v
```

测试全部离线运行（mock HTTP），覆盖候选构造、去重、多 key 展开、
拒答检测、错误分类、缓存、env 插值与熔断状态机。

## 已知边界

- 仅支持 OpenAI `chat.completions` 兼容协议（绝大多数网关与本地推理框架均兼容）
- 熔断状态为进程内态，多进程部署各自治愈（无跨进程共享）
- Prompt 缓存按 `(model, system_prompt, prompt)` 精确匹配，不做语义去重
