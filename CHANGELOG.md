# 变更日志 — LLM Router（llm-router）

本文件记录本项目的变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

## [1.0.1] - 2026-11-06
### 修复

- **图片读入大小上限常量补齐**：`_MAX_IMAGE_READ_BYTES` 未定义导致调用即报错
- **重试退避取值不再越界**：退避数组长度小于重试次数时改为回落到末项，不再抛 `IndexError`
- **单 KEY 兜底 helper 契约收敛**：`_eg_keys_pool` 原接受「前缀」入参并用截断取变量名（`LLM_P1_` → `LLM_P1`，与真实的 `LLM_P1_KEY` 无关），静默取到空值导致兜底失效；现强制传完整变量基名，前缀语义入参会立即读不到值而非静默取错
## [1.0.0] - 2026-11-01

### 新增

- P1→P12 优先级候选路由：YAML / 环境变量声明，多 key 池展开，URL+MODEL+KEY 三元组去重，缺槽跳过不中断
- 熔断状态机：硬错（连接 / 401 / 403 / 404）判死 300s 冷却，恢复后按实测平均延迟重排健康端点
- 多 key 轮换：429 / 403 自动切下一把 key，端点不熔断不掉链
- 推理预算治理：预算耗尽（`finish_reason=length` 且 reasoning 占满）自动放大 `max_tokens` 重试一次，上限可配
- deadline 预算闸：外层超时硬约束，内层重试不跨越，耗尽抛 `LLMDeadlineExceeded`
- `call_llm_json()` 强制 JSON Schema 结构化输出（tool_choice 约束，非提示词自觉）
- 进程内 TTL Prompt 缓存（默认 300s）、拒答检测、多模态图片（可选 Pillow 压缩）

### 变更

- 工程形态：单文件 `llm_router.py`，唯一必装依赖 `requests`（PyYAML / Pillow 可选缺失自动降级）
- 36 项离线单元测试全部通过（mock HTTP，不打真实端点）
- 凭据经 `${VAR}` 环境变量注入，配置与诊断视图零明文
