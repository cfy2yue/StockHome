# Bugs And Fixes

本文件记录会影响实验连续性、数据安全或工作流可复现的非平凡问题。

## 2026-06-26 Workflow YAML Indentation

问题：新增 `quantitative_kline_channel` 后，`config/agent_workflow_strategy.yaml` 中 `prompt_hygiene_policy` 多缩进了一级，导致 `yaml.safe_load` 失败。全量 Python 单元测试没有覆盖配置 YAML 语法，直到 preflight 才发现。

影响：`run_preflight()` 返回 false，后续 DeepSeek round 脚本会被 preflight 阻断，避免在坏配置下继续实验。

修复：恢复 `prompt_hygiene_policy` 到 `hard_guards.memory` 正确层级，并重新运行 YAML 解析、preflight 和定向测试。

验证：

- YAML 解析通过，`quantitative_kline_channel.status=observe_candidate`。
- `reports/date_generalization/preflight_check.md` 显示 `ok=True`。
- 定向测试 `21 passed`。
