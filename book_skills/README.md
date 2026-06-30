# Book Skill Active Set

本目录只保留 Book Skill 的来源、约束和少量可进入 Agent 证据包的压缩入口。旧版 deep dive、自动生成草稿、辅助卡和大体积 OCR 中间物已归档到项目外，避免后续 Agent 在过多中间材料里走偏。

## 默认进入 Evidence Pack

- `grounded_skill_cards.yaml`：已按回测/适配观察压缩后的 grounded skill，供 evidence pack 优先引用。
- `invalid_conditions.md`：Book Skill 不适用、失效或需要反证的条件。
- `core/low_confidence_or_deferred_cards.yaml`：低置信、暂缓或否决条目，不得进入正式判断。
- `source_manifest.yaml`、`source_audit_report.md`、`coverage_report.md`、`core/source_priority.md`：来源、审计、覆盖和优先级。

Agent/DeepSeek 默认只读取上面这些文件。默认 evidence pack 不允许整文件读取大策略表，也不允许把未验证 deep dive、草稿或自动卡直接塞进 prompt。

## Reference Only

- `strategy_cards.yaml`：正式策略卡总表，保留给人工复核、离线 grounding、检索 strategy_id 和重建 grounded cards。
- `core/macro_principles.md`：宏观、趋势、风险和市场结构原则总表。
- `core/quantitative_rules.md`：可量化判据、阈值和触发条件总表。

这些文件仍必须保留来源字段，且通过 preflight 校验；但默认不进入 DeepSeek evidence pack。只有在明确进行 Book Skill grounding、人工复核或重新生成 `grounded_skill_cards.yaml` 时才能读取。

## 来源与审计

- `source_manifest.yaml`：书籍来源、处理状态和可追溯信息。
- `source_audit_report.md`：来源审计摘要。
- `coverage_report.md`：策略卡覆盖情况。
- `core/source_priority.md`：来源优先级。

## 归档位置

清理时间：2026-06-25。

已归档但未删除的材料位于：

`/data/cyx/1030/stock_archive/cleanup_20260625/book_skills/`

归档内容包括旧版 01-10 skill 文档、deep dive、自动映射、辅助策略卡和重复的 core 策略卡。

## 读取策略

读取边界固化在 `config/agent_workflow_strategy.yaml`：

- `default_evidence_pack_files`：Agent 默认可读文件。
- `reference_only_files`：只供人工复核、离线 grounding 或重建 grounded cards。
- `allowed_active_files`：项目内允许保留的 Book Skill 文件白名单。
