# Cleanup Inventory

更新时间：2026-06-30

本次清理遵循 `/data/cyx/1030/scLatent/prompts/STOCK_DIRECTORY_ORGANIZATION_PROMPT.md`：文档化优先，只删除明确可复现噪声，不删除数据、报告、ledger、策略卡、配置或回测输出。

## Keep

必须保留：

- `AGENTS.md`、`goal.md`、`README.md`、`docs/START_HERE.md`、`docs/PROJECT_ENTRY.md`。
- `src/`、`scripts/`、`tests/`、`config/`、`examples/`。
- `book_skills/` 和 `data/book_processed/`，包括 OCR/source/strategy cards。
- `memory/*.csv` 和 `memory/*.md`，尤其 strategy/book/news/ablation/failure ledgers。
- `data/` 下的 as-of 缓存、回测缓存、live watch 缓存。
- `reports/` 和 `runs/` 中的实验报告、metrics、evidence pack、decision ledger、invalid ledger、usage summary、logs。
- `ds_api.txt`、`tushare_token.txt`：本地敏感凭证文件。保留但不得读取明文或写入任何文档。
- `.conda/stock-agent/`：服务器本地运行环境。体量较大但不是当前清理对象。

## Archive Candidates

这些文件/目录不建议现在删除，但可在后续做归档：

- `docs/archive/`：历史目标和旧计划已经归档，保留。
- `reports/date_generalization/archive_raw_rounds/`：历史 raw round 证据，已在 archive 子目录内，保留。
- 早期 `reports/date_generalization/*dryrun*`：很多 dry-run evidence/decision 文件用于证明泄漏隔离和 schema 稳定，不能直接删；后续可按 `cleanup_manifest` 做二级归档。
- `PROJECT_BRIEF.md`、`MEMORY.md`：有历史价值，但部分边界和当前 `AGENTS.md`/`goal.md` 不完全一致。建议后续归档为 historical context，主入口改用 `docs/PROJECT_ENTRY.md`。

## Safe Delete

本次可安全删除的可复现噪声：

- `.pytest_cache/`
- `scripts/__pycache__/`
- `src/__pycache__/`
- `src/agent_training/__pycache__/`
- `src/analysis/__pycache__/`
- `src/backtest/__pycache__/`
- `src/data/__pycache__/`
- `src/reports/__pycache__/`
- `src/world_model/__pycache__/`
- `tests/__pycache__/`

说明：这些都是 Python/test 本地缓存，可由下一次运行自动重建。

## Not Deleted Even If Empty

发现很多空 `*_invalid_outputs.jsonl` 或 dry-run `*_decision_ledger.jsonl`。这些文件虽然是 0 字节，但代表“该轮无 invalid 输出”或“dry-run 未调用模型”的实验审计证据，因此不在本次删除范围。

## Unknown / Needs User Decision

需要用户或项目负责人确认后再动：

- 是否把 `.conda/` 从可迁移交付包中排除，只保留 `environment.yml` 和 `requirements.txt`。当前服务器运行依赖它，不能删。
- 是否压缩/归档 `reports/date_generalization/` 的 4600+ 个文件。当前保留全部证据；后续可用 manifest 按“最终报告、关键 evidence、失败实验、raw round”分层归档。
- 是否把旧 `PROJECT_BRIEF.md`、`MEMORY.md` 的历史边界统一到当前口径。当前只记录冲突，不批量改历史。
- 是否清理 `deliverables/stock_agent_user_light/`。用户已说暂时不管轻量版，交给 cursor；本次保留。
- 是否把 `runs/` 中已完成的下载/Flash 任务归档。需要逐个检查 `EXIT_CODE`、`RUN_STATUS.md` 和对应报告引用。

## Current Organization Risks

- 文档口径有历史漂移：早期文件仍写“只输出四类研究分级”或“不输出确定买卖”，当前 `AGENTS.md`/`goal.md` 已允许明确操作建议，但必须带仓位、阈值和边界。
- `reports/date_generalization/` 文件数量大，短期保留更安全；长期需要按实验 family 和 promotion status 归档。
- 旧 prompt-only/DS 大 round 很多，后续 agent 容易被历史噪声带偏；恢复时应先看 `docs/PROJECT_ENTRY.md`、`goal.md` 顶部和 final reports，而不是从旧报告随机挑结论。
