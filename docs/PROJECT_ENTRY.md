# Project Entry

更新时间：2026-06-30

本文件是给后续 agent/工程师的短入口。详细实验历史仍以根目录 `goal.md`、`docs/PROJECT_REVIEW.md`、`docs/DECISIONS.md` 和 `reports/date_generalization/` 为准。

## 目标

本项目是本地 A 股研究 Agent。它面向用户输出清晰、可执行、可解释的研究辅助型操作建议，例如试探买入、加仓、持有、减仓、卖出复核、等待或补数据，并必须给出仓位/阈值、证据、反证、失效条件和复查条件。

项目边界：

- 不自动交易，不接券商接口，不替用户下单。
- 不承诺收益，不写目标价必达、稳赚、必涨或无风险。
- 可以使用用户合法授权的公开、会员、付费或标准化数据源，但 token/key 只能由代码安全读取，不得写入日志、报告、prompt、ledger 或 diff。
- 回测 evidence pack 不得包含未来收益、GT、未来事件或未披露财报。

## 当前工作流

正式执行前必须先判断用户任务属于哪类：

1. 单支股票调研/盯盘。
2. 已关注股票的风险复核。
3. 多股候选对比。
4. 盘中/实时盯盘。
5. 策略研究/组合回测。

主线工作流：

1. 路由用户意图，必要时先用选择题澄清。
2. 读取 as-of 数据：量价/K线、新闻公告、财报/业绩事件、同行/地域、筹码、BookSkill、memory/RAG。
3. 本地工具层给出可审计信号或动作草案，例如 P0 single-stock scorer、action-label tool、P1 ranker-anchor、风险队列。
4. DeepSeek Agent 做审计、融合、反证解释和用户转译；Agent 不应替代工具层成为主预测器。
5. 输出明确操作建议、仓位上限、买入/加仓触发、减仓/卖出触发、复查条件和数据缺口。
6. 回测/实验后更新 memory ledger、能力报告和必要的策略边界。

## 数据源与权限边界

核心输入：

- 行情/量价：mootdx/pytdx、BaoStock、AKShare/efinance、本地缓存。
- 新闻/公告：官方披露优先，公开聚合源补充；Tushare Pro/Wind/Choice/iFinD/同花顺会员等授权数据可作为 paid_standardized 离线缓存。
- 财报/业绩事件：必须保留报告期、披露日或 available_at；缺披露日不得进入 walk-forward 判断。
- 同行/地域/筹码：主要来自本地标准化缓存和 Tushare Pro 缓存。
- BookSkill：必须来自真实 OCR/文本整理，带书名、章节、页码/OCR_PAGE、策略 ID 和适用/失效条件。

凭证边界：

- `ds_api.txt`、`tushare_token.txt` 是本地敏感文件，只能由代码读取。
- 不读取明文、不打印、不复制、不写入报告。
- 交付给用户或迁移项目时，应改用用户本地凭证或安全注入方式。

## 当前最佳状态

当前项目不是“全市场主动买入已稳定达标”的状态。

- P0 单支盯盘：强黄灯 MVP。重点是小仓试探、持有、减仓/卖出复核、等待和补数据，适合作为用户主交互。
- P1 多股候选对比：黄灯可用。默认 `candidate_comparison_ranker_anchor_v2`，ranker-anchor 排序优先，Agent 只审计硬反证和解释差异。
- P2 组合/截面策略：辅助研究工具。`rev+chip_core` 是默认组合 ranker 参考，但 H2026/latest 仍未证明广义主动买入泛化。
- 新闻/财报/BookSkill/K线/同行/筹码：当前更多是 checklist、风险复核、软缺口/硬反证分叉和解释层；升权必须经过 fresh panel、ablation、leakage audit 和 active exposure 检查。

最近 action-label 方向的结论：

- corrected Flash v2 说明 action-label tool 能减少 no-tool 过度防守。
- balanced panel Flash v1 显示工具在强 buy/add 和 small-hold 分支有帮助，但 wait/reduce 分支需要 cap guard，不能直接升为 broad default。

## Agent 如何恢复

新 agent 按这个顺序恢复：

1. 读 `AGENTS.md`、`docs/START_HERE.md`、本文件、根目录 `goal.md`。
2. 看当前产品状态：`reports/date_generalization/final_capability_report.md`、`reports/date_generalization/final_product_workflow.md`、`reports/date_generalization/final_product_readiness_audit_v1.md`。
3. 看工作流和目录：`docs/WORKFLOW.md`、`docs/DATA_FLOW.md`、`docs/DIRECTORY_MAP.md`、`docs/CLEANUP_INVENTORY.md`。
4. 看结构化记忆：`memory/strategy_experience_ledger.csv`、`memory/book_skill_adaptation_ledger.csv`、`memory/news_world_model_ledger.csv`、`memory/ablation_findings_ledger.csv`、`memory/failure_case_ledger.csv`。
5. 先跑轻量工程 gate，再做任何昂贵实验：

```bash
/data/cyx/1030/stock/.conda/stock-agent/bin/python scripts/audit_tool_adoption_contract.py
/data/cyx/1030/stock/.conda/stock-agent/bin/python scripts/audit_user_actionability_contract.py
/data/cyx/1030/stock/.conda/stock-agent/bin/python scripts/audit_final_product_readiness.py
```

不要把 low exposure、全现金/全等待、单一小样本、单一日期块、或 prompt-only 改善写成策略能力提升。
