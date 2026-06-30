# Start Here

本项目是 A 股研究 Agent，目标是给用户清晰、可执行、可解释的研究辅助型操作建议。它可以输出买入、卖出、加仓、减仓、持有、等待、补数据等建议，并给出仓位/触发阈值、反证和复评条件；它不会自动下单，不接券商接口，不承诺收益。

## 先看这几份

1. `goal.md`：当前唯一推进目标和边界。
2. `AGENTS.md`：stock agent 的硬规则。
3. `docs/GIT_AND_COLLABORATION.md`：Git、CC、Codex 协作和文件所有权规则。
4. `docs/GITHUB_FILE_MAP.md`：高信号文件和 GitHub URL 映射。
5. `docs/USER_GUIDE.md`：给真实用户看的简版说明。
6. `docs/PROJECT_ENTRY.md`：给后续 agent/工程师的短入口，含目标、当前状态、恢复顺序。
7. `docs/DIRECTORY_MAP.md`：目录地图，区分源码、配置、数据、报告、memory 和缓存。
8. `docs/HANDOFF.md`：给后续 agent 或工程师的交接说明。

服务器本地证据在 `reports/`、`runs/`、`memory/` 和 `docs/local_archive/` 中，
这些目录默认不进入 GitHub。CC 本地 clone 后如果需要看这些证据，应通过
`ssh cyx-server-proxy-cfy` 到服务器读取，不要在 Windows 本地伪造或重建。

## 服务器与 GitHub

- 服务器目录：`/data/cyx/1030/stock`
- GitHub 目标仓库：`https://github.com/cfy2yue/StockHome`
- 服务器登录：`ssh cyx-server-proxy-cfy`
- Windows/CC 可 clone GitHub 仓库做文档、审计、代码 review 和小型本地测试；需要服务器缓存、凭证、长回测或大数据的任务必须交给服务器/Codex。

## 当前产品形态

- P0 单支盯盘：`strong_yellow_mvp`。这是当前最成熟的主线，默认交付“小仓试探/持有/减仓复核”的明确操作建议。`PPS-Q-017 + softgap v2` 和 general-channel small-entry 均已有三面板 Flash 支持；Pro 可用但未超过 Flash。广义全市场 active-buy 仍未稳定达到 `0.60/0.65`，不作为当前交付承诺。
- P1 多股候选对比：`default_ready_yellow`。适合回答“同领域或跨领域几支股票里，哪 1-2 支更值得操作；各自应等待、试探买入、持有还是回避”。
- P2 组合/截面策略：辅助研究工具，不作为用户主交互，也不覆盖 P0/P1 的结论。
- 最新非价格风险覆盖层只作为防错杀/二次确认 checklist：同行/地域弱、财报软缺口不得机械剔除；高风险新闻需要确认。三面板验证后，broad overlay 不再作为 P0 单支默认输入，只在 P1/P2 候选池研究中默认可见。

## 用户交互必须先确认任务

用户问题不清楚时，先给选择题：

1. 单支股票调研/盯盘
2. 已关注股票的风险复核
3. 多股候选对比
4. 盘中/实时盯盘
5. 策略研究/组合回测

确认任务后再调用对应工作流。不要把组合回测指标套到单支盯盘，也不要把单支个案包装成组合 alpha。

## 明确建议怎么表达

用户需要清楚结论，第一句必须是操作建议：

- 买入/试探买入：给仓位上限、证据组合、止损/复评阈值。
- 加仓/提高仓位：给提高条件、仓位区间和撤回条件。
- 持有/继续持有：给继续持有条件、跌破/反证处理。
- 减仓/卖出：给触发风险、退出阈值和重新评估条件。
- 等待/暂不操作：写清什么条件转为买入/加仓，什么条件转为减仓/卖出。
- 信息不足：列出关键缺口，补齐前不硬做方向判断。

每次回答必须给：明确操作建议、仓位/阈值、依据、反证、买入/加仓触发、减仓/卖出触发、下一次复查条件。研究分级可以作为辅助标签，不能代替建议。

## 常用入口

用户向导：

```bash
python -m src.user_wizard
```

盘中/实时盯盘单次复核：

```bash
python scripts/run_live_watch_session.py --code 000001 --name 平安银行 --interval-seconds 1200 --max-iterations 1
```

重新生成 Markdown 用户手册：

```bash
python scripts/build_final_user_manual.py
```

检查最终交付状态：

```bash
python scripts/audit_final_product_readiness.py
```

## 不要碰这些

- 不打印、不复制、不写入日志或报告任何 API key/token。
- 不删除、不移动、不覆盖原始书籍 PDF、BookSkill source、关键本地缓存和回测证据。
- 不把未来收益、GT 字段或后验标签放进 evidence pack。
- 不把低 active exposure 或全观察导致的好看结果当成真实选股能力。
- 不把 `reports/date_generalization/` 中的空 invalid/dry-run ledger 当作垃圾直接删除；它们常用于证明无 invalid 或 dry-run 未调用模型。
- 不删除 `.conda/stock-agent/`，除非用户明确要求重建环境。

## 目录组织状态

2026-06-30 已完成一次保守组织检查：

- 新增 `docs/PROJECT_ENTRY.md`、`docs/DIRECTORY_MAP.md`、`docs/CLEANUP_INVENTORY.md`。
- 清理范围只允许 pytest/Python 字节码缓存等可复现噪声。
- `reports/`、`data/`、`memory/`、`book_skills/`、`runs/` 先全部保留；后续归档必须先有 manifest 和用户确认。
- 发现历史文档口径漂移：旧 `PROJECT_BRIEF.md`、`MEMORY.md` 部分内容仍写旧边界；当前用户输出边界以 `AGENTS.md`、`goal.md` 和本文件为准。

## 滚动维护

每 1-3 个月或行情明显切换后：

1. 拉新数据并重建 as-of feature store。
2. 先跑 leakage、coverage、source_ref、披露日和凭证安全审计。
3. 复跑 P0/P1 deterministic 指标。
4. Flash 做低成本 smoke，必要时 Pro 做最终确认。
5. 更新 memory、BookSkill adaptation、能力报告和用户手册。
