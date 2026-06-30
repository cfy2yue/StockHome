# 日期泛化项目状态审计与下一步门禁

更新时间：2026-06-25

## 结论

当前 goal 方向合理，可以继续推进，但不应直接扩大规模或宣称策略已完成训练。项目已经具备时间分块、基础回测、决策卡 schema、DeepSeek 客户端、Book Skill 和新闻向量雏形；但核心训练主线仍有三处必须先补：

1. DeepSeek Agent 尚未真实接入主训练决策层；当前 `full_agent` 主要是本地确定性 policy runner 替身。
2. 新闻通道在 2023-2025 覆盖为 0，不能证明新闻 Agent 增量，也不能支撑日期泛化能力。
3. `H2026_1` 在当前日期 2026-06-25 尚未完整结束，不能把 2026-06-30 作为已完成验收窗口。

因此，下一步目标应是“修正训练闭环与验收口径”，而不是先冲更大的股票池或更多 epoch。

## 当前项目主线

项目主线是 A 股研究辅助型操作建议系统，不自动交易、不接券商、不下单、不承诺收益。对用户输出必须先给买入、卖出、加仓、减仓、持有、等待或补数据建议，并配套仓位/阈值、证据、反证和复评条件；研究分级只作为辅助标签。

当前关键文件：

- 目标文档：`docs/DATE_GENERALIZATION_GOAL.md`
- DeepSeek 接入说明：`docs/DEEPSEEK_AGENT_SETUP.md`
- DeepSeek 配置：`config/deepseek_agent.yaml`
- DeepSeek 客户端：`src/agent_training/deepseek_client.py`
- 决策卡 schema：`src/agent_training/decision_card.py`
- 本地 policy runner：`src/agent_training/policy_runner.py`
- 主训练脚本：`scripts/run_agent_strategy_training_rounds.py`
- 当前主报告目录：`reports/date_generalization/`

## Goal 合理性评估

合理：

- 用半年块 walk-forward 训练和验证，符合时间安全要求。
- 明确 Python 只做证据层，最终决策应由 Agent 综合多通道完成。
- 明确目标是日期泛化，而不是某个近期窗口最高收益。
- 要求 ablation、新闻覆盖审计、Book Skill 适配观察和反证记录。
- 要求未达标如实报告，不换口径。

需要修正：

- 验收阈值可以保留，但报告必须同时说明覆盖不足和未到期窗口，不得把未成熟数据算作最终通过。
- DeepSeek Agent 真实调用前，`full_agent` 指标只能标注为“本地规则代理”，不能标注为 DeepSeek Agent 实绩。
- 新闻通道覆盖不足前，不能用 news ablation 证明新闻通道有效。
- 如果人工看过结果后调权重，应作为规则层过拟合风险记录进 `memory/strategy_experience.md`。

## 当前回测/训练状态

最新一次主训练入口已跑通：

```text
python scripts/run_agent_strategy_training_rounds.py
```

输出目录：`reports/date_generalization/`

当前 `agent_policy_metrics.csv` 显示 `full_agent` 在各验证块 20 日正收益率均超过 0.60，`H2026_1` 超过 0.65；但这不是最终证明，原因：

- `full_agent` 目前由本地公式 `policy_runner.agent_decision()` 产生，不是 DeepSeek pro 的真实决策。
- 部分窗口相对原始 Top3 的均值提升为负，说明稳定超越基线尚未达成。
- `H2026_1` 数据截至 2026-06-23，当前日期为 2026-06-25，窗口未完整结束。
- 2023-2025 新闻覆盖为 0，新闻层未进入有效训练。

## 当前 DeepSeek 状态

已经完成：

- `deepseek-v4-flash` 设为回测训练、多 epoch 搜索、ablation 和错误反思的默认模型。
- `deepseek-v4-pro` 设为正式用户推理、最终冲指标和最终验收模型。
- API key 可由代码从本地未提交密钥文件、环境变量或未提交 `.env` 读取；不得写入 prompt、日志、报告、ledger、缓存元数据或 Git。
- 决策卡 schema 测试已覆盖默认模型和安全读取。

尚未完成：

- 旧的主时间线训练脚本仍需并入 DeepSeek 决策闭环，不能再把本地 deterministic runner 当成 Agent 实绩。
- 双模式 runner 已能生成 evidence pack 并调用 DeepSeek，但还需要扩大为正式多 epoch 训练入口。
- 最终冲指标和正式验收尚未用 `deepseek-v4-pro` 对锁定策略复核。
- 还需要把 DeepSeek 错误复盘系统化写入 policy 更新、memory 反证和 Book Skill 优先级调整。

## 新闻通道状态

当前 `reports/date_generalization/news_coverage.csv` 显示：

- 2023-2025 的训练和测试样本新闻 active rate 为 0。
- 2026_1 新闻 active rate 约 0.35。

结论：新闻框架方向正确，但历史覆盖不足。下一步应优先建立可回测的新闻/公告历史通道：

- 股票自身新闻/公告
- 同行业新闻和相对曝光
- 政策背景
- 地域背景
- 风险预警与机会信号
- 证据质量和时间戳质量

覆盖不足时必须显式降权或输出 `news_missing`，不得让模型把缺失当作中性好消息。

## Book Skill 状态

Book Skill 已有策略卡和适配日志，但下一步仍需加强：

- 每次触发必须有真实策略 ID、书名、章节、页码范围、提取方式。
- 缺页码或来源不完整的技能只能作为弱证据。
- 适配统计需要区分正向验证、反向失效、市场环境不适用。
- 不得修改 `E:\stock\ref` 原始 PDF。

## 已清理内容

已清理或收敛：

- 本地密钥文件：`ds_api.txt`、`tushare_token.txt` 可由代码读取，但必须被 `.gitignore` 忽略、权限收紧，且不得输出明文。
- Python/pytest 缓存：`.pytest_cache`、`src/scripts/tests` 下的 `__pycache__`
- 临时渲染/输出目录：`tmp/`、`output/`
- 过渡计划文档：`docs/BACKTEST_LIGHTWEIGHT_PLAN.md`、`docs/BACKTEST_NEXT_GOAL.md`、`docs/BACKTEST_SCALEUP_ROADMAP.md`

已更新 `.gitignore`：

- 忽略 `ds_api.txt`
- 忽略 `tushare_token.txt`
- 忽略 `*.key`、`*.pem`
- 忽略 `tmp/`、`output/`

保留：

- `reports/backtest_scale_500/epoch1/ground_truth.csv`
- `reports/backtest_scale_500/test/ground_truth.csv`
- `reports/date_generalization/` 当前主报告链路
- `data/` 缓存和 `book_skills/`
- `E:\stock\ref` 原始 PDF 未触碰

## 下一步门禁

继续开启目标前，建议先完成以下三件事：

1. 接入真实 DeepSeek 决策层

   - 生成 `evidence_pack.jsonl`。
   - 回测训练默认调用 `deepseek-v4-flash` 输出 JSON decision card；正式复核和最终验收显式切换到 `deepseek-v4-pro`。
   - 本地 schema 校验。
   - 失败时重试或标记 invalid，不得静默回落为成功。
   - 本地 deterministic policy runner 只作为 fallback/baseline，报告中必须分开标注。

2. 修正时间验收口径

   - `H2026_1` 按当前可用日期截断为 `H2026_1_YTD`。
   - 完整半年验收只能在 2026-06-30 之后运行。
   - 未到期窗口只能标记为 provisional。

3. 补新闻历史覆盖或缺失机制

   - 若拿不到历史新闻，至少用公告/交易所/巨潮/东方财富公开信息补历史事件。
   - 新闻缺失必须形成独立特征 `news_missing_rate` 和 evidence quality penalty。
   - 先证明新闻通道在 ablation 中有边际贡献，再把它作为核心优势写进用户报告。

## 并行 Agent 加速方案

时间块之间不能乱并行，因为策略更新有时间依赖；但每个时间块内部可以并行。

建议拆分：

1. Evidence Agent
   - 按 `time_block × stock_shard` 生成时间安全证据包。
   - 不接触未来收益。

2. News Agent
   - 补自身新闻、同行新闻、政策、地域、相对曝光、时间戳质量。
   - 输出缺口和证据质量。

3. Book Skill Agent
   - 只读已提取文本和策略卡。
   - 输出真实 strategy_id、来源、触发、反证、适用边界。

4. DeepSeek Decision Agent
   - 读取 evidence pack。
   - 回测训练、多 epoch 搜索和 ablation 调用 `deepseek-v4-flash`。
   - 正式用户推理、最终冲指标和最终验收调用 `deepseek-v4-pro`。
   - 不计算未来收益。

5. Metrics Agent
   - 只做后验 Ground Truth、ablation、稳定性统计。
   - 不参与当前 step 决策。

6. Audit Agent
   - 检查时间泄漏、付费源、缺页码、key 泄露、未到期窗口、报告措辞。

推荐 shard：500 股拆成 10 个 50 股 shard；portfolio_pool 和 single_stock 两条线分开并行；合并主键为：

```text
agent_policy_version + valid_block + decision_date + code + variant
```

## 是否可以继续

可以继续，但建议下一次 goal 先做“训练闭环纠偏”，不要先追更高指标。优先顺序：

1. DeepSeek 真实决策接入：训练 round 默认 `deepseek-v4-flash`，最终验收显式 `deepseek-v4-pro`。
2. H2026 YTD/完整窗口口径修正。
3. 历史新闻/公告通道补齐和缺失降权。
4. 再跑一轮小规模 50-100 股 smoke。
5. 通过后再扩大到 300/500 股、多 epoch、多 Agent 并行。




