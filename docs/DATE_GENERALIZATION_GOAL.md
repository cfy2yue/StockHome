# Goal: 日期泛化型 Agent 策略训练与最终验收

更新时间：2026-06-25

> 说明：本文件是历史 goal。当前唯一推进边界以根目录 `goal.md` 为准：用户端允许输出买入、卖出、加仓、减仓、持有、等待或补数据等研究辅助型操作建议；禁止的是自动下单、接券商、收益保证和未来字段泄漏。

## 最新执行指引

下一轮执行请优先读取：

```text
docs/NEXT_GOAL_STRATEGY_HARDENING.md
```

该文档是在当前 DeepSeek gate round 结果之后更新的下一步 goal，重点是：固化真实用户工作流、启用可选标准化数据层、强化 Book Skill grounding 和结构化经验记忆，并围绕组合模式 20 日正收益率偏低的问题做系统性优化。

## 0. 本轮目标

本 goal 是下一步执行指引和最终验收标准。核心目标不是把某个 Python 公式调到历史窗口最好，而是训练一套可解释、可复现、可跨时间块泛化的 A 股研究 Agent 工作流。

本轮必须先纠偏，再扩大：

1. 先把 DeepSeek 真实接入决策层，并区分训练模型与最终验收模型。
2. 再修正时间窗口和未到期 Ground Truth 口径。
3. 再补新闻/公告历史覆盖或缺失降权机制。
4. 小规模 smoke 通过后，再扩大到 300/500 股、最多 8 个 epoch、多 Agent 并行。

最终目标是得到一套锁定的 Agent 策略版本，它在最新可验证时间窗口上表现达标，同时回看早期时间块不明显失效。

## 1. 项目边界

项目输出研究辅助型操作建议：

- 可以输出买入、卖出、加仓、减仓、持有、等待或补数据建议。
- 必须配套仓位/阈值、证据、反证、失效条件和复评条件。
- 不自动交易，不接券商接口，不下单，不承诺收益，不输出目标价必达/稳赚/必涨。
- `继续深挖`、`放入观察`、`暂时剔除`、`信息不足` 只作为辅助研究分级。
- 回测内部可以记录模拟动作：增加研究暴露、降低研究暴露、保持观察、转入现金、信息不足不动作。

数据源边界：

- 允许本地缓存、AKShare、BaoStock、mootdx、东方财富、巨潮、交易所公开页面或公开接口。
- 允许用户已合法授权的 Wind、Choice、iFinD、同花顺会员、Tushare Pro 等付费/会员/标准化数据源；必须保护凭证、标注来源、优先走离线缓存，缺 `available_at` 或财报披露日的字段不得进入 walk-forward 决策。
- 不删除、不移动、不覆盖 `E:\stock\ref` 原始 PDF。
- Book Skill 必须基于真实提取文本或明确人工复核，不能编造。

DeepSeek key 安全边界：

- API key 不得写入代码、配置、报告、ledger、prompt、日志或 Git。
- 允许代码从本地未提交密钥文件、环境变量或未提交 `.env` 读取 key；当前本地密钥文件包括 `ds_api.txt` 和 `tushare_token.txt`。
- 本地密钥文件必须被 `.gitignore` 忽略，权限尽量收紧；不得把 key 明文写入 prompt、日志、报告、ledger、缓存元数据或 Git。
- 任何日志只能打印脱敏 key。

## 2. 当前状态基线

以 `docs/PROJECT_GOAL_AUDIT.md` 为当前状态审计依据。

当前已经具备：

- 时间块回测脚本：`scripts/run_agent_strategy_training_rounds.py`
- DeepSeek 客户端：`src/agent_training/deepseek_client.py`
- DeepSeek 回测训练默认模型：`deepseek-v4-flash`
- DeepSeek 正式推理/最终验收模型：`deepseek-v4-pro`
- 决策卡 schema：`src/agent_training/decision_card.py`
- 本地 policy runner：`src/agent_training/policy_runner.py`
- 当前主报告目录：`reports/date_generalization/`
- Book Skill 和新闻向量雏形

当前不能宣称完成：

- 主训练脚本还没有真实调用 `chat_json()`，当前 `full_agent` 主要是本地确定性替身。
- 2023-2025 新闻 active rate 为 0，不能证明新闻 Agent 增量。
- 当前日期是 2026-06-25，`H2026_1` 尚未完整结束；并且 20 日 Ground Truth 对窗口末端决策还未成熟。
- 部分时间块相对原始 Top3 的 20 日均值提升为负，稳定超越基线尚未证明。

## 3. 执行优先级

必须按以下顺序推进。前一阶段未达门禁，不得把后一阶段结果写成最终通过。

### Phase 0: Preflight 与安全检查

目标：确认项目处于可执行、可审计、无敏感泄露状态。

必须完成：

- 运行全量测试，至少 `pytest tests -q` 通过。
- 扫描项目文件中不得存在真实 API key 片段。
- 确认 `reports/backtest_scale_500/epoch1/ground_truth.csv` 和 `reports/backtest_scale_500/test/ground_truth.csv` 存在。
- 确认 `E:\stock\ref` 未被修改、删除或覆盖。
- 确认 `.gitignore` 忽略 `.env`、`.env.*`、`secrets/`、`*api_key*`、`*secret*`、`ds_api.txt`、`*.key`、`*.pem`。

输出：

- `reports/date_generalization/preflight_check.md`
- `reports/date_generalization/preflight_check.json`

### Phase 1: DeepSeek 真实决策接入

目标：把 Agent 决策从本地确定性替身升级为 DeepSeek API 真实决策，并采用模型分层策略降低训练成本。

必须实现：

- Python evidence builder 生成时间安全证据包，不包含未来收益。
- 回测训练、ablation、错误反思、候选策略搜索默认使用 `deepseek-v4-flash`。
- 正式用户推理、最终冲指标、最终验收报告使用 `deepseek-v4-pro`。
- DeepSeek 读取 evidence pack，输出 JSON decision card。
- 本地 schema 校验 DeepSeek 输出。
- 校验失败时允许重试；多次失败后标记 invalid，不得静默当成功。
- 本地 deterministic policy runner 只能作为 fallback/baseline，报告必须分开标注。
- 禁止把未来 5/10/20 日收益放入 DeepSeek 输入。

Evidence pack 必须至少包含：

- `agent_policy_version`
- `decision_date`
- `available_at`
- `time_block`
- `task_mode`: `portfolio_pool` 或 `single_stock`
- `code/name`
- `industry/peer_group`
- `region`
- `python_signal_summary`
- `python_features`
- `news_signal_summary`
- `news_features`
- `book_skill_candidates`
- `memory_context`
- `counter_evidence`
- `data_missing_flags`
- `allowed_research_grades`
- `allowed_simulated_actions`

Decision card 必须至少包含：

- `type`
- `agent_policy_version`
- `variant`
- `step`
- `train_blocks`
- `valid_block`
- `decision_date`
- `code/name`
- `task_mode`
- `research_grade`
- `simulated_action`
- `simulated_weight_change`
- `python_signal_summary`
- `news_signal_summary`
- `book_skill_evidence`
- `memory_experience_used`
- `counter_evidence`
- `final_agent_reasoning_summary`
- `confidence_level`
- `data_missing_flags`
- `error_reflection`
- `research_only`
- `not_investment_instruction`

输出：

- `reports/date_generalization/evidence_pack_sample.jsonl`
- `reports/date_generalization/deepseek_decision_ledger.jsonl`
- `reports/date_generalization/deepseek_invalid_outputs.jsonl`
- `reports/date_generalization/deepseek_usage_summary.csv`

验收门槛：

- 至少 50-100 股 smoke 能生成真实 DeepSeek 决策卡；训练 smoke 可使用 `deepseek-v4-flash`，最终验收 smoke 必须使用 `deepseek-v4-pro` 复核。
- 决策卡 schema 校验通过率不低于 95%；低于 95% 必须先修 prompt/schema。
- 报告明确区分 `deepseek_agent`、`python_fallback`、`python_only_baseline`。

### Phase 2: 时间窗口与 Ground Truth 口径修正

目标：避免把未到期数据算成通过。

时间块定义：

- `H2023_1`: 2023-01-01 至 2023-06-30
- `H2023_2`: 2023-07-01 至 2023-12-31
- `H2024_1`: 2024-01-01 至 2024-06-30
- `H2024_2`: 2024-07-01 至 2024-12-31
- `H2025_1`: 2025-01-01 至 2025-06-30
- `H2025_2`: 2025-07-01 至 2025-12-31
- `H2026_1_YTD`: 2026-01-01 至当前已可用交易日

严格规则：

- 当前日期早于 2026-06-30 时，`H2026_1` 只能标记为 `YTD/provisional`。
- 即使日期超过 2026-06-30，窗口末端决策也必须等未来 20 个交易日 Ground Truth 成熟后才可纳入 20 日验收。
- 未成熟样本必须标记为 `gt_pending`，不得计入最终通过率。
- 年度块如果 2021/2022 数据不足，必须写为数据缺口，不得伪造年度验证。

输出：

- `reports/date_generalization/year_blocks.yaml`
- `reports/date_generalization/gt_maturity_report.csv`
- `reports/date_generalization/data_coverage.md`

验收门槛：

- 每个指标表必须包含 `is_provisional` 或 `gt_status` 字段。
- `H2026_1_YTD` 可以作为阶段性观察，不能作为最终完整半年通过。

### Phase 3: 新闻/公告历史通道与缺失降权

目标：让新闻通道真实进入可回测框架；如果覆盖不足，也要正确降权。

新闻 Agent 输入必须覆盖：

- 股票自身新闻/公告
- 同行业新闻和相对曝光
- 政策背景
- 地域背景
- 风险预警
- 机会信号
- 证据质量
- 时间戳质量

必须输出字段：

- `self_news_intensity`
- `peer_news_intensity`
- `policy_background_score`
- `region_background_score`
- `self_vs_peer_attention_gap`
- `peer_active_self_silent_flag`
- `news_warning_score`
- `news_opportunity_score`
- `news_evidence_quality`
- `news_missing_rate`
- `news_timestamp_quality`

规则：

- 新闻/公告必须满足 `available_at <= decision_time`。
- 不能把新闻缺失当作中性好消息。
- 若历史新闻拿不到，至少尝试用公告、交易所、巨潮、东方财富公开信息补历史事件。
- 新闻覆盖不足时，DeepSeek evidence pack 必须显式提供 `news_missing_rate` 和证据质量提示。
- 只有 ablation 显示新闻通道有边际贡献，才能在用户报告中写成优势。

输出：

- `reports/date_generalization/news_coverage.csv`
- `reports/date_generalization/news_agent_feature_table.csv`
- `reports/date_generalization/news_ablation_report.md`

验收门槛：

- 每个时间块都必须有新闻/公告覆盖统计。
- 覆盖不足的时间块必须标记 `news_insufficient`。
- 不允许在 `news_insufficient` 时间块宣称新闻层有效。

### Phase 4: Book Skill 与 memory 纠偏

目标：把书籍策略从“触发文本”升级为“有来源、有反证、有适用边界”的 Agent 证据。

硬门禁：

- Book Skill 是本项目核心特点之一，不是报告里的附属说明。
- DeepSeek 决策前必须接收并审阅 Book Skill 候选材料；没有候选时也必须显式收到 `book_skill_candidates=[]` 和缺失原因。
- DeepSeek 的 `final_agent_reasoning_summary` 必须说明是否使用了 Book Skill，以及它和量价、新闻、memory 反证之间如何取舍。
- 反思阶段必须统计 Book Skill 的触发次数、正向验证、反向失效和适用边界。
- 被多个时间块反复验证的 Book Skill 适配规则应提升优先级；在下一时间块失败的必须降权或进入反证区。
- 不得编造 Book Skill 来源；缺书名、章节、页码或提取方式时只能作为弱证据。

Book Skill 使用要求：

- 引用真实策略 ID。
- 标注书名、章节、页码范围、策略 ID、提取方式。
- 不修改原始策略卡；新增内容只能写为派生观察、适配条件或反证。
- 缺页码或来源不完整的策略只能作为弱证据。

Memory 要求：

- 记录成功经验、失败反例、过拟合迹象、适用市场环境。
- 记录规则来源、训练块表现、下一块验证结果。
- 下一块失败的规则必须降权或废弃，不能覆盖失败证据。

输出：

- `memory/strategy_experience.md`
- `memory/book_skill_adaptation.md`
- `reports/date_generalization/book_skill_adaptation_log.csv`

验收门槛：

- 至少输出 3 条有数据支撑的 Book Skill 适配观察；不足 3 条必须说明样本不足。
- 每条适配观察必须包含触发次数、对照样本、20 日正收益率、20 日均值、适用/失效条件。
- 最终默认策略升级时，至少说明哪些 Book Skill 被提升、保留观察、降权或废弃。

## 4. Walk-forward 训练流程

每个 epoch 必须严格按时间线执行：

```text
H2023_1 -> H2023_2 -> H2024_1 -> H2024_2 -> H2025_1 -> H2025_2 -> H2026_1_YTD
```

规则：

- 第 `t` 块结束后，才能使用第 `t` 块 Ground Truth 做错误归因。
- 更新后的策略必须冻结为新的 `agent_policy_version`，再验证第 `t+1` 块。
- 第 `t+1` 块结果不能反向修改用于验证它的策略。
- `H2026_1_YTD` 只能作为阶段性最新窗口；最终完整窗口需要等 Ground Truth 成熟。

每个 step 必须输出：

- 训练块
- 验证块
- 策略版本
- DeepSeek prompt/schema 版本
- Python gate 版本
- news schema 版本
- Book Skill 版本
- memory 版本
- 决策数
- invalid 决策数
- gt_pending 决策数
- 20 日正收益率
- 20 日均值
- 相对原始 Top3 差值
- 是否达标

Epoch 规划：

- `epoch_1`: 建立真实 DeepSeek Agent 基线和失败地图。
- `epoch_2`: 只加入 epoch_1 中通过下一块验证的 1-3 条规则。
- `epoch_3` 至 `epoch_8`: 仅当上一轮在下一时间块有可解释改善时继续；每轮最多引入 1-3 条规则或阈值变化。
- 最多运行 8 个 epoch；若连续 2 个 epoch 在下一时间块退化，停止继续追参，转入错误归因和数据通道修复。
- epoch 扩大阶段优先使用 `deepseek-v4-flash`，把 token 花在更多股票、更多时间块和更多反事实/ablation 上。
- 最终锁定候选策略后，必须用 `deepseek-v4-pro` 在隔离 test 与最新可验证时间块上复核。

## 5. 两类任务必须同时支持

硬性要求：

- 每个回测 round 必须同时跑组合模式和单支模式，不能只优化其中一种。
- 每个 epoch 的策略更新必须分别记录：组合模式学到了什么、单支模式学到了什么、两者冲突时如何取舍。
- DeepSeek 决策前必须明确收到 `task_mode`，并按当前任务选择不同判断口径。
- 组合模式优化的是候选池内的研究注意力分配；单支模式优化的是某只股票的持续评估、盯盘和模拟路径。
- 两种模式都要纳入 ablation、Book Skill 验证、新闻通道验证、Python gate 阈值反思和最终报告。
- 若某一模式因数据不足、接口失败或样本过少无法完成，必须在报告和 ledger 中显式标记，不得用另一模式结果替代。

### 组合模式

回答“多支股票之间如何筛选和分配研究注意力”。

必须比较：

- 原始系统 Top3
- 原始系统 Top5
- 原始系统 Top10
- DeepSeek Agent 综合决策
- Python only
- no_news
- no_bookskill
- no_memory
- 随机 TopN
- 全候选池等权
- 银行 3% 年化现金基线

输出指标：

- 决策期数
- 平均入选数量
- 现金防守比例
- 20 日均值
- 20 日正收益率
- 20 日收益标准差
- 20 日跌幅超过 5% 比例
- 稳定性分
- 相对原始 Top3 的 raw 差值
- 相对原始 Top3 的 cash-adjusted 差值

### 单支模式

回答“单只股票如何盯盘、复核和模拟路径”。

每只股票每个决策日必须记录：

- 研究分级
- 模拟动作
- 模拟研究暴露比例
- 触发 Gate
- 触发 Book Skill
- 新闻预警/机会
- 反证标记
- DeepSeek 最终理由摘要
- 未来 5/10/20 日后验表现

单支模式输出是研究模拟路径和用户端操作建议；不得自动下单或承诺收益。

### 两类任务的共同优化对象

每轮反思和策略更新都可以优化以下对象，但必须记录来源、版本和下一时间块验证结果：

- DeepSeek prompt 和证据组织方式。
- 新闻/公告通道字段、缺失降权、风险/机会分类。
- Book Skill 触发、优先级、适用边界和反证规则。
- Python gate 的阈值、分叉判据和 fallback 规则。
- 组合模式 TopN、现金防守、候选池过滤和分散度约束。
- 单支模式模拟暴露变化、信息不足处理和风险降级规则。
- 用户端解释模板和数据缺口提示。

禁止只根据当前验证块追求历史最优。任何新规则都必须在下一时间块验证；失败时进入 memory 反证区。

## 6. 多 Agent 并行加速

时间块之间不能乱并行，因为策略版本有时间依赖。每个时间块内部可以并行。

推荐拆分：

1. Evidence Agent
   - 按 `time_block × stock_shard` 生成时间安全证据包。
   - 不接触未来收益。

2. News Agent
   - 补自身、同行、政策、地域、相对曝光、证据质量。
   - 覆盖不足时输出缺口。

3. Book Skill Agent
   - 只读已提取文本和策略卡。
   - 输出真实 strategy_id、来源、触发、反证、适用边界。

4. DeepSeek Decision Agent
   - 回测训练和多 epoch 搜索调用 `deepseek-v4-flash`。
   - 正式用户推理、最终冲指标和最终验收调用 `deepseek-v4-pro`。
   - 输出 JSON decision card。
   - 不计算未来收益。

5. Metrics Agent
   - 合并 ledger。
   - 用未来 5/10/20 日做后验统计。
   - 不参与当前 step 决策。

6. Audit Agent
   - 检查时间泄漏、付费源、key 泄露、Book Skill 来源、未到期窗口和报告措辞。


DeepSeek 并发策略：

- 回测训练默认 `--max-workers 0`，由项目按模型自动取最大并发：`deepseek-v4-flash` 上限 2500，`deepseek-v4-pro` 上限 500，并以本轮 evidence pack 数量封顶。
- 若出现 HTTP 429、timeout 或本机连接耗尽，必须记录到 usage/invalid，并把该轮稳定并发写回报告。
- 多 Agent/shard 并行时，所有 shard 的总并发仍需遵守同一账号级上限。
推荐 shard：

- 50-100 股 smoke：2 个 25-50 股 shard。
- 300 股训练：6 个 50 股 shard。
- 500 股完整：10 个 50 股 shard。

合并主键：

```text
agent_policy_version + valid_block + decision_date + code + task_mode + variant
```

## 7. 输出目录

保持目录简洁，主线统一写入：

```text
reports/date_generalization/
  preflight_check.md
  preflight_check.json
  data_coverage.md
  year_blocks.yaml
  gt_maturity_report.csv
  evidence_pack_sample.jsonl
  deepseek_decision_ledger.jsonl
  deepseek_invalid_outputs.jsonl
  deepseek_usage_summary.csv
  agent_decision_ledger.jsonl
  agent_policy_metrics.csv
  agent_policy_ablation.csv
  round_metrics.csv
  round_strategy_changes.csv
  gate_optimization_log.csv
  book_skill_adaptation_log.csv
  news_coverage.csv
  news_agent_feature_table.csv
  timeline_epoch_metrics.csv
  timeline_epoch_updates.csv
  timeline_epoch_state.yaml
  timeline_failure_diagnostics.csv
  final_acceptance_metrics.csv
  final_user_manual.md
  user_guide.md
```

缓存统一放：

```text
data/date_generalization_cache/
```

## 8. 最终验收标准

只有同时满足以下条件，才允许宣称“策略具备阶段性日期泛化能力”。

### 8.1 工程验收

- 全量测试通过。
- 主训练脚本可从干净状态跑通。
- 无真实 API key 落盘。
- DeepSeek 真调用已接入主训练决策层。
- 回测训练默认使用 `deepseek-v4-flash`，正式推理和最终验收使用 `deepseek-v4-pro`。
- 本地 deterministic policy runner 只作为 fallback/baseline，不冒充 DeepSeek Agent。
- 所有 DeepSeek 输出都通过 schema 校验或被记录为 invalid。
- `E:\stock\ref` 原始 PDF 未被修改、删除、覆盖。

### 8.2 数据验收

- 完成 `2021-当前` 数据覆盖审计；缺失年份必须报告。
- 若 2021/2022 不足，使用 2023 起半年块滚动验证。
- 每只参与股票的日线数据覆盖、财务披露日、新闻/公告覆盖都有记录。
- 财务字段必须有报告期和披露日；缺披露日则不得参与 walk-forward 判断。
- `gt_pending` 样本不得计入最终 20 日验收。
- `H2026_1_YTD` 未成熟时只能标记 provisional。

### 8.3 策略训练验收

- 至少完成一个真实 DeepSeek Agent smoke：50-100 股，组合模式和单支模式都跑通。
- 每个正式回测 round 必须同时产生组合模式 ledger 与单支模式 ledger，并分别计算指标。
- 组合模式必须输出候选池排序、TopN/现金防守结果、相对基线差值。
- 单支模式必须输出每只股票的模拟研究路径、分级变化、暴露变化和后验 5/10/20 日验证。
- 扩大训练前，smoke 必须证明 evidence pack、decision ledger、metrics、ablation 全链路可用。
- 完整训练至少包含 300 股训练池和隔离 test 池；若不足必须说明原因。
- 每轮必须 train -> 锁定策略 -> test -> 下一时间块验证。
- test 和下一时间块不得参与调参。
- 每个 round 必须冻结并记录 `agent_policy_version`。
- 最多 8 个 epoch；每个 epoch 必须保存策略变化、失败样本和反证。

### 8.4 指标验收

最终锁定策略必须满足：

- 最新完整且 Ground Truth 成熟的 6 个月 test，20 日正收益率不低于 `0.65`。
- 若当前只有 `H2026_1_YTD`，则 YTD 20 日正收益率不低于 `0.65` 只能算 provisional，不算最终完整通过。
- 同一锁定策略回看早期可用半年块，每块 20 日正收益率不低于 `0.60`。
- 相对原始系统 Top3 的 20 日均值提升目标不低于 `0.2` 个百分点。
- 如果均值提升未达 `0.2`，但稳定性显著提升，必须单独标记为“稳定性改善但收益目标未通过”，不能宣称完全通过。
- ablation 必须显示新闻、Book Skill、memory、Python gate 各自是否有边际贡献；无贡献则降权。

### 8.5 报告验收

最终用户报告必须用户友好，包含：

- 系统能做什么、不能做什么。
- Top3/TopN 定义。
- 哪些行是基线，为什么看基线。
- 数据覆盖、缺口、未到期窗口。
- 训练流程和判断流程。
- DeepSeek Agent 如何综合 Python、新闻、Book Skill、memory。
- 单支模式和组合模式结果。
- 两种模式的任务目标、决策口径、操作路径和指标差异。
- 每个时间块指标和是否达标。
- 新闻通道是否真的有效，若无效为什么。
- Book Skill 哪些被验证，哪些被反证。
- 数据源缺失、访问失败、覆盖不足、未到期样本和 NA 的原因。
- 对用户的建议必须清晰指向下一步操作：买入、卖出、加仓、减仓、持有、等待或补数据；必须给仓位/阈值、证据、反证和复评条件。
- 最终是否达标；不达标时说明原因和下一步。

报告不得输出自动下单、接券商执行、收益保证、目标价必达、稳赚、必涨等内容。

## 9. 本轮执行建议

下一次 goal 模式建议按以下顺序执行：

1. 实现 evidence pack builder 和 DeepSeek decision runner。
2. 对 50-100 股跑 smoke，确认真实 DeepSeek 决策卡可用。
3. 修正 `H2026_1_YTD` 和 `gt_pending` 口径。
4. 加入新闻/公告缺失降权和覆盖报告。
5. 使用 `deepseek-v4-flash` 跑 300 股 walk-forward 和多 epoch 训练。
6. 若指标和链路稳定，再扩展到 500 股、最多 8 epoch、多 Agent 并行。
7. 最终候选策略锁定后，用 `deepseek-v4-pro` 做正式复核和最终验收报告。

成功定义：不是一次跑出漂亮数字，而是完整链路能证明“规则从过去训练得来、在未来块验证、失败被记录、有效规则能跨时间块复用”。


