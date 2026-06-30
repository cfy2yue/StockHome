# Goal: 策略固化、数据升级与组合正收益率提升

更新时间：2026-06-25

> 说明：本文件是历史阶段计划。当前唯一推进边界以根目录 `goal.md` 为准：用户端必须给明确操作建议和阈值，研究分级只作为辅助标签；系统不自动下单、不接券商、不承诺收益。

## 0. 本轮目标

本 goal 是下一轮执行指引。目标不是继续堆 DeepSeek 调用，而是把当前项目从“能跑 round 的实验系统”推进到“真实用户可按固定工作流使用、策略可审计、组合正收益率有明确提升路径”的系统。

核心目标：

1. 固化真实用户工作流，确保单支股票和多股组合两种任务都按固定策略流程推进。
2. 清理并稳定项目目录，顶层只保留用户和后续 Agent 需要直接读取的核心报告。
3. 针对组合模式正收益率偏低的问题，系统化寻找能提升 20 日正收益率的方向。
4. 可选启用 Tushare Pro 标准化离线数据层，扩展时间、股票范围和同行/行业特征。
5. 强化 Book Skill grounding 和结构化经验记忆，避免模型走捷径或遗忘失败经验。
6. 将下一轮数据准备扩展到至少 5000 支股票或全 A 可覆盖股票，形成相关股票、同行、地域、概念和新闻扩散通道；这些数据不必全部进入训练，但必须作为可查询的市场底座。
7. 深挖书籍策略，并把公开论坛、社区经验帖、公开研报摘要等作为 `candidate_skill_seed`，通过 scaling 回测验证后再升级为正式策略经验。

项目输出研究辅助型操作建议，可以给买入、卖出、加仓、减仓、持有、等待或补数据建议，并必须给仓位/阈值、证据、反证、失效条件和复评条件；系统不自动交易、不接券商、不承诺收益。`继续深挖/放入观察/暂时剔除/信息不足` 只作为辅助研究分级。

## 1. 当前基线

当前 DeepSeek 双模式真实 round 已完成三组 100 股 panel。

### 单支股票模式

| 策略轮次 | 20日 raw 正收益率 | 20日 cash-adjusted 正收益率 | 判断 |
|---|---:|---:|---|
| default | 0.8003 | 0.9111 | 可作为盯盘/排雷雏形 |
| pullback | 0.7408 | 0.8778 | 稳定但需扩大样本 |
| gate | 0.7824 | 0.9000 | 仍较稳 |

结论：单支模式已有较强研究辅助价值，但仍需扩大样本和补新闻/同行通道。

### 多股组合模式

| 策略轮次 | 20日 raw 均值 | 20日 raw 正收益率 | 20日 cash-adjusted 均值 | 20日 cash-adjusted 正收益率 |
|---|---:|---:|---:|---:|
| default | -10.5658 | 0.2294 | -0.2858 | 0.5000 |
| pullback | -1.5364 | 0.3859 | 0.0380 | 0.5555 |
| gate | -0.2435 | 0.4043 | 0.1805 | 0.5333 |

结论：

- `pullback_recovery` 和 `pool_pullback + every_2_weeks` 改善了均值和大亏控制。
- 组合模式 20 日正收益率仍低于 0.60/0.65，不能宣称策略达标。
- 下一轮必须以提升组合正收益率为主，不只优化均值。

## 2. 目录整理要求

当前已做一次整理：

- 临时 smoke/plan/check 文件已删除。
- 原始 DeepSeek evidence/decision ledger 已移动到 `reports/date_generalization/archive_raw_rounds/`。
- 离线实验明细已移动到 `reports/date_generalization/experiments/`。
- 顶层保留 summary、aggregate、diagnostics、comparison、goal 需要读取的报告。
- 清理记录：`reports/date_generalization/cleanup_manifest.md`

下一轮要求：

- 新 round 输出必须带清晰前缀，不允许泛泛命名为 `deepseek_dual_mode_*` 后覆盖主文件。
- 每轮结束后必须生成：
  - 一个用户可读 summary。
  - 一个 aggregate CSV。
  - 一个 diagnostics CSV。
  - 一个 strategy change / memory update。
- 原始大文件进入 `archive_raw_rounds/` 或按 round 子目录归档，不挤在顶层。

## 3. 工作流固化

必须新增或更新一个机器可读工作流配置：

```text
config/agent_workflow_strategy.yaml
```

它必须表达两类任务：

1. `single_stock_watch`
   - 输入：单只股票、日期、可见数据、新闻/公告、同行、Book Skill、memory。
   - 输出：研究分级、模拟动作、模拟研究暴露、触发原因、反证、数据缺口。
   - 目标：盯盘、排雷、复核、解释。

2. `portfolio_pool_optimize`
   - 输入：候选池、日期、行业/同行特征、市场状态、新闻/公告、Book Skill、memory。
   - 输出：候选排序、TopN、现金防守比例、放弃决策原因、反证。
   - 目标：多股候选池研究注意力分配。

固定流程：

```text
preflight
-> data_cache_build_or_validate
-> evidence_pack_build
-> deepseek_decision
-> schema_validation
-> metric_backfill_after_gt_maturity
-> failure_reflection
-> strategy_update_proposal
-> strategy_freeze
-> next_block_or_test_validation
-> user_report
```

强制规则：

- DeepSeek 输入不得包含未来收益、`gt_status` 或未来事件。
- 每轮策略必须冻结 `agent_policy_version`。
- test 和下一时间块不得参与调参。
- DeepSeek 决策前必须读取 Book Skill 候选和 memory 反证。
- 如果数据缺失、新闻不足、Book Skill 来源不完整，必须进入 evidence pack 和用户报告。
- 本地 Python deterministic runner 只能作为 baseline/fallback，不能冒充 Agent 实绩。

## 4. 数据升级目标

### 4.1 Tushare Pro 可选通道

用户已提供本机 token 文件：

```text
E:\stock\tushare_token.txt
```

当前项目允许用户已合法授权的付费/会员/标准化数据源。Tushare Pro 应标记为 `paid_standardized` 离线缓存数据源：

- 报告必须标注使用了付费标准化源。
- 不得接券商，不得自动交易。
- 不得把 token 写入代码、日志、报告、prompt、ledger 或 Git。

权限与限速：

- Tushare Pro 积分：15000。
- 限速：100 次/分钟。
- 项目默认请求间隔不低于 0.7 秒。
- 回测只读本地缓存，不在决策点临时请求在线接口。

参考方案：

```text
docs/TUSHARE_PRO_OPTIONAL_DATA_CHANNEL.md
memory/data_source_upgrade.md
```

### 4.2 必须优先缓存的数据

第一批：

- 交易日历
- 股票列表和上市/退市状态
- 日线行情
- 复权因子
- 停复牌
- 涨跌停价格
- 财务报告期和披露日
- 行业分类、地域、概念标签

第二批：

- 财务指标
- 资产负债表、利润表、现金流量表
- 业绩预告/快报
- 筹码分布、每日筹码
- 券商金股

第三批：

- 公告/新闻辅助字段
- 每日胜率等特色数据

### 4.3 5000 股离线市场底座

下一轮数据准备目标从小样本扩展到“至少 5000 支股票或全 A 可覆盖股票”。如果因上市数量、权限、停牌退市、接口失败等原因实际不足 5000 支，必须在覆盖率报告中列明原因、缺口数量和补选逻辑。

目标不是把 5000 支股票全部喂给 DeepSeek，而是建立可查询的市场底座：

- 单支股票任务：可随时检索目标股的同行、上下游、地域、概念、指数成分、相似走势股票。
- 多股组合任务：可从更大候选池中抽取 50/100/300/500 股 panel 做训练、验证和 test。
- 新闻/公告通道：能计算“同行被提及但目标股未被提及”“政策覆盖同行但未覆盖目标股”“风险从同行扩散到目标股”等相对信号。
- Book Skill 验证：同一条策略不能只在少数热门股上验证，必须在不同行业、不同市值、不同年份中观察稳定性。
- Scaling 评估：支持 300/500/1000 股实验，不必每轮都跑全量 5000 股。

最小缓存字段：

- 股票主数据：代码、简称、上市日期、退市日期、交易所、市场板块、行业、地域、概念标签。
- 交易数据：日线、复权因子、停复牌、涨跌停、成交额、换手、基础波动率。
- 财务数据：报告期、披露日、主要财务指标、资产负债表、利润表、现金流量表；缺披露日的字段不得进入 walk-forward 决策。
- 关联图谱：同行、同概念、同地域、同指数、历史相关性 TopK、新闻共现 TopK。
- 质量标记：缺失字段、接口失败、异常值、复权/停牌处理状态。

建议目录：

```text
data/date_generalization_cache/market_5000/
reports/date_generalization/data_cache_5000_coverage.md
reports/date_generalization/data_cache_5000_coverage.csv
```

数据抓取要求：

- 一次请求尽量批量拿长时间窗口，不逐股逐日循环。
- 遵守 100 次/分钟限速，项目默认请求间隔不低于 0.7 秒。
- 决策点只读本地缓存，不在线请求。
- 接口失败时跳过并记录，不让整个流程崩溃。
- 不把 token 写入代码、日志、报告、prompt 或缓存元数据。

## 5. 组合正收益率提升方向

当前组合模式正收益率太低。下一轮必须以可验证实验寻找提升方向。

### 5.1 候选数量和频率

系统实验：

- Top1 / Top2 / Top3 / Top5 / Top10 / Top15。
- 周二/周五。
- 每周一次。
- 每两周一次。
- 市场高波动时降频，低波动时允许正常频率。

目标：

- 不为了凑满 TopN 而选择低质量日期。
- 如果 gate 后候选不足，可以输出“本期放弃组合新增研究暴露”，而不是强行选股。

### 5.2 行业/同行通道

必须新增 peer features：

- 行业/同行 20 日涨跌幅均值。
- 行业/同行正收益广度。
- 个股相对行业强弱。
- 行业内龙头/滞后关系。
- 同行业风险扩散。
- 同行被新闻提及但目标股未被提及的相对沉默信号。

预期价值：

- 避免只看单股动量。
- 组合模式按同池比较，减少风格漂移。
- 支持用户提出的“同类/不同类候选池”工作流。

### 5.3 新闻/公告 world model

必须扩展新闻通道：

- 股票自身新闻/公告。
- 同行业新闻。
- 政策背景。
- 地域背景。
- 风险预警。
- 机会信号。
- 证据质量。
- 时间戳质量。
- 新闻缺失率。

必须输出相对值：

- 自身新闻强度 vs 同行新闻强度。
- 同行活跃但自身沉默。
- 政策利好是否只覆盖同行而未覆盖目标股。
- 风险新闻是否出现行业扩散。

### 5.4 Book Skill 强弱证据

必须把 Book Skill 从“触发 ID”升级为“有来源、有适用边界、有后验表现”的证据。

下一轮至少完成：

- 高频触发 Top20 strategy_id 的来源 grounding。
- 每条策略补书名、章节、OCR 页码范围、提取方式。
- 输出 `reports/date_generalization/book_skill_adaptation_log.csv`。
- 每条适配观察包含触发次数、对照样本、20 日正收益率、20 日均值、适用/失效条件。

如果不足 3 条数据支撑观察，报告必须写明 Book Skill 样本不足。

### 5.5 Book Skill 深挖与公开经验土壤

Book Skill 当前偏弱，下一轮必须从“已有策略卡触发”升级为“书籍深挖 + 外部经验 seed + 回测验证”的闭环。

书籍深挖要求：

- 对 `E:\stock\ref` 已 OCR 文本做 coverage audit，列明每本书已覆盖页数、缺页、是否 partial。
- 对高频触发但描述笼统的 Book Skill 回到逐页文本，补充上下文、适用市场环境、触发条件、失效条件。
- 新增或更新策略时必须保留真实来源：书名、章节、页码范围、策略 ID、提取方式。
- 不修改、不移动、不覆盖 `E:\stock\ref` 原始 PDF。
- 如果某条书籍策略只来自 partial OCR，必须标注 `partial_source`，不得宣称全书系统覆盖。

公开经验 seed 要求：

- 可以把公开论坛、社区经验帖、公开投资者问答和公开数据说明作为 `candidate_skill_seed`，但默认状态必须是 `unverified`。
- 候选来源优先级：
  - 一手公开信息：巨潮资讯、上交所公告、深交所互动易、上证 e 互动、交易所公开数据目录。
  - 社区/情绪信息：东方财富股吧等公开股票社区，只作为情绪、关注度、经验假设来源。
  - 第三方开源接口说明：AKShare 等，只用于发现可抓字段和接口，不替代原始事实核验。
- 每条 seed 必须记录：来源 URL、抓取/阅读日期、原文摘要、抽象规则、适用场景、反证风险、是否可量化、验证状态。
- 社区经验不能直接变成投资判断；必须在 scaling 回测中通过多时间块、多行业、多股票验证，才可升级为 `accepted_skill`。

建议输出：

```text
reports/date_generalization/book_skill_grounding_audit.md
reports/date_generalization/external_skill_seed_log.csv
memory/book_skill_adaptation_ledger.csv
```

参考公开来源入口：

- 巨潮资讯：<https://www.cninfo.com.cn/>
- 上交所公司公告：<https://www.sse.com.cn/disclosure/listedinfo/announcement/>
- 深交所互动易：<https://irm.cninfo.com.cn/>
- 上证 e 互动：<https://sns.sseinfo.com/>
- 东方财富股吧：<https://guba.eastmoney.com/>
- AKShare 股票数据接口说明：<https://akshare.akfamily.xyz/data/stock/stock.html>

### 5.6 现金防守与放弃决策

组合模式必须允许“不做本期组合新增暴露”。

需要实验：

- 弱证据时全部转现金。
- 新闻缺失 + 财报披露日缺失 + Book Skill 弱证据时降权。
- 过热无证据时直接排除。
- 市场/行业广度差时降低 TopN 或跳过。

目标不是每天选股，而是提高被选择日期和股票的胜率。

## 6. 经验记忆与防遗忘

当前 memory 位置：

```text
memory/strategy_experience.md
memory/book_skill_adaptation.md
memory/data_source_upgrade.md
```

下一轮必须新增结构化经验文件：

```text
memory/strategy_experience_ledger.csv
memory/book_skill_adaptation_ledger.csv
```

每条经验必须包含：

- `experience_id`
- `source_round`
- `task_mode`
- `rule_or_observation`
- `train_blocks`
- `validation_block`
- `metric_before`
- `metric_after`
- `accepted_or_rejected`
- `failure_condition`
- `next_action`

DeepSeek 决策前只读取被接受或仍在观察的经验；被反证经验必须作为 counter evidence，而不是被覆盖。

## 7. 模型与泛化实验

训练阶段：

- 使用 `deepseek-v4-flash`。
- 默认 `--max-workers 0` 自动并发。
- 至少三组不同股票 sample。
- 每个 round 必须同时跑 `single_stock_watch` 和 `portfolio_pool_optimize`。

复核阶段：

- 用 `deepseek-v4-pro` 对锁定策略做小规模复核。
- Pro 不参与训练调参。
- Flash/Pro 差异必须写入报告。

时间泛化：

- 若只用现有数据：继续 2023 起半年块。
- 若 Tushare 缓存补齐：扩展到 2020/2021 起。
- 最终策略必须满足：
  - 最新成熟 test 正收益率不低于 0.65。
  - 往期半年块不低于 0.60。
  - 将来 zeroshot 不低于 0.60。

若不达标，必须写明“不达标原因”，不得换指标宣称成功。

## 8. 本轮交付物

必须交付：

- `config/agent_workflow_strategy.yaml`
- `docs/PROJECT_PROGRESS_AND_NEXT_PLAN.md` 更新版
- `docs/TUSHARE_PRO_OPTIONAL_DATA_CHANNEL.md` 更新版
- `reports/date_generalization/cleanup_manifest.md`
- `reports/date_generalization/strategy_hardening_summary.md`
- `reports/date_generalization/portfolio_positive_rate_experiments.csv`
- `reports/date_generalization/portfolio_positive_rate_experiments.md`
- `memory/strategy_experience_ledger.csv`
- `memory/book_skill_adaptation_ledger.csv`
- `reports/date_generalization/data_cache_5000_coverage.md`
- `reports/date_generalization/data_cache_5000_coverage.csv`
- `reports/date_generalization/book_skill_grounding_audit.md`
- `reports/date_generalization/external_skill_seed_log.csv`

若启用 Tushare Pro，还必须交付：

- `src/data/tushare_pro_adapter.py`
- `scripts/build_tushare_cache.py`
- `reports/date_generalization/tushare_data_coverage.md`
- `data/date_generalization_cache/tushare_pro/`
- `data/date_generalization_cache/market_5000/`

## 9. 验收标准

工程验收：

- 全量测试通过。
- 无真实 key/token 泄露。
- `E:\stock\ref` 原始 PDF 未被修改、删除、覆盖。
- Tushare token 只从本机文件或环境变量读取，不写入任何输出。
- 清理后顶层 reports 目录有 manifest，原始大文件可追溯。
- 若启用数据升级，必须生成 5000 股或全 A 可覆盖股票的缓存覆盖率报告；实际不足 5000 支时必须说明原因和补选逻辑。
- `data/date_generalization_cache/market_5000/` 中不得包含真实 key/token，缓存元数据只记录来源类型、请求时间、接口名和覆盖率。

工作流验收：

- 用户发起单支股票任务时，系统明确走 `single_stock_watch`。
- 用户发起多股组合任务时，系统明确走 `portfolio_pool_optimize`。
- 缺数据时如实报告，不崩溃，不胡编。
- 用户端输出清晰，不模棱两可；必须给买入、卖出、加仓、减仓、持有、等待或补数据建议，以及仓位/阈值、证据、反证和复评条件。
- 单支股票任务必须能展示目标股同行、同概念、同地域、相关走势股票和新闻/公告相对强弱。
- 多股组合任务必须能说明候选池来源、筛选 gate、TopN 选择原因、现金防守原因和放弃决策原因。

数据与 skill 验收：

- 至少完成 5000 支股票或全 A 可覆盖股票的数据 coverage audit。
- 相关股票通道至少包含同行、同概念、同地域、历史相关性 TopK 四类；缺任一类必须说明原因。
- Book Skill 高频触发 Top20 必须完成来源 grounding audit；无法 grounding 的策略不得提高优先级。
- 至少形成 20 条 `candidate_skill_seed`，来源可以来自书籍深挖、一手公开信息或公开社区经验，但必须全部标注 `unverified` / `accepted` / `rejected` / `watching`。
- 社区/论坛经验不得直接进入最终策略，必须先通过至少两个半年块和三组股票 sample 验证。
- 每条 accepted skill 必须写入适用条件、失效条件、后验 20 日正收益率、20 日均值、样本数和反证样本。

策略验收：

- 至少完成一轮组合正收益率导向的系统实验。
- 至少完成三组不同股票 sample 验证。
- portfolio_pool 相对当前 gate 轮必须至少改善一个核心指标：
  - raw 正收益率提高，或
  - raw 均值提高且大亏率下降，或
  - 在 raw 正收益率不下降的前提下，现金混合均值提高且覆盖率合理。
- 指标汇报必须区分：
  - `raw_positive_20d_rate`：实际选中组合后的原始 20 日正收益率，优先用于证明选股能力。
  - `cash_blended_avg_return_20d`：未触发策略时转现金的体验口径，不能单独证明选股能力。
  - `decision_coverage`：触发决策日期占可决策日期比例，过低时不得宣称策略稳定。
- 如果不能改善，必须输出失败复盘和下一步策略，不得宣称通过。

最终目标仍是：

- 最新成熟 test 20 日正收益率不低于 0.65。
- 往期半年块每块不低于 0.60。
- 将来 zeroshot 不低于 0.60。
