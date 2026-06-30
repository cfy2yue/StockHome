# DeepSeek 新闻语义问卷 V1

本项目的新闻通道分两层：第一层是可重复的事件/关键词统计，第二层是 DeepSeek 对时间安全新闻材料的语义问卷打分。两层都用于研究辅助型操作建议，但新闻问卷本身只输出结构化语义向量，不直接给买入/卖出/加减仓结论；最终操作建议必须由 Agent 结合量价、财报、同行、BookSkill、筹码和反证后生成。

## 现有提取流

1. 数据源进入本地缓存：Tushare 公告缓存、本地东方财富/AkShare 公开聚合新闻缓存，以及后续可接入的标准化/会员数据源。
2. `src/world_model/news_event_table.py` 把原始缓存归一化成事件表：代码、发布时间、可用时间、来源类型、事件类型、标题、摘要、链接、风险/机会/政策关键词分、官方确认度、证据质量。
3. 同一事件表再按股票和日期聚合成 `news_world_model_event_features.csv`，生成 `self_news_intensity`、`news_warning_score`、`news_opportunity_score`、`policy_background_score`、`official_confirmation_score`、`news_evidence_quality` 等字段。
4. `src/world_model/financial_report_channel.py` 把财报、业绩预告/快报、审计意见、问询/修正公告等派生成 `financial_report_events.csv` 和 `financial_report_features.csv`。这是高可信新闻类子通道，独立于普通新闻机会分。
5. `load_ground_truth()` 在构建回测决策行时做 30 天新闻 as-of join 和 90 天财报 as-of join，只允许 `available_at <= decision_date 15:00:00` 的材料进入 evidence pack。
6. `src/agent_training/evidence_pack.py` 把有界 V2 新闻字段放进 `news_features`，把财报事件字段放进 `financial_report_features`，DeepSeek 决策时同时看到 Python gate、Book Skill、memory 和反证。

当前问题是：第一层适合快速统计，但关键词不一定理解语义。例如“收到问询后回复完成”可能不是强负面，“中标”也要看金额、客户和公司规模。因此需要第二层语义问卷。

## 问卷层目标

DeepSeek 先阅读同一决策点可用的新闻/公告材料，然后回答固定 32 个问题，输出结构化 JSON。问卷不直接给研究分级，而是产出可回测的语义向量：

- `ds_news_risk_score`
- `ds_news_opportunity_score`
- `ds_news_peer_support_score`
- `ds_news_policy_support_score`
- `ds_news_region_support_score`
- `ds_news_uncertainty_score`
- `ds_news_quality_score`
- `ds_news_net_score`

这些字段进入下一步 Agent 决策和 ablation。若 full_news 相比 no_news 没有稳定提升，问卷字段只能作为风险/不确定性证据，不能宣称新闻 alpha。

问卷的正确顺序不是“看到关键词就打分”，而是三步：

1. **主线预读**：DeepSeek 先把过去窗口内的自身新闻、同行新闻、政策/行业背景、地域背景归纳成 0-3 条主线，并标记是否有官方确认、金额/客户/披露日、传导链和重复转载。
2. **固定问卷**：再按 32 个问题给数值分。每个分数必须落在固定范围内，缺来源、缺时间、缺正文时要提高不确定性，不能脑补。
3. **派生向量**：最后输出风险、机会、同行支持、政策支持、地域支持、不确定性、质量、净分 8 个派生分，用于后续 Agent 决策和消融。

因此 DeepSeek 的强项被用在“理解新闻主线和冲突”，不是替代行情、Book Skill 或财务披露。新闻机会分只有在主线清晰、决策相关、质量足够，并且得到 Book Skill、Python gate、同行/政策和财务披露交叉确认时，才允许作为正向候选证据。

## 当前实测结论

截至 2026-06-25，问卷已完成一个 50-pack 小型 round，并和此前 smoke 合并为 63 个唯一 matched-news 决策点。结论是：

- 关键词层继续保留：它负责覆盖率、时间安全、来源类型、事件数量和风险/机会关键词的可复现统计。
- 问卷层必须保留：它能读出“定增 + 监管处罚”“重大重组 + 股东质押”“常规公告很多但没有经营主线”这类关键词容易误判的语义。
- 机会分暂不升权：`ds_news_opportunity_score` 和 `ds_news_net_score` 在 63-row panel 中没有证明正向 alpha，不能直接作为正向 gate。
- 风险/不确定性先做反证：`ds_news_uncertainty_score < 0.6` 的样本 raw 正收益率从 baseline 0.5238 提到 0.5472，亏损超 5% 从 0.2698 降到 0.2453；提升较小，只能记为 observe。
- `risk_or_uncertainty_safe` 可提升现金防守后的正收益率到 0.6667，但 raw 均值低于 baseline，所以只能作为“避开低质量日期/弱证据”的辅助条件。
- 每道问题的分数已写入 scores CSV，后续可按问题级别做消融和升降权，而不是只看总分。

因此，当前策略不是“新闻越好越加分”，而是“新闻先解释主线，再找反证；正向主线必须由 Book Skill、量价、同行/行业和财务披露共同确认”。

2026-06-26 已完成财报/业绩公告事件通道的轻量接入：

- 4423 条时间安全财报事件、1470 条股票-日期特征、覆盖 1337 只股票。
- 90 天 as-of GT 匹配率为 3594 / 155690 = 0.0231，说明当前覆盖仍稀疏。
- 财报特征已作为独立 `financial_report_features` 进入 evidence pack，但尚未跑财报通道消融。
- 下一步必须比较 `no_financial_report_channel`、`financial_report_only`、`news_plus_financial_report`，验证它对单支模式和组合模式是否有稳定增益。
- 在验证前，财报通道只作为高可信复核/不确定性通道；缺失不能当好消息，普通新闻与官方披露冲突时官方披露优先。

随后又把问卷放入完整 Agent 决策做 lpm4 小样本消融，结果更保守：

- `no_news` 在该 shard 的单股和组合现金调整指标反而最好，说明新闻层当前不能直接宣称正向 alpha。
- `questionnaire_only` 和 `keyword_only` 都出现过把常规公告/弱主线误当正向证据的失败，最典型样本 20 日结果为 -11.26%。
- `risk_only_questionnaire` 更少触发正向动作，说明问卷当前更适合作为风险、不确定性和解释通道，而不是单独选股通道。
- 新增保护规则：若 `ds_news_opportunity_score >= 0.7`，但 `ds_news_mainline_clarity < 0.5` 或 `ds_news_decision_relevance < 0.5` 或 `ds_news_repetition_lag >= 0.6`，则机会分降为背景信息，不允许提升研究分级。

保护规则先在 `deepseek_news_guarded_lpm4_v1` 中出现小样本改善，但随后 `deepseek_news_guarded_lpm20_v1` 扩样复测没有泛化：guarded 组合模式 cash-adjusted 20 日均值 -0.0267，低于完整问卷 0.0606；guarded 单支模式 0.0885，低于完整问卷 0.2226。扩样还发现，cap 触发后 DeepSeek 仍可能绕过反证、基于 Python 信号给出过高动作。因此项目已新增执行层 guardrail：cap 触发且 Book Skill/财报披露日确认缺口存在时，最终卡片不得保留 `继续深挖` 或 `增加研究暴露`。该规则是安全护栏，不是正向收益公式。

2026-06-26 进一步把问卷拆成 risk / uncertainty / quality 子通道，并补跑 `keyword_only` / `questionnaire_only` 完整对照。`deepseek_news_risk_uncertainty_quality_lpm20_v1` 与 `deepseek_news_keyword_questionnaire_controls_lpm20_v1` 合计 459 张 DeepSeek Flash 决策卡，invalid=0。该轮只支持候选观察，不支持冻结策略：

- 组合模式：`uncertainty_only_questionnaire` cash-adjusted 20 日均值 0.3971，优于完整问卷 -0.6435、`no_news` -0.5563、`keyword_only` -0.0060 和 `questionnaire_only` 0.0086。它只说明本 shard 中可能减少高机会分干扰，不是正向新闻公式。
- 单支模式：`quality_only_questionnaire` cash-adjusted 20 日均值 0.3234，优于完整问卷 0.2200、`no_news` 0.1823、`keyword_only` 0.2520 和 `questionnaire_only` 0.2632；`semantic_risk_only_questionnaire` 为 0.2681。由于单支模式 exposure=0，该结果只说明可能改善复核/降权路径，不能证明正向暴露能力。
- 完整问卷仍有失败样本：`000681` opportunity=1/net=0.5，叠加 Python pullback 后增加研究暴露，后验 20 日 -14.05%。因此 `ds_news_opportunity_score` 和 `ds_news_net_score` 继续不得作为正向 alpha。
- 该结果仍只覆盖 2025/2026 matched-news 窗口，组合暴露样本很少，不能宣称日期泛化或稳定排序通过。

## 32 个问题分组

| 分组 | 问题数 | 作用 |
|---|---:|---|
| 证据质量 | 4 | 判断材料是否足够、来源是否可靠、时间戳是否安全 |
| 股票自身 | 7 | 判断公司公告、订单、业绩、监管、供应链等自身事件 |
| 同行/产业 | 6 | 判断同行扩散、目标股相对同行是否被支持或沉默 |
| 政策/宏观 | 5 | 判断政策顺风/逆风、景气周期、汇率/利率/贸易影响 |
| 地域背景 | 3 | 判断地方政策、地区风险、区域产业集群验证 |
| 冲突/拥挤 | 4 | 判断正负冲突、叙事拥挤、旧闻重复、信息新颖性 |
| 跨通道对齐 | 3 | 判断新闻与 Book Skill、Python gate、最终研究分级相关性 |

完整机器可读配置见 `config/news_deepseek_questionnaire.yaml`。

## 回测优化方式

每一轮 round 不能只看问卷分数本身，而要做消融：

- `no_news`：完全屏蔽新闻。
- `keyword_only`：只使用关键词/事件统计。
- `questionnaire_only`：只使用 DeepSeek 问卷派生分。
- `keyword_plus_questionnaire`：两层同时使用。
- `risk_only_questionnaire`：只允许风险/不确定性进入，机会分不加正向权重。
- `uncertainty_only_questionnaire`：只暴露主线清晰度、来源/时间戳、冲突、重复滞后和不确定性。
- `quality_only_questionnaire`：只暴露来源覆盖、官方确认、时间戳、主线相关性和证据质量。
- `semantic_risk_only_questionnaire`：隐藏关键词统计，只暴露 DeepSeek 语义风险字段。
- `risk_uncertainty_questionnaire`：保留风险关键词和语义风险/不确定性字段，继续隐藏机会/净分。

每个问题都要进入 ledger：在哪些时间块有效、在哪些任务模式有效、对 20 日 raw 正收益率和均值是否有提升、失败样本是什么。只有跨时间块、跨 sample 稳定有效的问题，才能升级权重；近期窗口有效但早期窗口失效的问题必须降权。

当前候选阈值只允许以 observe 进入下一轮：

- `ds_news_uncertainty_score >= 0.6`：高不确定性，要求更多证据或降权。
- `ds_news_risk_score >= 0.6`：高风险反证，必须解释风险来源。
- `ds_news_risk_score < 0.6 and ds_news_uncertainty_score < 0.6`：可作为通过新闻反证检查的辅助条件，但不能单独提高分级。
- `ds_news_net_score >= 0`：当前不接受为正向选股规则。
- `ds_news_opportunity_score >= 0.4`：当前不接受为正向选股规则，除非同时有 Book Skill、Python gate、同行/行业和财务披露确认。
- `ds_news_opportunity_score >= 0.7` 且主线清晰度/决策相关性不足：触发 `news_questionnaire_routine_announcement_positive_cap_v1`，机会分封顶为背景信息；若同时存在 Book Skill 未解析或财报披露日缺失，则执行层不得输出 `继续深挖` 或 `增加研究暴露`。这是安全护栏，不是正向收益公式。
- `uncertainty_only_questionnaire` 在组合模式暂列 observe 候选：下一轮必须扩大样本和时间块，不能因 11 张组合卡冻结策略。
- `quality_only_questionnaire` 在单支模式暂列 observe 候选：可用于判断复核强度、来源可信度和时间安全，但不能替代财报披露确认。

## 决策边界

- 新闻缺失不能当低风险。
- 公开聚合新闻不能等同于官方原文。
- 只有日期、没有具体时间的材料必须进入时间戳质量惩罚，并做保守可用时间对照。
- DeepSeek 问卷输出不能包含未来收益、未来事件或后验标签。
- 新闻层永远只是研究分级和反证的一部分，必须和 Book Skill、同行、Python gate、memory 一起判断。
