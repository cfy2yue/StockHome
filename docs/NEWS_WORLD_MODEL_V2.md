# News World Model V2

本文件定义新闻/公告通道如何被量化为 Agent evidence，用于支持主 Agent 输出研究辅助型操作建议；新闻单通道不自动下单、不承诺收益。

## 核心原则

- 所有事件必须满足 `available_at <= decision_time`。
- 新闻缺失不是中性好消息；缺失率进入 `news_missing_rate` 并降低置信度。
- 新闻通道分两层：先做可复现的事件/关键词统计，再由 DeepSeek 按固定问卷阅读时间安全材料并输出语义向量。
- 单支模式使用更细粒度的自身、同行、地域、政策、社区信号。
- 组合模式强调相对值：自身 vs 同行、同行活跃自身沉默、政策覆盖差、风险扩散。

## 字段定义

| feature_name | description | range | calculation | missing_policy | enter_evidence_pack | leakage_guard |
| --- | --- | --- | --- | --- | --- | --- |
| self_news_intensity | 股票自身新闻/公告强度 | 0-1 | min(1, log1p(self_count_30d)/log1p(20)) | missing -> 0 and raise news_missing_rate | True | available_at <= decision_time |
| peer_news_intensity | 同行新闻/公告强度 | 0-1 | min(1, peer_group_news_count_avg/10) | missing -> peer_unknown | True | available_at <= decision_time |
| policy_background_score | 政策背景 | -1..1 | industry_policy materiality normalized | missing -> 0 but lower evidence quality | True | available_at <= decision_time |
| region_background_score | 地域背景 | -1..1 | region event score, capped unless self event exists | missing -> 0 | True | available_at <= decision_time |
| self_vs_peer_attention_gap | 自身相对同行关注差 | -1..1 | self_news_intensity - peer_news_intensity | missing peer -> 0 and flag | True | available_at <= decision_time |
| peer_active_self_silent_flag | 同行活跃但自身沉默 | 0/1 | peer_news_intensity high and self_news_intensity low | missing -> 0 with warning | True | available_at <= decision_time |
| news_warning_score | 风险预警 | 0-1 | risk/regulatory/financing/holding-change materiality | missing -> 0 with missing-rate penalty | True | available_at <= decision_time |
| news_opportunity_score | 机会信号 | 0-1 | order/capacity/product/policy opportunity materiality | missing -> 0 | True | available_at <= decision_time |
| news_evidence_quality | 证据质量 | 0-1 | official source ratio + timestamp quality | missing -> low quality | True | available_at <= decision_time |
| news_missing_rate | 新闻缺失率 | 0-1 | missing expected source slots / expected slots | missing -> 1 | True | available_at <= decision_time |
| news_timestamp_quality | 时间戳质量 | 0-1 | available_at completeness and <= decision_time | missing -> 0 | True | available_at <= decision_time |
| news_peer_diffusion_score | 同行新闻扩散 | -1..1 | peer risk/opportunity co-occurrence around target | missing -> 0 | True | available_at <= decision_time |
| official_confirmation_score | 官方确认度 | 0-1 | official_count/(official_count+public_count) | missing -> 0 | True | available_at <= decision_time |
| community_attention_score | 社区关注度 | 0-1 | public/community count zscore capped | missing -> 0 | False | available_at <= decision_time |
| community_crowding_risk | 社区拥挤反证 | 0-1 | high positive crowding + high RSI/overheat | missing -> 0 | True | available_at <= decision_time |
| announcement_materiality_score | 公告重要性 | 0-1 | max official announcement materiality | missing -> 0 | True | available_at <= decision_time |

## DeepSeek 语义问卷层

配置入口：`config/news_deepseek_questionnaire.yaml`，说明文档：`docs/NEWS_DEEPSEEK_QUESTIONNAIRE.md`。

问卷层要求 DeepSeek 先阅读当前决策点可用的自身、同行、政策、地域新闻/公告材料，再回答 32 个固定问题。问卷层输出不是单独操作结论，而是以下派生字段：

- `ds_news_risk_score`
- `ds_news_opportunity_score`
- `ds_news_peer_support_score`
- `ds_news_policy_support_score`
- `ds_news_region_support_score`
- `ds_news_uncertainty_score`
- `ds_news_quality_score`
- `ds_news_net_score`

回测中必须做 `no_news`、`keyword_only`、`questionnaire_only`、`keyword_plus_questionnaire`、`risk_only_questionnaire` 消融。当前 combined 新闻接入尚未证明 alpha，因此问卷 V1 默认先作为风险、缺失和主线解释层，只有跨时间块稳定优于对照后才能升级为正向权重。

## 当前覆盖判断

- 当前 ground truth 行数：151235。
- 当前已有新闻相关原始字段：35 个。
- Tushare `anns_d` bounded 扩展已生成 `news_event_table.csv`：当前 25494 条 available-at-safe 公告事件，覆盖 5 个有效公告日期；2025-04 财报季 4 个有效分片各返回 6000 行，已在 `tushare_data_coverage` 标注为 `possible_row_cap_requests=4`。
- 已派生 `news_world_model_event_features.csv`：2925 条股票-日期特征；特征行保留最大 `available_at`，下游仍必须过滤 `available_at <= decision_time`。
- 本地 `data/backtest_scale_500/*/news.json` 东方财富/AKShare 公开聚合缓存已归一化为 `local_news_event_table.csv`：15163 条事件、4685 条股票-日期特征，覆盖 200 只已有新闻缓存股票，来源标注为 `public_aggregator`。
- 当前默认 evidence pack 上游使用 `combined_news_world_model_event_features.csv`：40657 条 combined 事件、7564 条股票-日期特征。
- 派生新闻/公告特征已接入 DeepSeek evidence pack 上游：`load_ground_truth()` 会对同一股票做 30 天 as-of join，只使用 `available_at <= decision_date 15:00:00` 的事件；当前 155690 个 GT 决策行中 9920 行匹配到事件窗口，`news_missing_rate` 均值 0.9363。
- 固定审计入口：`scripts/audit_news_event_feature_join.py`，用于每次扩展公告/新闻缓存后重算覆盖、缺失率和 evidence-pack smoke。
- Tushare `news` 和 `major_news` 返回独立权限未开通；本地公开聚合新闻可补近期覆盖，但不等同于交易所/巨潮官方原文闭环，社区源仍未覆盖。
- 阶段性实操结论：`local_news_round_experiment` 显示 combined 新闻接入没有提升组合最佳指标，`no_event_join` 最佳 `rank_score=0.6269` 高于 combined 最佳 `0.6243`；新闻当前只能作为风险/不确定性证据，不能宣称 alpha 优势。
- 历史新闻/公告仍偏稀疏，不能仅凭新闻通道宣称模型优势；高密度公告日可能触及接口行数上限，必须更细分或补充来源，并在 DeepSeek/ablation 稳定显示贡献后再升级新闻 gate。
