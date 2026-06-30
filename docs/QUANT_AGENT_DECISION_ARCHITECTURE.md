# Quant Tool + Agent Decision Architecture

本项目输出 A 股研究辅助型操作建议，不自动交易，不接券商接口，不下单，不承诺收益。用户端可以输出买入、卖出、加仓、减仓、持有、等待或补数据建议；`继续深挖`、`放入观察`、`暂时剔除`、`信息不足` 只作为辅助研究分级。

## Why

回测后的自然语言反思不能稳定形成有效“反传”。下一阶段把训练闭环改为：

```text
后验标签评估 -> 定量 tool 训练/校准 -> tool 输出压缩摘要和 rule_outcome -> Agent 综合判断 -> 回测复核
```

Agent 的优势是多通道整合、冲突判断、Book Skill/新闻/财报解释和把内部证据转成清晰操作建议；定量 tool 的优势是大样本训练、阈值校准、风险识别和可复现排名。两者必须分工，而不是让 Agent 直接从自然语言经验里“学会”策略。

## Decision Frequency

当前已实现并可回测的是日线级别：

- `twice_weekly`：周二/周五附近决策点。
- `weekly_friday`：每周五。
- `weekly_tuesday`：每周二。
- `every_2_weeks`：每两周一次。
- `date_gate_only`：仅 date gate 通过时决策。

小时/分钟级暂不进入主训练。当前可靠数据底座是日线和 15:00 as-of join，小时级只能先汇总为 event digest，再流入下一个日线决策点。

## Policy Profiles

- `long_horizon_review`：月度、财报披露后、重大公告后复核；重财务质量、估值压力、Book Skill、行业周期。
- `mid_horizon_research`：周/半周/每两周主线；重 K 线多尺度、同行/相关股票、新闻/公告、财报风险、组合 TopN。
- `fast_event_digest`：事件触发；只判断是否降级、补证、暂停观察或推迟到下一个正式决策点。

三类 profile 共享 feature store、as-of manifest、label schema 和 audit，不各自另起系统。

## Decision Points

每个决策点必须标注：

- `decision_point_type`: `scheduled`、`event_triggered`、`portfolio_rebalance`、`user_requested`
- `decision_priority`: 0-3
- `trigger_reason`
- `trigger_channel`
- `available_at`
- `source_ref_ids`
- `cooldown_group`
- `sampling_weight`

关键点包括：重大公告/财报事件、新闻风险或缺失异常、价格结构突变、同行/市场 regime 切换、组合 TopN 进出、用户点名股票。

防爆炸规则：单股每周最多 1 个非重大关键点；重大公告可覆盖 cooldown；组合每日最多保留 TopK 变化点；同源同事件去重。

## Labels

单支模式 label：

- `increase_research`
- `watch`
- `reduce_or_exclude`
- `insufficient`

组合模式 label：

- `top_candidate`
- `neutral`
- `avoid`
- `skip_date_to_cash`

label 只用于训练和评估 tool，不得进入 DeepSeek evidence pack、memory prompt 或用户报告原始材料。

## Quant Tools

第一批工具：

- `decision_point_sampler`
- `date_regime_gate`
- `single_stock_risk_opportunity_score`
- `portfolio_ranker`
- `drawdown_risk_guard`
- `source_quality_tool`
- `peer_graph_tool`
- `channel_ablation_scorer`

Tool 输出给 Agent 的格式：

- `tool_id`
- `tool_version`
- `policy_profile`
- `score`
- `score_quantile`
- `confidence`
- `action_hint`
- `top_features`
- `missing_flags`
- `counter_evidence`
- `source_ref_ids`
- `train_valid_test_blocks`
- `promotion_status`

任何单一 tool 分数不得单独触发买入/加仓或默认组合 TopN；必须经过 Agent 对新闻、财报、同行、BookSkill、K线/筹码、数据缺口和反证的综合审计。

## Minimum Experiment

1. 用现有 joined GT/feature cache 构建 `decision_point_table`，不新增 API 请求。
2. 对比 `scheduled_every_2_weeks`、`weekly`、`twice_weekly`、`key_points_only`、`scheduled_plus_key`。
3. 生成单支和组合 label/action 表。
4. 训练或校准 additive-bin、rank-formula、轻量 tree/logistic 工具。
5. 生成 48-96 个双任务 evidence pack，先跑 leakage/channel/governance audit。
6. 小样本 Flash 只比较 `python_only`、`quant_tool_summary_only`、`full_agent_with_quant_tools`。

不通过则不扩大。
