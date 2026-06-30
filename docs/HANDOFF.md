# 项目交接说明

本项目是 A 股研究 Agent，输出研究辅助型操作建议，不自动交易，不接券商接口，不下单。用户端必须先给明确买入/卖出/加仓/减仓/持有/等待/补数据建议，再给仓位或阈值、依据、反证、失效条件和复查条件。

## 用户入口

首选阅读：

- `docs/START_HERE.md`
- `docs/USER_GUIDE.md`
- `reports/date_generalization/final_user_manual.md`
- `reports/date_generalization/final_capability_report.md`
- `reports/date_generalization/final_product_readiness_audit_v1.md`

交互向导：

```bash
python -m src.user_wizard
```

盘中/实时盯盘单次复核：

```bash
python scripts/run_live_watch_session.py --code 000001 --name 平安银行 --interval-seconds 1200 --max-iterations 1
```

用户手册只保留 Markdown，不再生成 PDF。

## 当前能力状态

- P0 单支盯盘：`strong_yellow_mvp`。当前默认工作流冻结为 `single_stock_small_entry_watch_v3`：小仓分支先由 Python/ML 给出草案，再由 Agent 审计硬反证/软缺口。已验证的产品底座是“小仓试探/持有/减仓复核”，不是 broad active-buy。`PPS-Q-017 + softgap v2` 三面板 Flash target_cash_pos20 `0.8434±0.0501`、avg20 `+1.9139±0.0990pp`；general-channel 小仓三面板 Flash pos20 `0.8213`、avg20 `+2.6015±0.5526pp`。Pro 可用但未超过 Flash，Flash 仍是训练/批量回测默认。
- P1 多股候选对比：`default_ready_yellow`。默认 `candidate_comparison_ranker_anchor_v2`，三面板 Flash Top1/Top2 超额均值 `+3.5229pp` / `+1.5098pp`，仍需 Pro/滚动确认。
- P2 组合/截面：只作为底层工具和策略研究，不覆盖 P0/P1 产品结论。广义全市场 active-buy 仍未稳定达到 `0.60/0.65`，后续 20h 交付不得继续围绕它无限发散。

## 输出口径

研究分级只作为辅助标签，不再作为用户端唯一结论。第一句必须是操作建议：

- 买入/试探买入：给仓位上限、证据组合、止损/复评阈值。
- 加仓/提高仓位：给提高条件、仓位区间和撤回条件。
- 持有/继续持有：给继续持有条件、跌破/反证处理。
- 减仓/卖出：给触发风险、退出阈值和重新评估条件。
- 等待/暂不操作：必须写清什么条件转为买入/加仓，什么条件转为减仓/卖出。
- 信息不足：列出关键缺口，补齐前不硬做方向判断。

回测内部可以使用模拟动作/权重来衡量 20 日表现、错升、错降和大亏回避；用户端可以输出操作建议，但不得自动下单、不得连接券商、不得承诺收益或目标价必达。

## 后续推进

1. 20h 内优先收尾产品交付：更新 `final_product_workflow.md`、`final_capability_report.md`、`final_user_manual.md`，确保用户端第一句是明确操作建议、仓位、阈值和复查条件。
2. P0 以小仓盯盘工作流为默认；`p0_action_label_scorer_v1` 只作为 checklist 证据，不作为 standalone broad gate。
3. P1 v2 做 Pro 或滚动新数据确认；自由 Agent 排序不作为默认。
4. 每 1-3 个月拉新数据，重跑 leakage、coverage、readiness 和用户手册生成；行情切换时先降级为复核/排雷，不硬推研究暴露。

最新低成本实测：

- `scripts/audit_analogue_case_context_v2.py` 已验证时间安全相似案例 RAG/K 线同行上下文；报告见 `reports/date_generalization/analogue_case_context_v2_findings.md`。
- `analogue_case_context_flash_micro_v1` 已完成 evidence 接入、消融隔离和 8-card DS Flash micro；报告见 `reports/date_generalization/analogue_case_context_flash_micro_v1_findings.md`。
- `analogue_case_context_flash_3panel_v1` 已完成 48-card DS Flash on/off 扩样和通用 paired 诊断；报告见 `reports/date_generalization/analogue_case_context_flash_3panel_v1_findings.md` 与 `reports/date_generalization/analogue_case_context_flash_3panel_v1_paired.md`。
- 结论是“上下文/checklist 可用，不能当 alpha”：可提醒历史相似 base-rate、regime 衰减和失败案例，但不得单独提高研究分级，必须和新闻、财报、同行、BookSkill、量化工具共同确认。48-card 扩样均值小幅正，但出现 `lowered_positive` 和 `raised_negative`，因此不升权。
- `single_stock_kline_frequency_tool_v1` 已完成本地无 DS 的 P0 单支 K线/相关股票 K线频率审计；报告见 `reports/date_generalization/single_stock_kline_frequency_tool_v1.md`。结论：固定 `rev_chip_core` 可保留为观察/排雷 checklist，weekly Friday 在 H2026 有小幅改善；学习型短/长/震荡/同行/多尺度 K线在 H2026 明显失效，不得升为独立 alpha。下一步优先找新闻、财报、事件、BookSkill 和同行反证的正交通道，而不是继续堆价格派生特征。
- `nonprice_risk_overlay_v1` 已完成本地无 DS 的非价格风险覆盖层审计；报告见 `reports/date_generalization/nonprice_risk_overlay_v1.md`，Agent preview 见 `reports/date_generalization/nonprice_risk_overlay_v1_agent_preview.jsonl`。结论：新闻/财报/同行/BookSkill 不升为正向 alpha，但可指导冲突处理。同行/地域弱和“近期无财报事件”在高 `rev+chip` 回撤反转候选里常是 false-veto 区，不得机械剔除；高风险新闻在该范围内应提示二次确认或降低置信度。
- `nonprice_risk_overlay_3panel_flash_v1` 已完成 3-panel Flash on/off 扩样；报告见 `reports/date_generalization/nonprice_risk_overlay_3panel_flash_v1_findings.md` 与 `reports/date_generalization/nonprice_risk_overlay_3panel_flash_v1_paired.md`。结论覆盖早期 24-card smoke：portfolio_pool 均值仍为正，但 single_stock 三个 panel 全部为负，paired delta `-2.049474`，并出现 `raised_negative=4`、`lowered_positive=2`。因此 broad overlay 不再作为 P0 默认输入；CLI 默认仅对 `portfolio_pool` 可见。P1/P2 可保留为候选池 conflict checklist，不升为 alpha。
- `p0_acceptance_single_default_pro_v1` 已完成 36-card DeepSeek Pro 同样本确认；报告见 `reports/date_generalization/p0_acceptance_single_default_pro_v1_findings.md`。结果：invalid=0，总 token `379,074`，总体 20 日 cash-adjusted 正收益率 `0.7500`，与 Flash 持平；H2026_1 仍为 `0.5000` 且平均收益为负。因此 P0 可作为 MVP 底座，但 goal 不能标记完成，下一步优先滚动最新块。

## 不要清理或泄露

- 不打印、不复制、不提交 `ds_api.txt`、`tushare_token.txt` 或任何 key/token。
- 不删除、不移动、不覆盖原始书籍 PDF 或 BookSkill source。
- 不把未来收益、GT 字段或后验标签放进 evidence pack。
- 不删除 `data/date_generalization_cache/`、`reports/date_generalization/` 中关键 CSV/JSONL 回测证据，除非先确认已经归档。
