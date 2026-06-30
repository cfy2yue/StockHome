from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

import pandas as pd

from src.agent_training.book_skill_resolver import resolve_book_skill_candidates
from src.agent_training.decision_card import (
    ALLOWED_RESEARCH_GRADES,
    ALLOWED_SIMULATED_ACTIONS,
    normalize_action_weight,
    sanitize_decision_card_text_fields,
)
from src.agent_training.quant_tool_context import quant_tool_summary_text, sanitize_quant_tool_outcome


FUTURE_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "gt_status",
    "gt_pass",
}


PYTHON_FEATURE_FIELDS = [
    "timeline_score",
    "relative_strength_rank",
    "close_above_ma200",
    "counter_score",
    "rsi14",
    "prior_return_20d",
    "atr20_pct",
    "drawdown60",
    "ma200_slope20",
]

KLINE_FEATURE_FIELDS = [
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_return_120d",
    "kline_return_240d",
    "kline_volatility_20d",
    "kline_volatility_60d",
    "kline_volatility_120d",
    "kline_volatility_ratio_3_20",
    "kline_volatility_ratio_5_20",
    "kline_volatility_ratio_20_60",
    "kline_volatility_ratio_20_120",
    "kline_volatility_ratio_60_120",
    "kline_rsi14",
    "kline_macd_hist",
    "kline_atr20_pct",
    "kline_drawdown_60d",
    "kline_drawdown_120d",
    "kline_drawdown_240d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_range_position_120d",
    "kline_range_width_pct_20d",
    "kline_range_width_pct_60d",
    "kline_range_width_pct_120d",
    "kline_mean_reversion_z20",
    "kline_trend_consistency_20d",
    "kline_trend_consistency_60d",
    "kline_trend_consistency_120d",
    "kline_efficiency_ratio_20d",
    "kline_efficiency_ratio_60d",
    "kline_efficiency_ratio_120d",
    "kline_oscillation_cross_count_20d",
    "kline_oscillation_cross_count_60d",
    "kline_oscillation_cross_count_120d",
    "kline_direction_reversal_rate_20d",
    "kline_direction_reversal_rate_60d",
    "kline_signed_streak_norm_20d",
    "kline_signed_streak_norm_60d",
    "peer_kline_relative_to_group_20d",
    "peer_kline_group_positive_breadth_20d",
    "peer_kline_group_above_ma200_rate",
]

PEER_CONTEXT_FEATURE_FIELDS = [
    "tushare_industry",
    "tushare_area",
    "tushare_industry_group_size",
    "tushare_industry_avg_return_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_industry_news_warning_avg",
    "tushare_industry_news_opportunity_avg",
    "tushare_industry_news_attention_gap",
    "tushare_area_group_size",
    "tushare_area_avg_return_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "tushare_area_above_ma200_rate",
    "tushare_area_news_warning_avg",
    "tushare_area_news_opportunity_avg",
    "tushare_area_news_attention_gap",
]

CHIP_CORE_FEATURE_FIELDS = [
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
    "chip_core_source_type",
    "chip_core_source_name",
]


NEWS_FEATURE_FIELDS = [
    "news_count_30d",
    # News World Model V2 fields. Keep these as first-class evidence fields so
    # DeepSeek sees the same bounded 0-1 / -1..1 channel that docs/config describe.
    # Legacy unbounded *_30d risk/opportunity score columns stay available to
    # deterministic local experiments, but are not passed directly to DeepSeek.
    "self_news_intensity",
    "peer_news_intensity",
    "policy_background_score",
    "region_background_score",
    "self_vs_peer_attention_gap",
    "peer_active_self_silent_flag",
    "news_warning_score",
    "news_opportunity_score",
    "news_evidence_quality",
    "news_missing_rate",
    "news_timestamp_quality",
    "news_peer_diffusion_score",
    "official_confirmation_score",
    "community_attention_score",
    "community_crowding_risk",
    "announcement_materiality_score",
    "source_type",
    "source_name",
]

NEWS_QUESTIONNAIRE_FIELDS = [
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_mainline_clarity",
    "ds_news_source_coverage",
    "ds_news_official_support",
    "ds_news_timestamp_confidence",
    "ds_news_self_material_event",
    "ds_news_self_earnings_change",
    "ds_news_self_order_product",
    "ds_news_self_capital_financing",
    "ds_news_self_holder_change",
    "ds_news_self_regulatory_legal",
    "ds_news_self_supply_chain_position",
    "ds_news_peer_industry_heat",
    "ds_news_peer_relative_support",
    "ds_news_peer_risk_diffusion",
    "ds_news_peer_opportunity_diffusion",
    "ds_news_peer_silent_gap",
    "ds_news_cross_stock_confirmation",
    "ds_news_policy_tailwind",
    "ds_news_policy_headwind",
    "ds_news_cycle_demand_signal",
    "ds_news_macro_liquidity_signal",
    "ds_news_external_trade_fx_signal",
    "ds_news_region_policy_support",
    "ds_news_region_risk",
    "ds_news_region_cluster_confirmation",
    "ds_news_conflict_intensity",
    "ds_news_consensus_crowding",
    "ds_news_novelty",
    "ds_news_repetition_lag",
    "ds_news_bookskill_alignment",
    "ds_news_python_gate_alignment",
    "ds_news_decision_relevance",
    "ds_news_risk_score",
    "ds_news_opportunity_score",
    "ds_news_peer_support_score",
    "ds_news_policy_support_score",
    "ds_news_region_support_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_net_score",
    "ds_news_missing_or_conflict_notes",
    "ds_news_positive_capped_by_rule",
    "ds_news_positive_cap_rule_id",
    "ds_news_positive_cap_reason",
    "ds_news_original_opportunity_score",
    "ds_news_original_net_score",
]

FINANCIAL_REPORT_FEATURE_FIELDS = [
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
    "financial_report_latest_period",
    "financial_report_event_types",
    "financial_report_available_at",
    "financial_report_window_days",
    "financial_report_source_type",
    "financial_report_source_name",
    "financial_report_join_status",
]

TASK_MODE_ALIASES = {
    "portfolio_pool_optimize": "portfolio_pool",
    "portfolio_pool": "portfolio_pool",
    "single_stock_watch": "single_stock",
    "single_stock": "single_stock",
}


def build_evidence_pack(
    row: pd.Series | dict[str, Any],
    *,
    agent_policy_version: str,
    step: int,
    train_blocks: list[str],
    valid_block: str,
    task_mode: str = "portfolio_pool",
    variant: str = "deepseek_agent",
    available_at: str | None = None,
    python_candidate: str = "",
    memory_context: str = "none",
    retrieved_cases_context: str = "none",
    conflict_quality_context: str = "none",
    promote_context: str = "none",
) -> dict[str, Any]:
    data = dict(row)
    decision_date = str(data.get("date") or data.get("decision_date") or "")
    data_missing_flags = _data_missing_flags(data)
    pack = {
        "type": "agent_evidence_pack",
        "agent_policy_version": agent_policy_version,
        "variant": variant,
        "step": int(step),
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "decision_date": decision_date,
        "available_at": available_at or f"{decision_date} 15:00",
        "time_block": valid_block,
        "task_mode": task_mode,
        "case_memory_mode": _text(data.get("case_memory_mode") or "memory_compact_only"),
        "task_mode_requirement": _task_mode_requirement(task_mode),
        "code": str(data.get("code", "")).zfill(6),
        "name": _text(data.get("name")),
        "industry": _text(data.get("industry") or data.get("sector_group") or data.get("peer_group")),
        "peer_group": _text(data.get("peer_group") or data.get("industry") or data.get("sector_group")),
        "region": _text(data.get("region") or data.get("province")),
        "python_signal_summary": _python_summary(data, python_candidate),
        "sampler_context": _sampler_context(python_candidate),
        "python_features": _field_subset(data, PYTHON_FEATURE_FIELDS),
        "quant_tool_signal_summary": _quant_tool_summary(data),
        "quant_tool_summaries": _quant_tool_summaries(data),
        "quant_tool_requirement": "量化工具是训练/验证后的辅助层；若usable_in_agent_default=false或promotion_status未通过，只能作为灰色参考或反证。若工具为default_combo_ranker_yellow，score_quantile较高且没有明确负面新闻、财报风险、严重同行落后、过热或RAG失败等硬反证时，可在新闻/财报/BookSkill软缺口下支持partially_adopted和低权重观察；缺信息本身不是方向性负面。若不采用，必须说明具体硬反证。",
        "kline_signal_summary": _kline_summary(data),
        "kline_features": _field_subset(data, KLINE_FEATURE_FIELDS),
        "peer_context_signal_summary": _peer_context_summary(data),
        "peer_context_features": _field_subset(data, PEER_CONTEXT_FEATURE_FIELDS),
        "chip_signal_summary": _chip_summary(data),
        "chip_features": _field_subset(data, CHIP_CORE_FEATURE_FIELDS),
        "news_signal_summary": _news_summary(data),
        "news_features": _field_subset(data, NEWS_FEATURE_FIELDS),
        "news_semantic_questionnaire": _field_subset(data, NEWS_QUESTIONNAIRE_FIELDS),
        "financial_report_signal_summary": _financial_report_summary(data),
        "financial_report_features": _field_subset(data, FINANCIAL_REPORT_FEATURE_FIELDS),
        "book_skill_candidates": _book_skill_candidates(data),
        "book_skill_requirement": "DeepSeek决策前必须审阅Book Skill候选；若为空也要说明缺失原因。",
        "memory_context": memory_context,
        "retrieved_cases_context": retrieved_cases_context,
        "analogue_case_context": _analogue_case_context(data),
        "analogue_case_requirement": _text(data.get("analogue_case_requirement")),
        "nonprice_risk_overlay_context": _nonprice_risk_overlay_context(data),
        "nonprice_risk_overlay_requirement": _text(data.get("nonprice_risk_overlay_requirement")),
        "operation_plan_context": _operation_plan_context(data),
        "conflict_quality_context": _text(conflict_quality_context or "none"),
        "promote_context": _text(promote_context or "none"),
        "counter_evidence": _counter_evidence(data),
        "data_missing_flags": data_missing_flags,
        "allowed_research_grades": sorted(ALLOWED_RESEARCH_GRADES),
        "allowed_simulated_actions": sorted(ALLOWED_SIMULATED_ACTIONS),
        "research_only": True,
        "not_investment_instruction": True,
    }
    _assert_no_future_fields(pack)
    return _json_clean(pack)


def build_decision_messages(evidence_pack: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是A股研究辅助Agent，只能输出一个严格JSON对象，不要markdown，不要解释。"
        "你可以在理由中给买入、卖出、加仓、减仓、持有、等待或补数据等操作建议，但必须是研究辅助型建议，不能自动下单、不能承诺收益、不能写目标价必达/稳赚/必涨。"
        "用户端第一层需要明确操作建议，不允许只写继续深挖、放入观察、暂时剔除或信息不足。"
        "你必须在research_grade中选择辅助研究分级：继续深挖、放入观察、暂时剔除、信息不足。"
        "simulated_action只能选择：增加研究暴露、降低研究暴露、保持观察、转入现金、信息不足不动作。"
        "confidence_level必须是0到1的小数，simulated_weight_change必须是0到1的小数。"
        "simulated_action和simulated_weight_change必须一致：增加研究暴露=0.50到1.00，保持观察=0到0.20，降低研究暴露=0到0.10，转入现金=0，信息不足不动作=0。"
        "user_operation_suggestion必须从以下清晰动作中选择或组合：试探买入、加仓、持有、减仓、卖出/不买、等待不买、补数据后再定。"
        "target_position必须是0到1的小数；试探买入通常0.10到0.35，加仓/继续持有通常0.35到0.80，减仓通常0到0.35，卖出/不买和等待不买为0。"
        "若user_operation_suggestion为试探买入、加仓或高仓位持有，simulated_action应为增加研究暴露；若为减仓，simulated_action应为降低研究暴露；若为卖出/不买，simulated_action应为转入现金；若为补数据后再定，simulated_action应为信息不足不动作。"
        "final_agent_reasoning_summary不超过60个汉字，error_reflection不超过40个汉字。"
        "Book Skill是决策前必须审阅的核心材料；即使为空，也要在book_skill_evidence和理由中说明。"
        "book_skill_candidates若包含source_book、chapter、page_range、applicable_condition和failure_condition，必须优先引用这些来源与边界，而不是只复述strategy_id。"
        "若Book Skill候选为missing_grounded_card、needs_grounding或weak_until_grounded，只能作为弱线索，不能单独提高研究分级或研究暴露。"
        "若Book Skill与量价、新闻或memory反证冲突，必须在final_agent_reasoning_summary中说明取舍。"
        "若某类Book Skill在memory中被反复验证，应提高优先级；若被反证，应降权。"
        "retrieved_cases_context若不为none，表示本地RAG/案例检索找回的相似历史失败或已验证规则；只能作为规则状态、失败条件和下一步动作参考，不得把它当作后验收益或未来信息。"
        "analogue_case_context若不为空，表示时间安全的成熟历史相似案例审计摘要；它只能提供base-rate、regime衰减、失败案例和反证checklist，不能作为独立alpha、当前股票收益预测或单独提高研究分级的依据。"
        "若analogue_case_context提示relative_improvement，也只表示相对baseline的观察型改善，不等于绝对收益或正收益率达标；必须结合新闻、财报、同行、BookSkill、K线和筹码当前证据。"
        "nonprice_risk_overlay_context若不为空，表示本地walk-forward prior-only审计给出的非价格冲突处理提示；它只能用于防错杀、二次确认和置信度调整，不得作为独立alpha、交易指令或未来收益证据。"
        "nonprice_risk_overlay_context中action_hint=do_not_mechanically_veto表示该风险/软缺口历史上容易错杀，尤其是高rev+chip回撤反转候选的同行/地域弱或财报近窗口无事件，不得单独下调。"
        "nonprice_risk_overlay_context中action_hint=downweight_or_request_confirmation表示需要补充确认或降低置信度，尤其是高风险新闻；但必须先区分明确负面事件和可逆反转摩擦。"
        "nonprice_support_min2或其他非价格支持标签不能单独触发继续深挖或提高研究暴露，必须有当前多通道确认。"
        "conflict_quality_context若不为none，表示仅用当前决策点之前的训练块生成的冲突质量规则；可辅助判断冲突是反转摩擦还是降权风险。若来源不是walk-forward prior blocks，必须忽略以避免未来信息泄漏。"
        "promote_context若不为none，表示仅用当前决策点之前的训练块生成的正向升级候选规则；只能引用rule_status和agent_use，不得引用未来收益数字。"
        "sampler_context若不为none，表示本候选来自离线训练后形成的观察型采样器；它只能解释为什么此样本值得复核，不能替代新闻/财报/同行/BookSkill确认，也不能作为未来收益证据。"
        "operation_plan_context若不为空，表示本地确定性工作流给出的可执行动作草案；你必须审计它，而不是忽略它。"
        "若当前多通道证据支持operation_plan_context，应承接其动作和仓位；若不支持，必须在counter_evidence或final_agent_reasoning_summary中写明覆盖原因，并给出替代动作。"
        "operation_plan_context不是未来收益，也不是必须照抄；它是待Agent审计的工具建议。"
        "若operation_plan_context.operation_action=small_buy_hold，表示小仓试探/继续持有分叉，不是模糊观察；news_missing、financial_no_event_in_window、peer_weak、bookskill_observe_only这类软缺口只能降低仓位和置信度，不能单独把目标仓位归零。"
        "small_buy_hold只有遇到明确负面新闻/监管债务停产、财报质量风险或负惊喜、极端过热、筹码强上压、同行显著走弱且目标持续落后、或RAG相似失败等硬反证时，才覆盖为等待不买、卖出或信息不足。"
        "若small_buy_hold无硬反证但证据不够强，优先给试探买入/持有，target_position可降到0.10到0.25，并写清复核/减仓阈值；不要只输出放入观察而无操作。"
        "新闻通道分为关键词/事件统计和DeepSeek语义问卷；问卷结果只能作为研究证据，且必须结合新闻缺失率、来源质量、Book Skill、Python gate和反证。"
        "若news_semantic_questionnaire显示ds_news_positive_capped_by_rule=true，说明新闻机会分已因主线弱、决策相关性低或重复滞后被降为背景信息；不得据此提高研究分级。"
        "若ds_news_net_score为负主要来自news_missing_rate高、ds_news_uncertainty_score高或source_coverage低，而没有明确风险事件，这只是信息空窗/置信度折扣，不是看空或剔除理由。"
        "若news_branch_case_context不为空，表示本地离线工具把新闻问卷转成了分叉标签和prior-only相似案例元信息；它只能作为checklist、反证、置信度折扣或防false-veto材料，不得当作独立alpha或未来收益证据。"
        "news_branch_case_context中的reversible_reversal_friction表示高ranker反转候选遇到新闻风险/不确定性时需要区分可逆摩擦和明确负面事件；不得仅凭ds_news_risk_score或net_score硬否决。"
        "新闻缺失不得被解释为低风险，也不得被解释为确定负面；它应推动放入观察、降低置信度或要求补资料，不能单独压倒已验收量化工具。"
        "若新闻问卷与关键词统计冲突，必须说明冲突并降低置信度。"
        "财报/业绩公告是高可信新闻类事件通道，但必须只使用available_at不晚于决策时间的本地离线缓存。"
        "若financial_report_features显示财务质量风险、披露质量低、问询/修正/非标审计等，应作为反证复核；若财报通道真实缺失，只能增加不确定性，不能脑补正向或负向结论。"
        "普通新闻与财报/官方披露冲突时，官方披露优先，公开聚合新闻降权。"
        "若financial_report_features.financial_report_join_status=event_window_matched，说明本决策点已有时间安全财报/业绩公告事件，不得再声称财报披露日缺失；但仍可根据财报质量风险、负惊喜、Book Skill未解析等降权。"
        "若financial_report_features.financial_report_join_status=no_event_in_window，说明近窗口没有新的财报/公告事件，这是中性缺少正向确认，不等同于披露日缺失或数据源失败。"
        "必须先识别task_mode：portfolio_pool/portfolio_pool_optimize用于候选池排序和研究注意力分配，single_stock/single_stock_watch用于单只股票盯盘、复核和模拟路径。"
        "组合模式要关注相对排序、现金防守和同池比较；单支模式要关注持续持有观察、降级、信息缺口和风险复核。"
        "研究暴露是回测内部的研究注意力权重；用户端会把它翻译成买入/卖出/加减仓/持有/等待等操作建议和阈值。不得因为边界提示而自动把所有候选降为观察或现金。"
        "portfolio_pool中的单个证据包已经经过Python候选筛选；若Python信号较强且无高风险/高不确定性/严重数据缺口，可以给继续深挖和增加研究暴露。"
        "只有在新闻风险、不确定性、过热、财报真实缺失或财报风险、Book Skill失效或memory反证足以压倒候选优势时，才应降为观察、剔除或现金。"
        "量化工具层是机器学习/规则训练后形成的辅助工具；它不是最终决策者，必须由Agent结合Book Skill、新闻/财报、peer、K线和memory综合判断。"
        "若quant_tool_summaries中usable_in_agent_default=false、promotion_status包含observe或reject，或counter_evidence包含latest_time_block_failed/time_block_instability，只能作为灰色参考或反证；不得单独触发继续深挖或增加研究暴露。"
        "若quant_tool_summaries中portfolio_rev_chip_core_ranker为usable_in_agent_default=true且score_quantile>=0.80，可作为组合排序的正向工具证据，但若新闻/财报/同行/BookSkill存在硬反证，只能保持观察或降低研究暴露。"
        "硬反证包括明确负面新闻或监管/债务/停产事件、财报质量风险或负惊喜、同行显著弱且目标相对落后、过热高波动、筹码强上压或RAG相似失败；软缺口包括新闻空窗、财报近窗口无事件、BookSkill需grounding或普通来源不足。"
        "accepted量化工具遇到软缺口时优先写partially_adopted，而不是not_adopted_counter_evidence；只有硬反证足以覆盖工具时才写not_adopted_counter_evidence。"
        "若quant_tool_summaries中存在usable_in_agent_default=true且promotion_status包含accepted、pass、promot或default_combo_ranker_yellow的工具，只有在本卡片实际adopted或partially_adopted时，才在accepted_quant_tool_ids中列出tool_id。"
        "quant_tool_adoption_decision只能是adopted、partially_adopted、not_adopted_counter_evidence、not_applicable。若未采用accepted工具，accepted_quant_tool_ids必须为none，并在quant_tool_override_reasons写明覆盖它的当前反证通道，例如news_gap、financial_gap、peer_gap、bookskill_gap、chip_overhang、overheat或data_missing。"
        "组合模式必须在final_agent_reasoning_summary中写明给或不给研究暴露的主因，例如ranker高分位但财报缺失、peer弱、新闻空窗、BookSkill缺口、或多通道确认不足。"
        "若quant_tool_summary_only变体只暴露量化工具摘要，必须按工具自身promotion_status保守输出，不能脑补被消融的新闻、财报、Book Skill或同行信息。"
        "K线通道是量价辅助，不是最终决策者；需要同时看短周期冲击、中长期趋势、波动收缩/扩张、均值回归、震荡循环和同行/相关股票确认。"
        "筹码通道来自本地Tushare离线缓存，当前仅rev+chip_core通过no-harm默认排序检验；若筹码与新闻、BookSkill、同行或财报反证冲突，必须降低置信度。"
        "kline_return_20d<=-10.1231 只能作为20日回撤后的弱观察提示，必须结合新闻/财报/同行/Book Skill确认。"
        "kline_return_60d<=-16.9912 和低peer广度已在时间泛化中被反证，不得当作正向升级理由。"
        "真实行业/地域peer_context来自本地离线标准化缓存，只能作为同行相对背景、风险复核和单支观察候选；当前实验证明它没有通过组合默认策略或H2026最新块验收。"
        "若tushare行业/地域同行广度弱、目标相对同行落后、或同行新闻更热但目标沉默，必须作为反证或不确定性，不能把同行热度直接转嫁给目标股。"
        "财报披露日真实缺失是降权因素，但不是自动判信息不足；no_event_in_window只是近期无财报事件。若量价、新闻、Book Skill和反证足够，可以给出观察或继续深挖。"
        "若出现20日涨幅极高或RSI过热，同时新闻空窗、财报真实缺失、Book Skill未解析，应视为强反证，通常只能低权重观察或降级。"
        "数据源缺失、访问失败、新闻覆盖不足、Book Skill来源不完整时，必须在data_missing_flags或理由中如实说明。"
        "这是研究回测内部模拟，不接券商、不自动交易、不承诺收益。"
    )
    user = {
        "task": "根据时间安全证据包生成一张agent_decision_card JSON。",
        "required_output_fields": [
            "type",
            "agent_policy_version",
            "variant",
            "step",
            "train_blocks",
            "valid_block",
            "decision_date",
            "code",
            "name",
            "task_mode",
            "research_grade",
            "simulated_action",
            "simulated_weight_change",
            "user_operation_suggestion",
            "target_position",
            "position_plan",
            "buy_or_add_trigger",
            "reduce_or_sell_trigger",
            "review_condition",
            "python_signal_summary",
            "kline_signal_summary",
            "news_signal_summary",
            "book_skill_evidence",
            "memory_experience_used",
            "counter_evidence",
            "accepted_quant_tool_ids",
            "quant_tool_adoption_decision",
            "quant_tool_override_reasons",
            "final_agent_reasoning_summary",
            "confidence_level",
            "data_missing_flags",
            "error_reflection",
            "research_only",
            "not_investment_instruction",
        ],
        "book_skill_requirement": "必须审阅evidence_pack.book_skill_candidates；若采用、降权、忽略或缺失，需在book_skill_evidence和final_agent_reasoning_summary中说明。",
        "kline_requirement": "K线特征只可作为量价辅助；20日回撤是弱提示，60日深跌和低peer广度不得作为正向升级理由。",
        "peer_context_requirement": "peer_context_features若存在，表示本地离线行业/地域同行相对证据；只能辅助复核单支/组合相对排序，不得单独触发继续深挖或增加研究暴露。",
        "quant_tool_requirement": evidence_pack.get("quant_tool_requirement", ""),
        "accepted_quant_tool_requirement": "若存在accepted/default量化工具且本卡片实际采用或部分采用，必须显式列出tool_id。遇到新闻空窗、财报近窗口无事件、BookSkill需grounding这类软缺口时，应优先写quant_tool_adoption_decision=partially_adopted并保持低权重观察；只有明确负面新闻/财报风险/同行显著弱/过热高波动/筹码强上压/RAG失败等硬反证覆盖工具时，才写not_adopted_counter_evidence且accepted_quant_tool_ids=none。没有accepted工具时写accepted_quant_tool_ids=none且quant_tool_adoption_decision=not_applicable。",
        "sampler_context_requirement": "若evidence_pack.sampler_context不为none，必须说明该采样器是观察型假设还是可升级证据；不得引用未来收益数字。",
        "operation_plan_requirement": "若evidence_pack.operation_plan_context不为空，必须把它作为本地动作草案审计：支持则承接动作/仓位，不支持则明确写出覆盖原因。输出必须包含user_operation_suggestion、target_position、position_plan、buy_or_add_trigger、reduce_or_sell_trigger、review_condition。若operation_action=small_buy_hold，新闻空窗、财报无近窗事件、轻度同行弱、BookSkill观察类缺口只能降置信或降到小仓地板，不能单独归零；新闻语义问卷/分叉上下文应作为硬风险与软缺口区分材料，不得整体忽略。若本地草案给出0.20到0.35小仓且无硬反证，优先承接原仓位，不要机械压到更低仓位；只有明确负面新闻、财报质量风险、极端过热、筹码强上压、同行显著走弱且目标持续落后或RAG相似失败时，才大幅降仓或归零。",
        "case_memory_requirement": "若evidence_pack.retrieved_cases_context不为none，必须把相似失败/反证案例作为memory_experience_used或counter_evidence的一部分；不得引用未来收益字段。",
        "news_branch_case_requirement": evidence_pack.get("news_branch_case_requirement", ""),
        "nonprice_risk_overlay_requirement": evidence_pack.get("nonprice_risk_overlay_requirement", ""),
        "conflict_quality_requirement": "若evidence_pack.conflict_quality_context不为none，必须检查其是否声明walk_forward_prior_only；只能引用rule_status和agent_use，不得引用未来收益数字。",
        "promote_context_requirement": "若evidence_pack.promote_context不为none，必须检查其是否声明walk_forward_prior_only；只能作为正向升级假设，仍需当前新闻/财报/同行/BookSkill确认。",
        "task_mode_requirement": evidence_pack.get("task_mode_requirement", ""),
        "evidence_pack": evidence_pack,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, allow_nan=False)},
    ]


def card_from_evidence_pack(evidence_pack: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    card = {
        "type": "agent_decision_card",
        "agent_policy_version": evidence_pack["agent_policy_version"],
        "variant": evidence_pack.get("variant", "deepseek_agent"),
        "step": evidence_pack["step"],
        "train_blocks": evidence_pack["train_blocks"],
        "valid_block": evidence_pack["valid_block"],
        "decision_date": evidence_pack["decision_date"],
        "code": evidence_pack["code"],
        "name": evidence_pack["name"],
        "task_mode": evidence_pack["task_mode"],
        "sample_panel_id": evidence_pack.get("sample_panel_id"),
        "sample_rank_in_panel": evidence_pack.get("sample_rank_in_panel"),
        "case_memory_mode": evidence_pack.get("case_memory_mode", "memory_compact_only"),
        "research_grade": decision.get("research_grade"),
        "simulated_action": decision.get("simulated_action"),
        "simulated_weight_change": _coerce_weight(decision.get("simulated_weight_change"), decision.get("simulated_action")),
        "user_operation_suggestion": _operation_suggestion(decision.get("user_operation_suggestion"), decision.get("simulated_action")),
        "target_position": _coerce_target_position(decision.get("target_position"), decision.get("simulated_weight_change"), decision.get("simulated_action")),
        "position_plan": decision.get("position_plan") or "",
        "buy_or_add_trigger": decision.get("buy_or_add_trigger") or "",
        "reduce_or_sell_trigger": decision.get("reduce_or_sell_trigger") or "",
        "review_condition": decision.get("review_condition") or "",
        "python_signal_summary": decision.get("python_signal_summary") or evidence_pack["python_signal_summary"],
        "kline_signal_summary": decision.get("kline_signal_summary") or evidence_pack["kline_signal_summary"],
        "news_signal_summary": decision.get("news_signal_summary") or evidence_pack["news_signal_summary"],
        "book_skill_evidence": decision.get("book_skill_evidence") or _book_skill_text(evidence_pack) or "无Book Skill候选",
        "memory_experience_used": decision.get("memory_experience_used") or _memory_used_text(evidence_pack),
        "counter_evidence": decision.get("counter_evidence") or evidence_pack["counter_evidence"],
        "accepted_quant_tool_ids": "none",
        "quant_tool_adoption_decision": "not_applicable",
        "quant_tool_override_reasons": "none",
        "final_agent_reasoning_summary": decision.get("final_agent_reasoning_summary") or "",
        "confidence_level": _coerce_confidence(decision.get("confidence_level")),
        "data_missing_flags": decision.get("data_missing_flags") or evidence_pack.get("data_missing_flags", ""),
        "error_reflection": decision.get("error_reflection") or "待后验Ground Truth成熟后生成反思。",
        "research_only": True,
        "not_investment_instruction": True,
    }
    apply_decision_guardrails(card, evidence_pack)
    _sync_operation_fields(card, evidence_pack)
    _attach_quant_tool_adoption_fields(card, evidence_pack, decision)
    sanitize_decision_card_text_fields(card)
    return _json_clean(card)


def _attach_quant_tool_adoption_fields(card: dict[str, Any], evidence_pack: dict[str, Any], decision: dict[str, Any]) -> None:
    accepted_ids = _accepted_quant_tool_ids(evidence_pack)
    explicit_adoption = _text(decision.get("quant_tool_adoption_decision"))
    if not accepted_ids:
        card["quant_tool_adoption_decision"] = "not_applicable"
    elif explicit_adoption in {"adopted", "partially_adopted", "not_adopted_counter_evidence"}:
        card["quant_tool_adoption_decision"] = explicit_adoption
    else:
        card["quant_tool_adoption_decision"] = _infer_quant_tool_adoption(card, accepted_ids)

    if card["quant_tool_adoption_decision"] in {"adopted", "partially_adopted"}:
        card["accepted_quant_tool_ids"] = ";".join(accepted_ids)
    else:
        card["accepted_quant_tool_ids"] = "none"
    card["quant_tool_override_reasons"] = _infer_quant_tool_override_reasons(card, evidence_pack, accepted_ids)


def _accepted_quant_tool_ids(evidence_pack: dict[str, Any]) -> list[str]:
    rows = evidence_pack.get("quant_tool_summaries")
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("usable_in_agent_default") is not True:
            continue
        status = _text(row.get("promotion_status")).lower()
        if any(term in status for term in ["accepted", "accept", "pass", "promot", "default_combo_ranker_yellow"]):
            tool_id = _text(row.get("tool_id"))
            if tool_id and tool_id not in ids:
                ids.append(tool_id)
    return ids


def _infer_quant_tool_adoption(card: dict[str, Any], accepted_ids: list[str]) -> str:
    if not accepted_ids:
        return "not_applicable"
    action = _text(card.get("simulated_action"))
    grade = _text(card.get("research_grade"))
    weight = _safe(card.get("simulated_weight_change"))
    if action == "增加研究暴露" and not math.isnan(weight) and weight >= 0.5 and grade == "继续深挖":
        return "adopted"
    if action in {"保持观察", "增加研究暴露"} and not math.isnan(weight) and weight > 0:
        return "partially_adopted"
    return "not_adopted_counter_evidence"


def _infer_quant_tool_override_reasons(card: dict[str, Any], evidence_pack: dict[str, Any], accepted_ids: list[str]) -> str:
    if not accepted_ids:
        return "none"
    adoption = _text(card.get("quant_tool_adoption_decision"))
    if adoption in {"adopted", "not_applicable"}:
        return "none"
    reasons = _current_counter_evidence_reasons(card, evidence_pack)
    return ";".join(reasons) if reasons else "accepted_tool_not_adopted_without_structured_reason"


def _current_counter_evidence_reasons(card: dict[str, Any], evidence_pack: dict[str, Any]) -> list[str]:
    haystack = " ".join(
        _text(value)
        for value in [
            card.get("counter_evidence"),
            card.get("data_missing_flags"),
            card.get("final_agent_reasoning_summary"),
            evidence_pack.get("counter_evidence"),
            evidence_pack.get("data_missing_flags"),
            evidence_pack.get("news_signal_summary"),
            evidence_pack.get("financial_report_signal_summary"),
            evidence_pack.get("peer_context_signal_summary"),
            evidence_pack.get("chip_signal_summary"),
            evidence_pack.get("book_skill_requirement"),
        ]
    ).lower()
    checks = [
        ("news_gap", ["新闻", "news", "coverage", "空窗", "缺失率"]),
        ("financial_gap", ["财报", "financial", "披露日", "no_event", "publish_date"]),
        ("peer_gap", ["同行", "peer", "行业", "落后", "广度弱"]),
        ("bookskill_gap", ["book", "skill", "bookskill", "策略", "弱适配", "缺口"]),
        ("chip_overhang", ["筹码", "chip", "upper_overhang", "套牢", "上档"]),
        ("overheat_or_volatility", ["过热", "rsi", "波动", "volatility", "高位"]),
        ("data_missing", ["missing", "缺失", "unavailable", "信息不足"]),
        ("memory_or_rag_counter", ["memory", "rag", "历史", "反证", "失败"]),
    ]
    reasons = []
    for reason, needles in checks:
        if any(needle in haystack for needle in needles):
            reasons.append(reason)
    return reasons


def _field_subset(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: _json_value(data.get(field)) for field in fields if field in data}


def _quant_tool_summaries(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("quant_tool_summaries")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append(sanitize_quant_tool_outcome(item))
    return rows[:8]


def _quant_tool_summary(data: dict[str, Any]) -> str:
    existing = _text(data.get("quant_tool_signal_summary"))
    if existing:
        return existing
    return quant_tool_summary_text(_quant_tool_summaries(data))


def _analogue_case_context(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("analogue_case_context")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    safe_fields = [
        "tool_id",
        "tool_version",
        "date",
        "code",
        "time_block",
        "task_mode",
        "policy_profile",
        "policy_status",
        "decision_frequency",
        "feature_group",
        "selection_mode",
        "score",
        "confidence",
        "risk_tier",
        "primary_risk_branch",
        "risk_branch_labels",
        "branch_policy",
        "required_confirmation",
        "known_false_veto_risk",
        "calibration_policy",
        "action_hint",
        "usable_in_agent_default",
        "missing_flags",
        "counter_evidence",
        "source_ref_ids",
        "source_variant",
        "base_branch",
        "analog_id",
        "gate_id",
        "position_cap_hint",
        "transfer_score",
        "transfer_threshold",
        "analog_neighbor_count",
        "analog_pos_rate",
        "analog_avg_return",
        "analog_historical_tail_risk_rate",
        "analog_top_case_refs",
        "channel_support_count",
        "channel_hard_counter_count",
        "news_low_warning",
        "financial_no_recent_event",
        "chip_support_visible",
        "agent_instruction",
        "promotion_status",
        "agent_use",
        "forbidden_use",
        "research_only",
        "not_investment_instruction",
    ]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append({field: _json_value(item.get(field)) for field in safe_fields if field in item})
    return rows[:6]


def _nonprice_risk_overlay_context(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("nonprice_risk_overlay_context")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    safe_fields = [
        "tool_id",
        "tool_version",
        "task_mode",
        "policy_profile",
        "policy_status",
        "feature_group",
        "selection_mode",
        "risk_tier",
        "primary_risk_branch",
        "risk_branch_labels",
        "branch_policy",
        "promotion_status",
        "usable_in_agent_default",
        "top_features",
        "description",
        "required_confirmation",
        "known_false_veto_risk",
        "calibration_policy",
        "action_hint",
        "counter_evidence",
        "missing_flags",
        "source_ref_ids",
        "train_valid_test_blocks",
        "agent_use",
        "forbidden_use",
        "flag_active_on_current_row",
        "row_scope",
        "research_only",
        "not_investment_instruction",
    ]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append({field: _json_value(item.get(field)) for field in safe_fields if field in item})
    return rows[:6]


def _operation_plan_context(data: dict[str, Any]) -> dict[str, Any]:
    action = _text(data.get("operation_action") or data.get("operation_action_cn") or data.get("user_operation_suggestion"))
    user_action, default_target = _operation_action_defaults(action)
    target = _safe(data.get("local_target_position") or data.get("target_position"))
    if math.isnan(target) and default_target is not None:
        target = default_target
    reason = _text(data.get("local_reason_code") or data.get("operation_reason_code"))
    frequency = _text(data.get("decision_frequency"))
    period = _text(data.get("period") or data.get("valid_block") or data.get("time_block"))
    if not action and math.isnan(target) and not reason:
        return {}
    return {
        "tool_id": "local_user_operation_plan_context_v1",
        "operation_action": action,
        "user_operation_suggestion": user_action,
        "target_position": None if math.isnan(target) else round(float(target), 4),
        "reason_code": reason or action,
        "decision_frequency": frequency,
        "period": period,
        "agent_instruction": (
            "审计本地动作草案；证据支持则承接动作/仓位，证据不支持则明确覆盖原因，"
            "并输出替代的买入/加仓/持有/减仓/卖出/等待/补数据建议。"
            "若operation_action=small_buy_hold，含义是小仓试探/继续持有，不是模糊观察。"
        ),
        "forbidden_use": "not_future_label_not_mandatory_copy",
        "research_only": True,
    }


def _operation_action_defaults(action: str) -> tuple[str, float | None]:
    if "small_buy_hold" in action or "小仓" in action:
        return "试探买入/持有", 0.25
    if "buy_add" in action or "买入" in action or "加仓" in action:
        return "试探买入/加仓", 0.5
    if "reduce_review" in action or "复核" in action:
        return "减仓/卖出复核", 0.15
    if "reduce_sell" in action or "卖出" in action or "回避" in action:
        return "卖出/不买", 0.0
    if action == "wait" or "等待" in action or "不操作" in action:
        return "等待不买", 0.0
    return action, None


def _python_summary(data: dict[str, Any], python_candidate: str) -> str:
    parts = []
    if python_candidate:
        parts.append(f"candidate={python_candidate}")
    for field in ["timeline_score", "relative_strength_rank", "counter_score", "close_above_ma200"]:
        if field in data:
            parts.append(f"{field}={_fmt(data.get(field))}")
    return "; ".join(parts) if parts else "no python signal"


def _sampler_context(python_candidate: str) -> str:
    text = str(python_candidate or "")
    if "kline_reversal_friction_confirmed" in text:
        return (
            "observe_only_sampler=kline_reversal_friction_confirmed; "
            "meaning=该候选故意采样K线深跌/波动摩擦，但要求至少两个正向确认、筹码下方支撑且上方压力不过高；"
            "agent_use=不要把K线风险机械否决，应判断它是可接受反转摩擦还是基本面/新闻/同行反证；"
            "boundary=若新闻/财报/同行/BookSkill仍存在硬缺口，只能保持观察或低权重，不能直接升级。"
        )
    if "financial_event_quality_pc2" in text:
        return (
            "observe_only_sampler=financial_event_quality_pc2; "
            "meaning=该候选采样已匹配财报/业绩事件、财务质量风险较低、惊喜非负且有至少两个正向确认的样本；"
            "agent_use=财报事件可作为高可信复核材料，但样本仍偏薄且可能集中；"
            "boundary=不得把单个财报事件单独当作正向alpha，仍需新闻、同行、BookSkill和量价共同确认。"
        )
    return "none"


def _kline_summary(data: dict[str, Any]) -> str:
    k3 = _safe(data.get("kline_return_3d"))
    k10 = _safe(data.get("kline_return_10d"))
    k20 = _safe(data.get("kline_return_20d"))
    k60 = _safe(data.get("kline_return_60d"))
    k120 = _safe(data.get("kline_return_120d"))
    k240 = _safe(data.get("kline_return_240d"))
    atr = _safe(data.get("kline_atr20_pct"))
    cycle = _safe(data.get("kline_volatility_ratio_20_60"))
    efficiency = _safe(data.get("kline_efficiency_ratio_20d"))
    reversal = _safe(data.get("kline_direction_reversal_rate_20d"))
    streak = _safe(data.get("kline_signed_streak_norm_20d"))
    peer_breadth = _safe(data.get("peer_kline_group_positive_breadth_20d"))
    if all(math.isnan(value) for value in [k3, k10, k20, k60, k120, k240, atr, cycle, efficiency, reversal, streak, peer_breadth]):
        return "kline_channel_not_collected"
    flags = []
    if not math.isnan(k20) and k20 <= -10.1231:
        flags.append("20d_pullback_observe")
    if not math.isnan(k60) and k60 <= -16.9912:
        flags.append("60d_deep_drawdown_not_positive")
    if not math.isnan(efficiency) and not math.isnan(reversal) and efficiency <= 0.25 and reversal >= 0.45:
        flags.append("choppy_cycle_recheck")
    return (
        f"return3/10/20/60/120/240={_fmt(k3)}/{_fmt(k10)}/{_fmt(k20)}/{_fmt(k60)}/{_fmt(k120)}/{_fmt(k240)}; "
        f"atr20_pct={_fmt(atr)}; vol_ratio20_60={_fmt(cycle)}; "
        f"eff20={_fmt(efficiency)}; rev20={_fmt(reversal)}; streak20={_fmt(streak)}; "
        f"peer_breadth20={_fmt(peer_breadth)}; "
        f"flags={','.join(flags) if flags else 'none'}"
    )


def _peer_context_summary(data: dict[str, Any]) -> str:
    industry = _text(data.get("tushare_industry"))
    area = _text(data.get("tushare_area"))
    ind_size = _safe(data.get("tushare_industry_group_size"))
    ind_ret = _safe(data.get("tushare_industry_avg_return_20d"))
    ind_rel = _safe(data.get("tushare_industry_relative_return_20d"))
    ind_breadth = _safe(data.get("tushare_industry_positive_breadth_20d"))
    ind_ma = _safe(data.get("tushare_industry_above_ma200_rate"))
    ind_gap = _safe(data.get("tushare_industry_news_attention_gap"))
    area_size = _safe(data.get("tushare_area_group_size"))
    area_ret = _safe(data.get("tushare_area_avg_return_20d"))
    area_rel = _safe(data.get("tushare_area_relative_return_20d"))
    area_breadth = _safe(data.get("tushare_area_positive_breadth_20d"))
    area_gap = _safe(data.get("tushare_area_news_attention_gap"))
    numeric_values = [ind_size, ind_ret, ind_rel, ind_breadth, ind_ma, ind_gap, area_size, area_ret, area_rel, area_breadth, area_gap]
    if not industry and not area and all(math.isnan(value) for value in numeric_values):
        return "peer_context_not_collected"
    flags = []
    if not math.isnan(ind_breadth) and ind_breadth <= 0.4:
        flags.append("industry_breadth_weak")
    if not math.isnan(ind_rel) and ind_rel < 0:
        flags.append("lagging_industry_peers")
    if not math.isnan(ind_gap) and ind_gap < -0.3:
        flags.append("peer_attention_hotter_than_self")
    return (
        f"industry={industry or 'unknown'} size={_fmt(ind_size)} avg20={_fmt(ind_ret)} rel20={_fmt(ind_rel)} "
        f"breadth20={_fmt(ind_breadth)} above_ma200={_fmt(ind_ma)} attention_gap={_fmt(ind_gap)}; "
        f"area={area or 'unknown'} size={_fmt(area_size)} avg20={_fmt(area_ret)} rel20={_fmt(area_rel)} "
        f"breadth20={_fmt(area_breadth)} attention_gap={_fmt(area_gap)}; "
        f"flags={','.join(flags) if flags else 'none'}"
    )


def _chip_summary(data: dict[str, Any]) -> str:
    lower = _safe(data.get("lower_support"))
    concentration = _safe(data.get("chip_concentration"))
    band = _safe(data.get("cost_band_width"))
    overhang = _safe(data.get("upper_overhang"))
    winner = _safe(data.get("winner_rate_pct"))
    source = _text(data.get("chip_core_source_name"))
    if all(math.isnan(value) for value in [lower, concentration, band, overhang, winner]) and not source:
        return "chip_channel_not_collected"
    flags = []
    if not math.isnan(lower) and lower >= 0.25:
        flags.append("lower_support_high")
    if not math.isnan(overhang) and overhang >= 0.35:
        flags.append("upper_overhang_high")
    if not math.isnan(band) and band >= 0.60:
        flags.append("cost_band_wide")
    return (
        f"lower_support={_fmt(lower)}; concentration={_fmt(concentration)}; "
        f"band_width={_fmt(band)}; upper_overhang={_fmt(overhang)}; "
        f"winner_rate={_fmt(winner)}; source={source or 'unknown'}; "
        f"flags={','.join(flags) if flags else 'none'}"
    )


def _news_summary(data: dict[str, Any]) -> str:
    warning = max(
        _safe(data.get("news_warning_score")),
        _safe(data.get("ds_news_risk_score")),
        _safe(data.get("news_warning_score_30d")),
        _safe(data.get("news_risk_event_score_30d")),
        0.0,
    )
    opportunity = max(
        _safe(data.get("news_opportunity_score")),
        _safe(data.get("ds_news_opportunity_score")),
        _safe(data.get("news_opportunity_alert_score_30d")),
        _safe(data.get("news_opportunity_event_score_30d")),
        0.0,
    )
    count = _safe(data.get("news_count_30d"))
    missing = _safe(data.get("news_missing_rate"))
    semantic_net = _safe(data.get("ds_news_net_score"))
    if math.isnan(count) and math.isnan(warning) and math.isnan(opportunity):
        return "news_missing_or_not_collected"
    return (
        f"count={_fmt(count)}; warning={_fmt(warning)}; opportunity={_fmt(opportunity)}; "
        f"semantic_net={_fmt(semantic_net)}; missing_rate={_fmt(missing)}"
    )


def _financial_report_summary(data: dict[str, Any]) -> str:
    count = _safe(data.get("financial_report_event_count"))
    materiality = _safe(data.get("financial_report_materiality_score"))
    quality_risk = _safe(data.get("financial_quality_risk_score"))
    surprise = _safe(data.get("financial_surprise_score"))
    disclosure_quality = _safe(data.get("financial_disclosure_quality_score"))
    missing = _safe(data.get("financial_report_missing_rate"))
    status = _text(data.get("financial_report_join_status"))
    if all(math.isnan(value) for value in [count, materiality, quality_risk, surprise, disclosure_quality, missing]) and not status:
        return "financial_report_channel_not_collected"
    if status == "no_event_in_window":
        return (
            "no_recent_financial_report_event; "
            "financial_positive_confirmation=false; "
            "not_disclosure_missing=true; "
            "status=no_event_in_window"
        )
    return (
        f"events={_fmt(count)}; materiality={_fmt(materiality)}; quality_risk={_fmt(quality_risk)}; "
        f"surprise={_fmt(surprise)}; disclosure_quality={_fmt(disclosure_quality)}; "
        f"missing_rate={_fmt(missing)}; status={status or 'unknown'}"
    )


def _book_skill_candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    return resolve_book_skill_candidates(data.get("triggered_skills"))


def _book_skill_text(evidence_pack: dict[str, Any]) -> str:
    candidates = evidence_pack.get("book_skill_candidates") or []
    parts = []
    for item in candidates:
        strategy_id = str(item.get("strategy_id", "")).strip()
        if not strategy_id:
            continue
        source = str(item.get("source_book") or item.get("source_status") or "").strip()
        status = str(item.get("validation_status") or item.get("confidence") or "").strip()
        text = strategy_id
        if source:
            text += f"({source})"
        if status:
            text += f"[{status}]"
        parts.append(text)
    return ";".join(parts)


def _memory_used_text(evidence_pack: dict[str, Any]) -> str:
    memory = str(evidence_pack.get("memory_context") or "none")
    retrieved = str(evidence_pack.get("retrieved_cases_context") or "none")
    if retrieved and retrieved != "none":
        return f"{memory}\n\n{retrieved}" if memory and memory != "none" else retrieved
    return memory


def _task_mode_requirement(task_mode: str) -> str:
    canonical = TASK_MODE_ALIASES.get(str(task_mode), str(task_mode))
    if canonical == "single_stock":
        return "单支模式：判断单只股票的研究分级、盯盘优先级和模拟研究暴露变化，重点说明风险复核、信息缺口和后验反思。正式工作流名为 single_stock_watch。"
    if canonical == "portfolio_pool":
        return "组合模式：在候选池内做相对排序和研究注意力分配，重点比较同池机会、现金防守、TopN适配和基线差异。正式工作流名为 portfolio_pool_optimize。"
    return "必须先识别任务模式；若任务模式未知，优先标记信息不足并说明缺失原因。"


def _counter_evidence(data: dict[str, Any]) -> str:
    flags = []
    if _safe(data.get("news_warning_score_30d")) >= 1:
        flags.append("新闻预警")
    if _safe(data.get("atr20_pct")) >= 4:
        flags.append("波动偏高")
    if _safe(data.get("counter_score")) <= 5:
        flags.append("反证分偏低")
    if "financial_publish_date_missing" in _data_missing_flags(data):
        flags.append("财报披露日缺失")
    if _safe(data.get("news_missing_rate")) >= 0.8:
        flags.append("新闻覆盖不足")
    if _safe(data.get("financial_quality_risk_score")) >= 0.6:
        flags.append("财报质量风险")
    if _safe(data.get("financial_disclosure_quality_score")) <= 0.3 and _safe(data.get("financial_report_event_count")) > 0:
        flags.append("财报披露质量弱")
    if _financial_report_true_missing(data) and _safe(data.get("financial_report_missing_rate")) >= 0.8:
        flags.append("财报事件覆盖不足")
    industry_breadth = _safe(data.get("tushare_industry_positive_breadth_20d"))
    industry_relative = _safe(data.get("tushare_industry_relative_return_20d"))
    attention_gap = _safe(data.get("tushare_industry_news_attention_gap"))
    if not math.isnan(industry_breadth) and industry_breadth <= 0.4 and not math.isnan(industry_relative) and industry_relative < 0:
        flags.append("真实行业peer弱且目标落后")
    if not math.isnan(attention_gap) and attention_gap <= -0.3:
        flags.append("同行关注高于目标自身")
    prior = _safe(data.get("prior_return_20d"))
    rsi = _safe(data.get("rsi14"))
    news_count = _safe(data.get("news_count_30d"))
    if ((not math.isnan(prior) and prior >= 80) or (not math.isnan(rsi) and rsi >= 85)) and (math.isnan(news_count) or news_count <= 0):
        flags.append("过热且新闻空窗")
    return ";".join(flags) if flags else "无强反证"


def apply_decision_guardrails(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    _apply_financial_report_guardrail(card, evidence_pack)
    _apply_financial_report_only_no_upgrade_guardrail(card, evidence_pack)
    _apply_news_positive_cap_guardrail(card, evidence_pack)
    _apply_portfolio_cross_channel_confirmation_guardrail(card, evidence_pack)
    _apply_rag_applicable_failure_no_upgrade_guardrail(card, evidence_pack)
    _apply_single_stock_risk_review_queue_no_raise_guardrail(card, evidence_pack)
    _apply_small_entry_softgap_floor_guardrail(card, evidence_pack)
    _apply_action_label_buy_add_softgap_floor_guardrail(card, evidence_pack)
    _apply_action_label_wait_reduce_cap_guardrail(card, evidence_pack)


def _apply_small_entry_softgap_floor_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    context = evidence_pack.get("operation_plan_context")
    if not isinstance(context, dict):
        return
    if _text(context.get("operation_action")) != "small_buy_hold":
        return
    if _small_entry_has_hard_counter(card, evidence_pack):
        return

    local_target = _safe(context.get("target_position"))
    default_floor = _safe(context.get("default_position_floor_if_no_hard_counter"))
    if math.isnan(local_target):
        local_target = default_floor
    if math.isnan(local_target):
        local_target = 0.1
    floor = max(0.1, min(0.35, float(local_target)))
    current = _safe(card.get("target_position"))
    current = 0.0 if math.isnan(current) else current
    if current >= floor:
        return

    card["research_grade"] = "放入观察" if card.get("research_grade") not in {"继续深挖", "放入观察"} else card.get("research_grade")
    card["simulated_action"] = "保持观察"
    card["simulated_weight_change"] = max(_safe(card.get("simulated_weight_change")) if not math.isnan(_safe(card.get("simulated_weight_change"))) else 0.0, min(floor, 0.2))
    card["user_operation_suggestion"] = "试探买入/持有"
    card["target_position"] = floor
    reason = "small_buy_hold软缺口地板：未见硬反证，peer/财报无事件/BookSkill观察类缺口不得把小仓归零。"
    existing = _text(card.get("counter_evidence"))
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = _text(card.get("error_reflection"))
    note = "guardrail_applied: small_entry_softgap_floor_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "小仓分支无硬反证，执行层保留试探仓位并列明复核阈值。"


def _apply_action_label_buy_add_softgap_floor_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    context = evidence_pack.get("operation_plan_context")
    if not isinstance(context, dict):
        return
    if _text(context.get("operation_action")) != "buy_add":
        return
    if _text(context.get("reason_code")) != "p0_action_label_scorer_v1":
        return
    if _action_label_buy_add_has_hard_counter(card, evidence_pack):
        return

    default_floor = _safe(context.get("default_position_floor_if_no_hard_counter"))
    local_target = _safe(context.get("target_position"))
    if math.isnan(default_floor):
        default_floor = 0.10
    if math.isnan(local_target):
        local_target = default_floor
    floor = max(0.10, min(float(default_floor), float(local_target), 0.25))
    current = _safe(card.get("target_position"))
    current = 0.0 if math.isnan(current) else current
    if current >= floor:
        return

    card["research_grade"] = "放入观察" if card.get("research_grade") not in {"继续深挖", "放入观察"} else card.get("research_grade")
    card["simulated_action"] = "保持观察"
    current_weight = _safe(card.get("simulated_weight_change"))
    card["simulated_weight_change"] = max(0.0 if math.isnan(current_weight) else current_weight, min(floor, 0.2))
    card["user_operation_suggestion"] = "试探买入/持有复核"
    card["target_position"] = floor
    reason = "action_label买入候选软缺口地板：HGB动作标签工具给出buy/add草案且未见硬反证，新闻空窗/财报无近窗事件/BookSkill观察类缺口不得单独归零。"
    existing = _text(card.get("counter_evidence"))
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = _text(card.get("error_reflection"))
    note = "guardrail_applied: action_label_buy_add_softgap_floor_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "动作标签工具给出buy/add草案且无硬反证，执行层降级保留低仓位试探/持有复核。"


def _apply_action_label_wait_reduce_cap_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    context = evidence_pack.get("operation_plan_context")
    if not isinstance(context, dict):
        return
    if _text(context.get("reason_code")) != "p0_action_label_scorer_v1":
        return
    action = _text(context.get("operation_action"))
    if action not in {"wait", "reduce_review"}:
        return

    if action == "wait":
        target_cap = 0.0
        weight_cap = 0.0
        suggestion = "等待不买"
        reason = "action_label等待分支护栏：本地动作标签为wait_for_better_evidence，回测内部权重和用户目标仓位必须同步归零，避免把等待样本误算成买入暴露。"
    else:
        target_cap = 0.10
        weight_cap = 0.05
        suggestion = "减仓/卖出复核"
        reason = "action_label减仓复核护栏：本地动作标签为reduce_review，除非后续证据重新触发buy/add工具，否则只能保留极低复核仓位。"

    current_target = _safe(card.get("target_position"))
    current_target = 0.0 if math.isnan(current_target) else current_target
    current_weight = _safe(card.get("simulated_weight_change"))
    current_weight = 0.0 if math.isnan(current_weight) else current_weight
    if current_target <= target_cap and current_weight <= weight_cap:
        return

    card["target_position"] = min(current_target, target_cap)
    card["simulated_weight_change"] = min(current_weight, weight_cap)
    card["user_operation_suggestion"] = suggestion
    if action == "wait":
        card["simulated_action"] = "转入现金"
        card["research_grade"] = "暂时剔除" if card.get("research_grade") not in {"信息不足", "暂时剔除"} else card.get("research_grade")
    else:
        card["simulated_action"] = "降低研究暴露"
        card["research_grade"] = "放入观察" if card.get("research_grade") not in {"信息不足", "放入观察", "暂时剔除"} else card.get("research_grade")
    existing = _text(card.get("counter_evidence"))
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = _text(card.get("error_reflection"))
    note = "guardrail_applied: action_label_wait_reduce_cap_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note


def _action_label_buy_add_has_hard_counter(card: dict[str, Any], evidence_pack: dict[str, Any]) -> bool:
    text = ";".join(
        _text(item)
        for item in [
            evidence_pack.get("counter_evidence"),
            card.get("counter_evidence"),
            card.get("final_agent_reasoning_summary"),
            evidence_pack.get("nonprice_risk_overlay_context"),
        ]
    )
    hard_terms = [
        "明确负面",
        "监管",
        "债务",
        "停产",
        "财报质量风险",
        "负惊喜",
        "非标",
        "问询",
        "修正",
        "极端过热",
        "筹码强上压",
        "历史相似失败",
        "过热且新闻空窗",
        "财报风险护栏触发",
    ]
    if any(term in text for term in hard_terms):
        return True

    news = evidence_pack.get("news_features")
    news = news if isinstance(news, dict) else {}
    news_warning = _safe(news.get("news_warning_score"))
    if not math.isnan(news_warning) and news_warning >= 0.70:
        return True

    questionnaire = evidence_pack.get("news_semantic_questionnaire")
    questionnaire = questionnaire if isinstance(questionnaire, dict) else {}
    ds_risk = _safe(questionnaire.get("ds_news_risk_score"))
    if not math.isnan(ds_risk) and ds_risk >= 0.70:
        return True

    financial = evidence_pack.get("financial_report_features")
    financial = financial if isinstance(financial, dict) else {}
    quality_risk = _safe(financial.get("financial_quality_risk_score"))
    surprise = _safe(financial.get("financial_surprise_score"))
    if (not math.isnan(quality_risk) and quality_risk >= 0.60) or (not math.isnan(surprise) and surprise <= -0.60):
        return True

    chip = evidence_pack.get("chip_features")
    chip = chip if isinstance(chip, dict) else {}
    lower_support = _safe(chip.get("lower_support"))
    upper_overhang = _safe(chip.get("upper_overhang"))
    if (not math.isnan(upper_overhang) and upper_overhang >= 0.65) and (math.isnan(lower_support) or lower_support < 0.10):
        return True

    return False


def _small_entry_has_hard_counter(card: dict[str, Any], evidence_pack: dict[str, Any]) -> bool:
    text = ";".join(
        _text(item)
        for item in [
            evidence_pack.get("counter_evidence"),
            card.get("counter_evidence"),
            card.get("final_agent_reasoning_summary"),
            evidence_pack.get("nonprice_risk_overlay_context"),
        ]
    )
    base_hard_terms = [
        "明确负面",
        "监管",
        "债务",
        "停产",
        "财报质量风险",
        "负惊喜",
        "非标",
        "问询",
        "修正",
        "极端过热",
        "筹码强上压",
        "历史相似失败",
        "过热且新闻空窗",
        "RAG",
        "财报风险护栏触发",
    ]
    if any(term in text for term in base_hard_terms):
        return True
    peer_hard_terms = ["真实行业peer弱且目标落后", "同行显著走弱"]
    if any(term in text for term in peer_hard_terms) and not _small_entry_peer_softgap_supported(evidence_pack):
        return True
    return False


def _small_entry_peer_softgap_supported(evidence_pack: dict[str, Any]) -> bool:
    context = evidence_pack.get("operation_plan_context")
    context = context if isinstance(context, dict) else {}
    reason_code = _text(context.get("reason_code"))
    if reason_code in {"pps_m003_tuesday", "news_financial_clean"}:
        return _small_entry_news_financial_clean(evidence_pack)
    if reason_code in {"peer_weak_clean_chip", "news_financial_clean_chip_pullback", "news_financial_clean_chip"}:
        return _small_entry_clean_reversal_support(evidence_pack)
    return _small_entry_clean_reversal_support(evidence_pack)


def _small_entry_clean_reversal_support(evidence_pack: dict[str, Any]) -> bool:
    return _small_entry_chip_support_ok(evidence_pack) and _small_entry_news_financial_clean(evidence_pack)


def _small_entry_chip_support_ok(evidence_pack: dict[str, Any]) -> bool:
    chip = evidence_pack.get("chip_features")
    chip = chip if isinstance(chip, dict) else {}
    lower_support = _safe(chip.get("lower_support"))
    upper_overhang = _safe(chip.get("upper_overhang"))
    return (not math.isnan(lower_support) and lower_support >= 0.1) and (math.isnan(upper_overhang) or upper_overhang <= 0.4)


def _small_entry_news_financial_clean(evidence_pack: dict[str, Any]) -> bool:
    news = evidence_pack.get("news_features")
    news = news if isinstance(news, dict) else {}
    news_warning = _safe(news.get("news_warning_score"))
    news_ok = math.isnan(news_warning) or news_warning < 0.7

    questionnaire = evidence_pack.get("news_semantic_questionnaire")
    questionnaire = questionnaire if isinstance(questionnaire, dict) else {}
    ds_risk = _safe(questionnaire.get("ds_news_risk_score"))
    questionnaire_ok = math.isnan(ds_risk) or ds_risk < 0.7

    financial = evidence_pack.get("financial_report_features")
    financial = financial if isinstance(financial, dict) else {}
    quality_risk = _safe(financial.get("financial_quality_risk_score"))
    surprise = _safe(financial.get("financial_surprise_score"))
    financial_ok = (math.isnan(quality_risk) or quality_risk < 0.6) and (math.isnan(surprise) or surprise > -0.6)
    return bool(news_ok and questionnaire_ok and financial_ok)


def _apply_news_positive_cap_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    questionnaire = evidence_pack.get("news_semantic_questionnaire")
    if not isinstance(questionnaire, dict):
        return
    if questionnaire.get("ds_news_positive_capped_by_rule") is not True:
        return
    if not _has_unresolved_confirmation_gap(evidence_pack):
        return
    changed = False
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
        changed = True
    reason = "新闻机会分已被弱主线规则封顶，且Book Skill/披露日等确认缺口未解决，禁止增加研究暴露。"
    reflection = str(card.get("error_reflection") or "").strip()
    guardrail_note = "guardrail_applied: capped_news_positive_with_unresolved_confirmation_gap"
    if card.get("simulated_action") == "增加研究暴露":
        card["simulated_action"] = "保持观察"
        card["simulated_weight_change"] = 0.1
        existing = str(card.get("counter_evidence") or "").strip()
        card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
        if guardrail_note not in reflection:
            card["error_reflection"] = f"{reflection}; {guardrail_note}" if reflection else guardrail_note
        changed = True
    if changed or guardrail_note in reflection:
        card["final_agent_reasoning_summary"] = "新闻机会被封顶且确认缺口未解决，执行层改为放入观察并保持低权重。"


def _apply_financial_report_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    guardrail = evidence_pack.get("financial_report_guardrail")
    if not isinstance(guardrail, dict):
        return
    if guardrail.get("rule_id") != "financial_risk_to_zero_guard_v1":
        return
    if guardrail.get("triggered_for_pack") is not True:
        return
    if card.get("simulated_action") not in {"增加研究暴露", "保持观察", "降低研究暴露"} and card.get("research_grade") != "继续深挖":
        return
    reason = "财报高风险或负惊喜过热且缺少交叉确认，执行层触发财报风险归零护栏。"
    existing = str(card.get("counter_evidence") or "").strip()
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
    card["simulated_action"] = "信息不足不动作"
    card["simulated_weight_change"] = 0.0
    reflection = str(card.get("error_reflection") or "").strip()
    note = "guardrail_applied: financial_risk_to_zero_guard_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "财报风险护栏触发，缺少交叉确认，执行层压到0权重研究状态。"


def _apply_financial_report_only_no_upgrade_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    if evidence_pack.get("variant") != "financial_report_only":
        return
    if card.get("research_grade") != "继续深挖" and card.get("simulated_action") != "增加研究暴露":
        return
    reason = "财报单通道缺少普通新闻、同行、Python和Book Skill共同确认，不允许单独升级研究分级。"
    existing = str(card.get("counter_evidence") or "").strip()
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
    if card.get("simulated_action") == "增加研究暴露":
        card["simulated_action"] = "保持观察"
        weight = _safe(card.get("simulated_weight_change"))
        card["simulated_weight_change"] = min(weight, 0.1) if not math.isnan(weight) else 0.1
    reflection = str(card.get("error_reflection") or "").strip()
    note = "guardrail_applied: financial_report_only_no_upgrade_without_confirmation_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "财报单通道只能作为复核/反证，缺少交叉确认时执行层压回放入观察。"


def _apply_portfolio_cross_channel_confirmation_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    if str(evidence_pack.get("task_mode")) not in {"portfolio_pool", "portfolio_pool_optimize"}:
        return
    if not _is_default_or_full_agent_variant(evidence_pack.get("variant")):
        return
    if card.get("research_grade") != "继续深挖" and card.get("simulated_action") != "增加研究暴露":
        return
    gap_reasons = _cross_channel_gap_reasons(evidence_pack)
    if len(gap_reasons) < 3:
        return
    if _has_strong_default_quant_support(evidence_pack) and _only_soft_confirmation_gaps(gap_reasons):
        return
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
    if card.get("simulated_action") == "增加研究暴露":
        card["simulated_action"] = "保持观察"
        card["simulated_weight_change"] = 0.1
    reason = "组合模式缺少新闻/财报/同行/Book Skill多重确认，禁止仅靠Python强信号主动暴露。"
    existing = str(card.get("counter_evidence") or "").strip()
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = str(card.get("error_reflection") or "").strip()
    note = "guardrail_applied: portfolio_cross_channel_confirmation_gap_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "多通道确认缺口过多，执行层压回放入观察并保持低权重。"


def _apply_rag_applicable_failure_no_upgrade_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    if str(evidence_pack.get("case_memory_mode") or "") != "retrieved_cases_v2_applicable":
        return
    if card.get("research_grade") != "继续深挖" and card.get("simulated_action") != "增加研究暴露":
        return
    reasons = _rag_applicable_failure_guard_reasons(evidence_pack)
    if not reasons:
        return
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
    if card.get("simulated_action") == "增加研究暴露":
        card["simulated_action"] = "保持观察"
        weight = _safe(card.get("simulated_weight_change"))
        card["simulated_weight_change"] = min(weight, 0.1) if not math.isnan(weight) else 0.1
    reason = "历史相似失败条件命中，仅可作为反证/复核清单，禁止据此升级研究暴露。"
    existing = str(card.get("counter_evidence") or "").strip()
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = str(card.get("error_reflection") or "").strip()
    note = "guardrail_applied: rag_applicable_failure_no_upgrade_v1"
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = "RAG 命中历史相似失败/缺口条件，执行层压回放入观察并保持低权重。"


def _apply_single_stock_risk_review_queue_no_raise_guardrail(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    if str(evidence_pack.get("task_mode")) not in {"single_stock", "single_stock_watch"}:
        return
    risk_rows = _single_stock_risk_review_queue_rows(evidence_pack)
    if not risk_rows:
        return
    review_rows = [row for row in risk_rows if _is_review_only_risk_queue_row(row)]
    if not review_rows:
        return
    branch_no_downweight = _risk_review_branch_no_downweight(review_rows)
    max_observe_weight = 0.1 if branch_no_downweight else 0.05

    changed = False
    if card.get("research_grade") == "继续深挖":
        card["research_grade"] = "放入观察"
        changed = True

    action = str(card.get("simulated_action") or "")
    weight = _safe(card.get("simulated_weight_change"))
    if action == "增加研究暴露":
        card["simulated_action"] = "保持观察" if branch_no_downweight else "降低研究暴露"
        card["simulated_weight_change"] = max_observe_weight
        changed = True
    elif action in {"保持观察", "降低研究暴露"}:
        if branch_no_downweight and card.get("research_grade") == "放入观察":
            target_weight = max_observe_weight if math.isnan(weight) or weight < max_observe_weight else min(weight, max_observe_weight)
            if math.isnan(weight) or target_weight != weight:
                card["simulated_weight_change"] = target_weight
                changed = True
            if action == "降低研究暴露":
                card["simulated_action"] = "保持观察"
                changed = True
        else:
            capped_weight = min(weight, max_observe_weight) if not math.isnan(weight) else max_observe_weight
            if math.isnan(weight) or capped_weight != weight:
                card["simulated_weight_change"] = capped_weight
                changed = True
        if not branch_no_downweight and action == "保持观察" and not math.isnan(weight) and weight > max_observe_weight:
            card["simulated_action"] = "降低研究暴露"
            changed = True

    if not changed:
        return

    reason = (
        "单支风险复核队列命中low_hard_counter_with_reversal_support分叉，仅禁止上调，不得因风险队列本身机械降权。"
        if branch_no_downweight
        else "单支风险复核队列为review-only工具，只能作为排雷/限额证据，不得抬升研究分级或模拟暴露。"
    )
    existing = str(card.get("counter_evidence") or "").strip()
    card["counter_evidence"] = f"{existing};{reason}" if existing and existing != "无强反证" else reason
    reflection = str(card.get("error_reflection") or "").strip()
    note = (
        "guardrail_applied: single_stock_risk_review_queue_branch_no_downweight_v1"
        if branch_no_downweight
        else "guardrail_applied: single_stock_risk_review_queue_no_raise_v1"
    )
    card["error_reflection"] = f"{reflection}; {note}" if reflection and note not in reflection else note
    card["final_agent_reasoning_summary"] = (
        "低硬反例且有支撑，风险队列仅复核，不机械降权。"
        if branch_no_downweight
        else "风险复核队列仅作排雷和限额，执行层禁止据此上调暴露。"
    )


def _single_stock_risk_review_queue_rows(evidence_pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows = evidence_pack.get("quant_tool_summaries")
    if not isinstance(rows, list):
        return []
    risk_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("tool_id") == "single_stock_risk_calibration_v2_review_queue":
            risk_rows.append(row)
    return risk_rows


def _is_review_only_risk_queue_row(row: dict[str, Any]) -> bool:
    if row.get("usable_in_agent_default") is False:
        return True
    status = _text(row.get("promotion_status")).lower()
    policy = _text(row.get("policy_status")).lower()
    evidence = _text(row.get("counter_evidence")).lower()
    return any(term in status for term in {"observe", "review", "reject"}) or "review" in policy or "review_only" in evidence


def _risk_review_branch_no_downweight(rows: list[dict[str, Any]]) -> bool:
    branches = {_text(row.get("primary_risk_branch")) for row in rows}
    if "explicit_hard_negative_event" in branches:
        return False
    return "low_hard_counter_with_reversal_support" in branches


def _rag_applicable_failure_guard_reasons(evidence_pack: dict[str, Any]) -> list[str]:
    context = _text(evidence_pack.get("retrieved_cases_context"))
    if "retrieved_cases_applicability" not in context:
        return []
    if "none" in context and "applicability=" not in context:
        return []
    reasons: list[str] = []
    if "applicability=applicable" in context and "counter-evidence before upgrading" in context:
        reasons.append("applicable_failure_case")
    if str(evidence_pack.get("variant") or "") == "no_news" and "news_hidden_or_missing" in context:
        reasons.append("news_hidden_or_missing_case")
    risk_terms = [
        "rejected_for_default",
        "do not promote",
        "bad active exposure",
        "single-channel",
        "单通道",
        "坏主动暴露",
    ]
    if any(term in context for term in risk_terms):
        reasons.append("retrieved_failure_warning")
    return reasons


def _is_default_or_full_agent_variant(value: Any) -> bool:
    variant = str(value or "deepseek_agent")
    if variant.startswith("no_") or variant == "python_only":
        return False
    return variant in {
        "deepseek_agent",
        "full_agent",
        "full_agent_with_quant_tools",
        "full_agent_without_quant_tools",
        "keyword_plus_questionnaire",
        "keyword_plus_questionnaire_guarded",
        "news_plus_financial_report",
        "news_plus_financial_report_guarded",
        "kline_weak_prompt",
    }


def _cross_channel_gap_reasons(evidence_pack: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    news = evidence_pack.get("news_features") if isinstance(evidence_pack.get("news_features"), dict) else {}
    news_missing = _safe(news.get("news_missing_rate"))
    if not news or (not math.isnan(news_missing) and news_missing >= 0.8) or "hidden" in _text(evidence_pack.get("news_signal_summary")):
        reasons.append("news_gap")

    financial = evidence_pack.get("financial_report_features") if isinstance(evidence_pack.get("financial_report_features"), dict) else {}
    financial_missing = _safe(financial.get("financial_report_missing_rate"))
    financial_status = _text(financial.get("financial_report_join_status"))
    if (
        "financial_publish_date_missing" in _text(evidence_pack.get("data_missing_flags"))
        or not financial
        or (
            _financial_status_true_missing(financial_status)
            and not math.isnan(financial_missing)
            and financial_missing >= 0.8
        )
    ):
        reasons.append("financial_gap")

    candidates = evidence_pack.get("book_skill_candidates")
    if not candidates or not _has_grounded_book_skill(candidates):
        reasons.append("book_skill_gap")

    peer = evidence_pack.get("peer_context_features") if isinstance(evidence_pack.get("peer_context_features"), dict) else {}
    ind_breadth = _safe(peer.get("tushare_industry_positive_breadth_20d"))
    ind_rel = _safe(peer.get("tushare_industry_relative_return_20d"))
    peer_hidden = "hidden" in _text(evidence_pack.get("peer_context_signal_summary"))
    if not peer or peer_hidden or (
        not math.isnan(ind_breadth)
        and not math.isnan(ind_rel)
        and ind_breadth <= 0.4
        and ind_rel < 0
    ):
        reasons.append("peer_gap_or_weak")

    if _quant_tool_confirmation_gap(evidence_pack):
        reasons.append("quant_tool_gap")

    python = evidence_pack.get("python_features") if isinstance(evidence_pack.get("python_features"), dict) else {}
    prior = _safe(python.get("prior_return_20d"))
    rsi = _safe(python.get("rsi14"))
    atr = _safe(python.get("atr20_pct"))
    if (
        (not math.isnan(prior) and prior >= 60)
        or (not math.isnan(rsi) and rsi >= 80)
        or (not math.isnan(atr) and atr >= 4)
    ):
        reasons.append("overheat_or_high_volatility")
    return reasons


def _quant_tool_confirmation_gap(evidence_pack: dict[str, Any]) -> bool:
    summary = _text(evidence_pack.get("quant_tool_signal_summary")).lower()
    if "hidden" in summary:
        return True
    rows = evidence_pack.get("quant_tool_summaries")
    if not isinstance(rows, list) or not rows:
        return True
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = _text(row.get("promotion_status")).lower()
        if row.get("usable_in_agent_default") is True and any(term in status for term in ["accept", "pass", "promot"]):
            return False
        if row.get("usable_in_agent_default") is True and "default_combo_ranker_yellow" in status:
            return False
    return True


def _has_strong_default_quant_support(evidence_pack: dict[str, Any]) -> bool:
    rows = evidence_pack.get("quant_tool_summaries")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("tool_id") != "portfolio_rev_chip_core_ranker":
            continue
        if row.get("usable_in_agent_default") is not True:
            continue
        status = _text(row.get("promotion_status")).lower()
        if "default_combo_ranker_yellow" not in status:
            continue
        quantile = _safe(row.get("score_quantile"))
        if not math.isnan(quantile) and quantile >= 0.80:
            return True
    return False


def _only_soft_confirmation_gaps(gap_reasons: list[str]) -> bool:
    hard = {"peer_gap_or_weak", "overheat_or_high_volatility", "quant_tool_gap"}
    return not any(reason in hard for reason in gap_reasons)


def _has_grounded_book_skill(candidates: Any) -> bool:
    if not isinstance(candidates, list):
        return False
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("source_book") and item.get("source_status") not in {"missing_grounded_card", "needs_grounding", "weak_until_grounded"}:
            return True
    return False


def _has_unresolved_confirmation_gap(evidence_pack: dict[str, Any]) -> bool:
    gaps = _text(evidence_pack.get("data_missing_flags"))
    if "financial_publish_date_missing" in gaps:
        return True
    candidates = evidence_pack.get("book_skill_candidates")
    if not candidates:
        return True
    for item in candidates:
        if isinstance(item, dict) and str(item.get("source_status", "")).lower() in {
            "must_resolve_before_strong_evidence",
            "needs_grounding",
            "weak_until_grounded",
        }:
            return True
    return False


def _data_missing_flags(data: dict[str, Any]) -> str:
    raw = _text(data.get("data_gaps"))
    if _financial_report_is_matched_or_neutral(data):
        raw = raw.replace("financial_publish_date_missing_or_unavailable", "")
        raw = raw.replace("financial_publish_date_missing", "")
        raw = raw.replace(";;", ";")
        raw = ";".join(part.strip() for part in raw.split(";") if part.strip())
    return raw


def _financial_report_is_matched_or_neutral(data: dict[str, Any]) -> bool:
    if _text(data.get("financial_report_join_status")) in {"event_window_matched", "no_event_in_window"}:
        return True
    count = _safe(data.get("financial_report_event_count"))
    return not math.isnan(count) and count > 0


def _financial_report_true_missing(data: dict[str, Any]) -> bool:
    return _financial_status_true_missing(_text(data.get("financial_report_join_status")))


def _financial_status_true_missing(status: str) -> bool:
    return status in {"", "unknown", "feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"}


def _assert_no_future_fields(pack: dict[str, Any]) -> None:
    leaked = _find_future_keys(pack)
    if leaked:
        raise ValueError(f"future result field leaked into evidence pack: {sorted(leaked)}")


def _find_future_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        leaked = {key for key in value if key in FUTURE_RESULT_FIELDS}
        for child in value.values():
            leaked.update(_find_future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked: set[str] = set()
        for child in value:
            leaked.update(_find_future_keys(child))
        return leaked
    return set()


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, str):
        mapping = {"低": 0.25, "较低": 0.35, "中": 0.5, "中等": 0.5, "较高": 0.7, "高": 0.8}
        text = value.strip()
        if text in mapping:
            return mapping[text]
    number = _safe(value)
    if math.isnan(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _coerce_weight(value: Any, action: Any) -> float:
    action_text = str(action)
    number = _safe(value)
    if not math.isnan(number):
        return normalize_action_weight(action_text, max(0.0, min(1.0, number)))
    mapping = {"增加研究暴露": 1.0, "保持观察": 0.6, "降低研究暴露": 0.25, "转入现金": 0.0, "信息不足不动作": 0.0}
    return normalize_action_weight(action_text, mapping.get(action_text, 0.0))


def _operation_suggestion(value: Any, action: Any) -> str:
    text = _text(value)
    if text:
        return text
    action_text = _text(action)
    if action_text == "增加研究暴露":
        return "试探买入/加仓"
    if action_text == "降低研究暴露":
        return "减仓/卖出复核"
    if action_text == "转入现金":
        return "卖出/不买"
    if action_text == "信息不足不动作":
        return "补数据后再定"
    return "持有/等待"


def _coerce_target_position(value: Any, fallback_weight: Any, action: Any) -> float:
    raw = _safe(value)
    if math.isnan(raw):
        raw = _safe(fallback_weight)
    if math.isnan(raw):
        action_text = _text(action)
        if action_text == "增加研究暴露":
            raw = 0.5
        elif action_text == "保持观察":
            raw = 0.1
        else:
            raw = 0.0
    return max(0.0, min(1.0, round(float(raw), 4)))


def _sync_operation_fields(card: dict[str, Any], evidence_pack: dict[str, Any]) -> None:
    action = _text(card.get("simulated_action"))
    weight = _safe(card.get("simulated_weight_change"))
    if math.isnan(weight):
        weight = 0.0
    target = _safe(card.get("target_position"))
    if action == "增加研究暴露":
        suggestion = _text(card.get("user_operation_suggestion"))
        if not suggestion or suggestion in {"持有/等待", "等待不买", "补数据后再定", "卖出/不买"}:
            suggestion = "试探买入/加仓" if weight < 0.7 else "加仓/继续持有"
        card["user_operation_suggestion"] = suggestion
        card["target_position"] = max(target if not math.isnan(target) else 0.0, weight)
    elif action == "降低研究暴露":
        card["user_operation_suggestion"] = "减仓/卖出复核"
        card["target_position"] = min(target if not math.isnan(target) else weight, weight)
    elif action == "转入现金":
        card["user_operation_suggestion"] = "卖出/不买"
        card["target_position"] = 0.0
    elif action == "信息不足不动作":
        card["user_operation_suggestion"] = "补数据后再定"
        card["target_position"] = 0.0
    else:
        suggestion = _text(card.get("user_operation_suggestion")) or "持有/等待"
        if suggestion in {"试探买入", "加仓", "试探买入/加仓"} and weight <= 0.2:
            suggestion = "等待不买"
        card["user_operation_suggestion"] = suggestion
        card["target_position"] = _coerce_target_position(card.get("target_position"), weight, action)

    card["position_plan"] = _text(card.get("position_plan")) or _default_position_plan(card)
    card["buy_or_add_trigger"] = _text(card.get("buy_or_add_trigger")) or _default_buy_trigger(evidence_pack)
    card["reduce_or_sell_trigger"] = _text(card.get("reduce_or_sell_trigger")) or _default_reduce_trigger(evidence_pack)
    card["review_condition"] = _text(card.get("review_condition")) or _default_review_condition(evidence_pack)


def _default_position_plan(card: dict[str, Any]) -> str:
    target = _safe(card.get("target_position"))
    if math.isnan(target):
        target = 0.0
    suggestion = _text(card.get("user_operation_suggestion")) or "持有/等待"
    return f"{suggestion}；目标仓位约{target:.0%}，下一决策点按触发条件复核。"


def _default_buy_trigger(evidence_pack: dict[str, Any]) -> str:
    if str(evidence_pack.get("task_mode") or "").startswith("single_stock"):
        return "新闻预警<0.5、财务风险<0.5、同行相对不恶化，且K线/筹码未出现强上压时，才允许试探买入或加仓一档。"
    return "保持候选池Top2/Top3，且新闻预警<0.5、财务风险<0.5、同行相对不恶化时，才允许提高仓位。"


def _default_reduce_trigger(evidence_pack: dict[str, Any]) -> str:
    if str(evidence_pack.get("task_mode") or "").startswith("single_stock"):
        return "新闻预警>=0.6、财务风险>=0.6、同行显著走弱、筹码上压或跌破关键均线时，减仓或卖出复核。"
    return "跌出候选池Top2/Top3，或新闻/财务/同行/BookSkill出现硬反证时，降仓或剔除。"


def _default_review_condition(evidence_pack: dict[str, Any]) -> str:
    context = evidence_pack.get("operation_plan_context")
    frequency = _text(context.get("decision_frequency")) if isinstance(context, dict) else ""
    if frequency:
        return f"按{frequency}复核；若有重大公告、财报或异常波动，提前复核。"
    return "下一计划决策点复核；若有重大公告、财报或异常波动，提前复核。"


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any) -> str:
    number = _safe(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.4g}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)
