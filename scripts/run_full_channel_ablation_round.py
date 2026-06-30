from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, model_concurrency_limit
from src.agent_training.deepseek_runner import decide_evidence_packs, write_jsonl
from src.agent_training.case_memory_retriever import (
    format_applicable_retrieved_cases,
    format_retrieved_cases,
    retrieve_applicable_cases,
    retrieve_cases,
)
from src.agent_training.conflict_quality_context import (
    attach_conflict_quality_contexts,
    build_walkforward_conflict_quality_rulebooks,
)
from src.agent_training.promote_context import (
    attach_promote_contexts,
    build_walkforward_promote_rulebooks,
)
from src.agent_training.dual_mode_round import (
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    build_dual_mode_evidence_packs,
    dual_mode_metrics,
    load_ground_truth,
    _is_quant_tool_ranker_preset,
    _portfolio_ranker_details,
)
from src.agent_training.evidence_pack import apply_decision_guardrails, build_evidence_pack, card_from_evidence_pack
from src.agent_training.memory_context import load_compact_memory_context
from src.agent_training.preflight import run_preflight, write_preflight_reports
from src.agent_training.quant_tool_context import (
    load_quant_tool_summaries,
    quant_tool_summary_text,
    sanitize_quant_tool_outcome,
    select_quant_tool_summaries,
)
from src.agent_training.risk_branch_policy import build_single_stock_risk_branch_policy
from src.world_model.news_questionnaire import load_news_questionnaire, questionnaire_output_fields


OUTPUT = ROOT / "reports" / "date_generalization"
DEFAULT_ACCEPTED_QUANT_TOOL_RULE_OUTCOMES_PATH = OUTPUT / "kline_peer_chip_turnover_cost_audit_v1_rule_outcomes.jsonl"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

DEFAULT_VARIANTS = [
    "full_agent",
    "no_news",
    "no_peer",
    "no_bookskill",
    "no_pps_q017",
    "no_memory",
    "no_python_gate",
    "python_only",
]
ALLOWED_VARIANTS = [
    *DEFAULT_VARIANTS,
    "full_agent_with_quant_tools",
    "full_agent_without_quant_tools",
    "no_quant_tools",
    "quant_tool_summary_only",
    "full_agent_with_hard_counter_tool",
    "full_agent_without_channel_classifier",
    "full_agent_with_risk_review_queue",
    "full_agent_without_risk_review_queue",
    "full_agent_with_opportunity_tool",
    "full_agent_without_opportunity_tool",
    "full_agent_with_action_label_tool",
    "full_agent_without_action_label_tool",
    "no_action_label_tool",
    "no_questionnaire",
    "no_branch_case_context",
    "no_analogue_case_context",
    "no_chip_context",
    "no_financial_report",
    "no_nonprice_risk_overlay",
    "keyword_only",
    "questionnaire_only",
    "news_hard_risk_only",
    "aggressive_small_entry_035",
]
DERIVED_QUESTIONNAIRE_FIELDS = [
    "ds_news_risk_score",
    "ds_news_opportunity_score",
    "ds_news_peer_support_score",
    "ds_news_policy_support_score",
    "ds_news_region_support_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_net_score",
]
PORTFOLIO_QUANT_ADOPTION_GUARD_SAFE_COLUMNS = [
    "date",
    "decision_date",
    "code",
    "guard_probability",
    "guard_threshold",
    "guard_allow_raise",
    "quant_score_pct_by_date",
    "quant_raise_candidate",
    "logistic_kline_peer_chip",
    "logistic_kline_peer_chip_regime",
    "baseline_rev_chip_score",
    "rev_chip_score_quantile",
    "manual_positive_evidence_score",
    "manual_all_channel_score",
    "logistic_all_channels",
    "logistic_channel_outcome__prob_hard_counter",
    "ml_keypoint_score",
    "ml_keypoint_selected",
    "heuristic_key_score_pct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small full-channel component ablation for dual-mode DeepSeek decisions.")
    parser.add_argument("--limit-per-mode", type=int, default=1)
    parser.add_argument("--panel-count", type=int, default=1, help="Build non-overlapping ranked sample panels per block/task mode.")
    parser.add_argument("--valid-blocks", default="H2023_2,H2024_1,H2024_2,H2025_1,H2025_2,H2026_1")
    parser.add_argument(
        "--portfolio-preset",
        default=DEFAULT_PORTFOLIO_PRESET,
        choices=[
            "rev_plus_chip_core",
            "reversal_ranker_v1",
            "no_overheat_no_evidence",
            "pullback_recovery",
            "balanced_momentum",
            "peer_confirmed_pullback",
        ],
    )
    parser.add_argument("--portfolio-date-gate", default="pool_pullback", choices=["all_dates", "pool_pullback", "pool_not_hot", "low_overheat_ratio"])
    parser.add_argument("--portfolio-row-gate", default="news_risk_low", choices=["none", "peer_relative_positive", "peer_breadth_above_half", "no_major_data_gap", "news_risk_low", "peer_and_gap_safe", "cross_channel_min2", "cross_channel_min3", "positive_confirmation_min1_no_hard", "positive_confirmation_min2", "positive_confirmation_min2_no_hard", "kline_reversal_friction_confirmed", "financial_event_quality_pc2"])
    parser.add_argument("--decision-frequency", default="every_2_weeks", choices=["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"])
    parser.add_argument("--agent-policy-version", default="full_channel_ablation_v1")
    parser.add_argument("--output-prefix", default="full_channel_ablation_small_v1")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--quant-tool-rule-outcomes", type=Path, default=DEFAULT_ACCEPTED_QUANT_TOOL_RULE_OUTCOMES_PATH)
    parser.add_argument("--quant-tool-max-items", type=int, default=6)
    parser.add_argument("--channel-classifier-scores", type=Path, default=None, help="Optional scored_detail CSV from run_channel_rule_outcome_classifier.py; matched by decision_date/code.")
    parser.add_argument("--portfolio-quant-adoption-guard", type=Path, default=None, help="Optional scored_detail CSV from train_portfolio_quant_adoption_guard.py; safe row-level context matched by decision_date/code for portfolio packs only.")
    parser.add_argument("--single-stock-risk-review-queue", type=Path, default=None, help="Optional JSONL from audit_single_stock_risk_calibration_v2.py; matched by decision_date/code for single_stock packs only.")
    parser.add_argument("--single-stock-opportunity-preview", type=Path, default=None, help="Optional JSONL from audit_single_stock_opportunity_scorer_v2.py; matched by decision_date/code for single_stock packs only.")
    parser.add_argument("--single-stock-action-label-preview", type=Path, default=None, help="Optional JSONL from audit_p0_action_label_scorer_v1.py; matched by decision_date/code for single_stock packs only.")
    parser.add_argument("--case-memory-mode", default="memory_compact_only", choices=["memory_compact_only", "retrieved_cases_v1", "retrieved_cases_v2_applicable"], help="Optional local case-memory retrieval injected before ablation variants.")
    parser.add_argument("--case-memory-top-k", type=int, default=3)
    parser.add_argument("--news-branch-case-preview", type=Path, default=None, help="Optional JSONL from audit_news_questionnaire_branch_cases.py matched by decision_date/code.")
    parser.add_argument("--analogue-case-preview", type=Path, default=None, help="Optional JSONL from audit_analogue_case_context_v2.py; global task-mode checklist context.")
    parser.add_argument("--analogue-case-max-items", type=int, default=4)
    parser.add_argument("--nonprice-risk-overlay-preview", type=Path, default=None, help="Optional JSONL from audit_nonprice_risk_overlay_v1.py; prior-only policy context.")
    parser.add_argument("--nonprice-risk-overlay-flags", type=Path, default=None, help="Optional safe flag CSV from audit_nonprice_risk_overlay_v1.py matched by decision_date/code.")
    parser.add_argument("--nonprice-risk-overlay-max-items", type=int, default=6)
    parser.add_argument(
        "--nonprice-risk-overlay-task-modes",
        default="portfolio_pool",
        help="Comma-separated task modes where nonprice overlay is visible, or 'all'. Defaults to portfolio_pool because 3-panel validation rejected P0 single-stock default visibility.",
    )
    parser.add_argument("--conflict-quality-context", default="walkforward_prior", choices=["none", "walkforward_prior"])
    parser.add_argument("--promote-context", default="none", choices=["none", "walkforward_prior"])
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--reuse-decision-ledger", action="store_true")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--user-id", default="stock_agent_full_channel_ablation")
    parser.add_argument("--sample-plan", type=Path, default=None, help="Optional safe sample plan with date/code/task_mode rows; future fields are ignored.")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)

    preflight = run_preflight(ROOT)
    write_preflight_reports(preflight, OUTPUT)
    if not preflight["ok"]:
        raise SystemExit("preflight failed; see reports/date_generalization/preflight_check.md")

    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    frame = _merge_questionnaire_scores(frame, _load_questionnaire_scores())
    blocks = _parse_blocks(args.valid_blocks)
    variants = _parse_variants(args.variants)
    quant_tool_summaries = load_quant_tool_summaries(args.quant_tool_rule_outcomes, max_items=10_000)
    if args.sample_plan:
        base_packs = _build_sample_plan_base_packs(frame, args=args, quant_tool_summaries=quant_tool_summaries)
        blocks = sorted({str(pack.get("valid_block")) for pack in base_packs if pack.get("valid_block")})
    else:
        base_packs = _build_base_packs(frame, args=args, blocks=blocks, quant_tool_summaries=quant_tool_summaries)
    channel_classifier_scores = _load_channel_classifier_scores(args.channel_classifier_scores)
    if channel_classifier_scores:
        _attach_channel_classifier_scores(base_packs, channel_classifier_scores)
    portfolio_quant_guard = _load_portfolio_quant_adoption_guard(args.portfolio_quant_adoption_guard)
    if portfolio_quant_guard:
        _attach_portfolio_quant_adoption_guard(base_packs, portfolio_quant_guard)
    risk_review_queue = _load_single_stock_risk_review_queue(args.single_stock_risk_review_queue)
    if risk_review_queue:
        _attach_single_stock_risk_review_queue(base_packs, risk_review_queue)
    opportunity_preview = _load_single_stock_opportunity_preview(args.single_stock_opportunity_preview)
    if opportunity_preview:
        _attach_single_stock_opportunity_preview(base_packs, opportunity_preview)
    action_label_preview = _load_single_stock_action_label_preview(args.single_stock_action_label_preview)
    if action_label_preview:
        _attach_single_stock_action_label_preview(base_packs, action_label_preview)
    news_branch_cases = _load_news_branch_case_preview(args.news_branch_case_preview)
    if news_branch_cases:
        _attach_news_branch_case_context(base_packs, news_branch_cases)
    analogue_cases = _load_analogue_case_preview(args.analogue_case_preview)
    if analogue_cases:
        _attach_analogue_case_context(base_packs, analogue_cases, max_items=args.analogue_case_max_items)
    nonprice_overlay = _load_nonprice_risk_overlay_preview(args.nonprice_risk_overlay_preview)
    nonprice_flags = _load_nonprice_risk_overlay_flags(args.nonprice_risk_overlay_flags)
    if nonprice_overlay and nonprice_flags:
        _attach_nonprice_risk_overlay_context(
            base_packs,
            nonprice_overlay,
            nonprice_flags,
            max_items=args.nonprice_risk_overlay_max_items,
            task_modes=_parse_task_modes(args.nonprice_risk_overlay_task_modes),
        )
    if args.conflict_quality_context == "walkforward_prior":
        rulebooks = build_walkforward_conflict_quality_rulebooks(frame, valid_blocks=blocks)
        attach_conflict_quality_contexts(base_packs, rulebooks)
    if args.promote_context == "walkforward_prior":
        promote_rulebooks = build_walkforward_promote_rulebooks(frame, valid_blocks=blocks)
        attach_promote_contexts(base_packs, promote_rulebooks)
    _attach_case_memory_retrieval(base_packs, mode=args.case_memory_mode, top_k=args.case_memory_top_k)
    questionnaire_plan_path = OUTPUT / f"{prefix}_questionnaire_sample_plan.csv"
    write_questionnaire_sample_plan(base_packs, questionnaire_plan_path)
    ablation_packs = expand_full_channel_ablation_packs(base_packs, variants)

    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    metrics_path = OUTPUT / f"{prefix}_metrics.csv"
    step_metrics_path = OUTPUT / f"{prefix}_step_metrics.csv"
    report_path = OUTPUT / f"{prefix}_summary.md"
    write_jsonl(str(evidence_path), ablation_packs)

    if args.reuse_decision_ledger:
        cards = _apply_posthoc_guardrails(_read_jsonl(decision_path), ablation_packs)
        write_jsonl(str(decision_path), cards)
        invalid = _read_jsonl(invalid_path)
        usage = pd.read_csv(usage_path) if usage_path.exists() else pd.DataFrame()
        _write_result_tables(metrics_path, step_metrics_path, report_path, cards, invalid, usage, frame, args=args, base_count=len(base_packs), pack_count=len(ablation_packs), called=True, reused=True, questionnaire_plan_path=questionnaire_plan_path)
        print("A股研究Agent")
        print(f"reused_decision_ledger=True base_packs={len(base_packs)} ablation_packs={len(ablation_packs)} cards={len(cards)} invalid={len(invalid)}")
        print(f"wrote: {report_path}")
        return

    if args.call_deepseek:
        existing_cards: list[dict[str, Any]] = []
        packs_to_call = ablation_packs
        if args.resume_missing:
            existing_cards = _apply_posthoc_guardrails(_read_jsonl(decision_path), ablation_packs)
            seen = {_card_key(card) for card in existing_cards}
            packs_to_call = [pack for pack in ablation_packs if _pack_key(pack) not in seen]
        result = decide_evidence_packs(
            packs_to_call,
            model=args.model,
            max_workers=args.max_workers,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            user_id=args.user_id,
        )
        cards = _apply_posthoc_guardrails(_dedupe_cards([*existing_cards, *result.ok_cards]), ablation_packs)
        write_jsonl(str(decision_path), cards)
        write_jsonl(str(invalid_path), result.invalid_outputs)
        new_usage = pd.DataFrame(result.usage_rows)
        previous_usage = pd.read_csv(usage_path) if args.resume_missing and usage_path.exists() else pd.DataFrame()
        usage = pd.concat([previous_usage, new_usage], ignore_index=True) if not previous_usage.empty else new_usage
        if not usage.empty:
            model_limit = model_concurrency_limit(args.model)
            usage["requested_max_workers"] = args.max_workers
            usage["model_concurrency_limit"] = model_limit
            if "effective_workers" not in usage:
                usage["effective_workers"] = max(1, min(model_limit if args.max_workers <= 0 else args.max_workers, len(packs_to_call), model_limit))
        usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
        _write_result_tables(metrics_path, step_metrics_path, report_path, cards, result.invalid_outputs, usage, frame, args=args, base_count=len(base_packs), pack_count=len(ablation_packs), called=True, reused=False, questionnaire_plan_path=questionnaire_plan_path)
        print("A股研究Agent")
        print(f"called_deepseek=True base_packs={len(base_packs)} ablation_packs={len(ablation_packs)} called_packs={len(packs_to_call)} ok_cards={len(cards)} invalid={len(result.invalid_outputs)}")
        print(f"wrote: {report_path}")
        return

    write_jsonl(str(decision_path), [])
    write_jsonl(str(invalid_path), [])
    usage = pd.DataFrame(columns=["model", "status", "total_tokens", "requested_max_workers", "effective_workers", "model_concurrency_limit"])
    usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
    planned = planned_variant_metrics(ablation_packs)
    planned.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(metrics_path, index=False, encoding="utf-8-sig")
    _write_summary(report_path, args=args, base_count=len(base_packs), pack_count=len(ablation_packs), called=False, reused=False, metrics=pd.DataFrame(), step_metrics=planned, usage=usage, invalid_count=0, questionnaire_plan_path=questionnaire_plan_path, cards=[])
    print("A股研究Agent")
    print(f"called_deepseek=False base_packs={len(base_packs)} ablation_packs={len(ablation_packs)}")
    print(f"wrote: {report_path}")


def expand_full_channel_ablation_packs(base_packs: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in base_packs:
        for variant in variants:
            rows.append(apply_full_channel_variant(base, variant))
    return rows


def apply_full_channel_variant(base_pack: dict[str, Any], variant: str) -> dict[str, Any]:
    pack = deepcopy(base_pack)
    pack["variant"] = variant
    pack["component_ablation_variant"] = variant
    if variant == "full_agent":
        pack["component_ablation_policy"] = "all available channels visible."
    elif variant == "full_agent_with_quant_tools":
        pack["component_ablation_policy"] = "all available channels visible, including quant tool summaries."
    elif variant == "full_agent_with_hard_counter_tool":
        pack["component_ablation_policy"] = "all available channels visible, including row-level channel hard-counter classifier summaries."
    elif variant == "full_agent_without_channel_classifier":
        _hide_channel_classifier_context(pack)
    elif variant == "full_agent_with_risk_review_queue":
        pack["component_ablation_policy"] = "all available channels visible, including single-stock capped risk review queue summaries when matched."
    elif variant == "full_agent_without_risk_review_queue":
        _hide_single_stock_risk_review_queue_context(pack)
    elif variant == "full_agent_with_opportunity_tool":
        pack["component_ablation_policy"] = "all available channels visible, including single-stock opportunity scorer summaries when matched."
    elif variant == "full_agent_without_opportunity_tool":
        _hide_single_stock_opportunity_tool_context(pack)
    elif variant == "full_agent_with_action_label_tool":
        pack["component_ablation_policy"] = "all available channels visible, including p0_action_label_scorer_v1 summaries when matched."
    elif variant in {"full_agent_without_action_label_tool", "no_action_label_tool"}:
        _hide_single_stock_action_label_tool_context(pack)
    elif variant in {"full_agent_without_quant_tools", "no_quant_tools"}:
        _hide_quant_tool_context(pack, variant)
    elif variant in {"no_questionnaire", "keyword_only"}:
        pack["news_semantic_questionnaire"] = {}
        _hide_news_branch_case_context(pack, variant)
        pack["component_ablation_policy"] = "hide DeepSeek semantic news questionnaire; keep keyword/event news features."
    elif variant == "no_branch_case_context":
        _hide_news_branch_case_context(pack, variant)
        pack["component_ablation_policy"] = "hide only news branch/case prior context; keep news features, semantic questionnaire, memory, and other Agent channels visible."
    elif variant == "no_analogue_case_context":
        _hide_analogue_case_context(pack, variant)
        pack["component_ablation_policy"] = "hide only time-safe analogue case context; keep news, memory, BookSkill, peer, financial, kline, chip, and quant channels visible."
    elif variant == "no_chip_context":
        _hide_chip_context(pack, variant)
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["component_ablation_policy"] = "hide only chip/cost-distribution context; keep news, memory, BookSkill, peer, financial, kline, analogue, and quant channels visible."
    elif variant == "no_financial_report":
        _hide_financial_report_context(pack, variant)
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["component_ablation_policy"] = "hide only financial report/as-of event context; keep news, memory, BookSkill, peer, kline, chip, analogue, and quant channels visible."
    elif variant == "no_nonprice_risk_overlay":
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["component_ablation_policy"] = "hide only non-price risk overlay prior context; keep raw news, financial, peer, BookSkill, memory, K-line, chip, and quant channels visible."
    elif variant == "questionnaire_only":
        pack["news_features"] = {}
        pack["news_signal_summary"] = "component ablation: keyword/event news hidden; semantic questionnaire remains visible"
        pack["component_ablation_policy"] = "hide keyword/event news statistics; keep semantic questionnaire."
    elif variant == "no_news":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = {}
        _hide_news_branch_case_context(pack, variant)
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["news_signal_summary"] = "component ablation: news and questionnaire hidden"
        pack["component_ablation_policy"] = "hide news keyword/event statistics and semantic questionnaire."
    elif variant == "news_hard_risk_only":
        _apply_news_hard_risk_only_variant(pack)
    elif variant == "aggressive_small_entry_035":
        _apply_aggressive_small_entry_context(pack)
    elif variant == "no_peer":
        pack["peer_context_features"] = {}
        pack["peer_context_signal_summary"] = "component ablation: peer context hidden"
        pack["kline_features"] = {key: value for key, value in dict(pack.get("kline_features") or {}).items() if not key.startswith("peer_kline_")}
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["component_ablation_policy"] = "hide Tushare industry/area peer context and peer_kline features."
    elif variant == "no_bookskill":
        pack["book_skill_candidates"] = []
        pack["book_skill_requirement"] = "component ablation: Book Skill hidden from DeepSeek."
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["component_ablation_policy"] = "hide grounded Book Skill candidates."
    elif variant == "no_pps_q017":
        _hide_bookskill_strategy_id(pack, "PPS-Q-017", variant)
    elif variant == "no_memory":
        pack["memory_context"] = "none"
        pack["retrieved_cases_context"] = "none"
        _hide_news_branch_case_context(pack, variant)
        _hide_analogue_case_context(pack, variant)
        _hide_nonprice_risk_overlay_context(pack, variant)
        pack["conflict_quality_context"] = "none"
        pack["promote_context"] = "none"
        pack["case_memory_mode"] = "no_memory"
        pack["component_ablation_policy"] = "hide compact memory, retrieved cases, and conflict-quality prior rules."
    elif variant == "no_python_gate":
        pack["python_features"] = {}
        pack["python_signal_summary"] = "component ablation: Python gate hidden"
        pack["component_ablation_policy"] = "hide deterministic Python gate and candidate score summary."
    elif variant == "python_only":
        _apply_python_only_variant(pack)
    elif variant == "quant_tool_summary_only":
        _apply_quant_tool_summary_only_variant(pack)
    else:
        raise ValueError(f"unknown full-channel ablation variant: {variant}")
    return pack


def planned_variant_metrics(packs: list[dict[str, Any]]) -> pd.DataFrame:
    if not packs:
        return pd.DataFrame()
    frame = pd.DataFrame(packs)
    keys = ["agent_policy_version", "variant", "step", "train_blocks", "valid_block", "task_mode"]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "planned_evidence_packs": int(len(group)),
                "decision_cards": 0,
                "invalid_outputs": 0,
                "schema_pass_rate": None,
                "called_deepseek": False,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def variant_metrics(cards: list[dict[str, Any]], invalid_outputs: list[dict[str, Any]], source_frame: pd.DataFrame, *, portfolio_preset: str) -> pd.DataFrame:
    card_frame = pd.DataFrame(cards)
    variants = set(card_frame.get("variant", pd.Series(dtype=str)).dropna().astype(str)) if not card_frame.empty else set()
    variants.update(str(item.get("evidence_pack", {}).get("variant", "")) for item in invalid_outputs if item.get("evidence_pack", {}).get("variant"))
    rows = []
    for variant in sorted(variants):
        variant_cards = card_frame[card_frame["variant"].astype(str).eq(variant)].to_dict("records") if not card_frame.empty else []
        variant_invalid = [item for item in invalid_outputs if str(item.get("evidence_pack", {}).get("variant", "")) == variant]
        metrics = dual_mode_metrics(variant_cards, source_frame, invalid_outputs=variant_invalid, portfolio_preset=portfolio_preset)
        if not metrics.empty:
            metrics.insert(0, "variant", variant)
            rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def variant_step_metrics(cards: list[dict[str, Any]], invalid_outputs: list[dict[str, Any]], source_frame: pd.DataFrame, *, portfolio_preset: str) -> pd.DataFrame:
    card_frame = pd.DataFrame(cards)
    invalid_rows = [
        {
            "agent_policy_version": item.get("agent_policy_version"),
            "variant": item.get("evidence_pack", {}).get("variant"),
            "step": item.get("evidence_pack", {}).get("step"),
            "train_blocks": item.get("evidence_pack", {}).get("train_blocks"),
            "valid_block": item.get("evidence_pack", {}).get("valid_block"),
            "task_mode": item.get("evidence_pack", {}).get("task_mode"),
        }
        for item in invalid_outputs
    ]
    keys = ["agent_policy_version", "variant", "step", "train_blocks", "valid_block", "task_mode"]
    groups: set[tuple[Any, ...]] = set()
    if not card_frame.empty:
        groups.update(tuple(row.get(key) for key in keys) for _, row in card_frame.iterrows())
    groups.update(tuple(row.get(key) for key in keys) for row in invalid_rows)
    rows = []
    for group in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        subset = card_frame.copy()
        for key, value in zip(keys, group):
            if subset.empty:
                break
            subset = subset[subset[key].astype(str).eq(str(value))]
        invalid_subset = [row for row in invalid_rows if all(str(row.get(key)) == str(value) for key, value in zip(keys, group))]
        metrics = dual_mode_metrics(subset.to_dict("records"), source_frame, invalid_outputs=[{"evidence_pack": row} for row in invalid_subset], portfolio_preset=portfolio_preset)
        if metrics.empty:
            continue
        row = {key: value for key, value in zip(keys, group)}
        mode = str(group[-1])
        mode_metrics = metrics[metrics["task_mode"].astype(str).eq(mode)]
        if mode_metrics.empty:
            continue
        row.update(mode_metrics.iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_python_only_variant(pack: dict[str, Any]) -> None:
    pack["quant_tool_summaries"] = []
    pack["quant_tool_signal_summary"] = "component ablation: hidden for python_only"
    pack["quant_tool_requirement"] = "component ablation: hidden for python_only"
    _hide_chip_context(pack, "python_only")
    pack["python_signal_summary"] = "component ablation: deterministic python features only; quant ranker preset hidden for python_only"
    pack["kline_features"] = {}
    pack["kline_signal_summary"] = "component ablation: hidden for python_only"
    pack["peer_context_features"] = {}
    pack["peer_context_signal_summary"] = "component ablation: hidden for python_only"
    pack["news_features"] = {}
    pack["news_semantic_questionnaire"] = {}
    _hide_news_branch_case_context(pack, "python_only")
    _hide_analogue_case_context(pack, "python_only")
    _hide_nonprice_risk_overlay_context(pack, "python_only")
    pack["news_signal_summary"] = "component ablation: hidden for python_only"
    pack["financial_report_features"] = {}
    pack["financial_report_signal_summary"] = "component ablation: hidden for python_only"
    pack["book_skill_candidates"] = []
    pack["book_skill_requirement"] = "component ablation: hidden for python_only"
    pack["memory_context"] = "none"
    pack["retrieved_cases_context"] = "none"
    pack["conflict_quality_context"] = "none"
    pack["promote_context"] = "none"
    pack["case_memory_mode"] = "python_only"
    pack["counter_evidence"] = "component ablation: non-python counter-evidence hidden"
    pack["component_ablation_policy"] = "only deterministic Python gate summary/features plus basic identifiers and data_missing_flags remain visible."


def _apply_quant_tool_summary_only_variant(pack: dict[str, Any]) -> None:
    pack["python_features"] = {}
    pack["python_signal_summary"] = "component ablation: hidden for quant_tool_summary_only"
    _hide_chip_context(pack, "quant_tool_summary_only")
    pack["kline_features"] = {}
    pack["kline_signal_summary"] = "component ablation: hidden for quant_tool_summary_only"
    pack["peer_context_features"] = {}
    pack["peer_context_signal_summary"] = "component ablation: hidden for quant_tool_summary_only"
    pack["news_features"] = {}
    pack["news_semantic_questionnaire"] = {}
    _hide_news_branch_case_context(pack, "quant_tool_summary_only")
    _hide_analogue_case_context(pack, "quant_tool_summary_only")
    _hide_nonprice_risk_overlay_context(pack, "quant_tool_summary_only")
    pack["news_signal_summary"] = "component ablation: hidden for quant_tool_summary_only"
    pack["financial_report_features"] = {}
    pack["financial_report_signal_summary"] = "component ablation: hidden for quant_tool_summary_only"
    pack["book_skill_candidates"] = []
    pack["book_skill_requirement"] = "component ablation: hidden for quant_tool_summary_only"
    pack["memory_context"] = "none"
    pack["retrieved_cases_context"] = "none"
    pack["conflict_quality_context"] = "none"
    pack["promote_context"] = "none"
    pack["case_memory_mode"] = "quant_tool_summary_only"
    pack["counter_evidence"] = "component ablation: non-quant counter-evidence hidden"
    pack["component_ablation_policy"] = "only quant tool summaries plus basic identifiers and data_missing_flags remain visible."


def _hide_quant_tool_context(pack: dict[str, Any], variant: str) -> None:
    pack["quant_tool_summaries"] = []
    pack["quant_tool_signal_summary"] = f"component ablation: hidden for {variant}"
    pack["quant_tool_requirement"] = f"component ablation: hidden for {variant}"
    pack["python_signal_summary"] = f"component ablation: quant ranker preset hidden for {variant}; deterministic python features retained"
    pack["component_ablation_policy"] = "hide trained quant tool summaries while keeping other Agent channels visible."


def _apply_news_hard_risk_only_variant(pack: dict[str, Any]) -> None:
    original_news = dict(pack.get("news_features") or {})
    warning = _safe_number(original_news.get("news_warning_score"))
    official = _safe_number(original_news.get("official_confirmation_score"))
    materiality = _safe_number(original_news.get("announcement_materiality_score"))
    hard_warning_visible = warning is not None and warning >= 0.7
    official_event_visible = (official is not None and official >= 0.8) or (materiality is not None and materiality >= 0.7)
    pack["news_features"] = {
        "news_hard_risk_only_policy": "only explicit hard risk and material official events are visible; low-warning, no-catalyst, stale, missing, or opportunity-only news must not downweight small_entry by itself",
        "news_count_30d": original_news.get("news_count_30d"),
        "news_missing_rate": original_news.get("news_missing_rate"),
        "news_timestamp_quality": original_news.get("news_timestamp_quality"),
        "news_warning_score": warning if hard_warning_visible else 0.0,
        "official_confirmation_score": official if official_event_visible else 0.0,
        "announcement_materiality_score": materiality if official_event_visible else 0.0,
        "news_opportunity_score": 0.0,
        "source_type": original_news.get("source_type"),
        "source_name": original_news.get("source_name"),
        "hidden_soft_news_fields": "self_news_intensity;policy_background_score;opportunity_score;low_warning;no_catalyst;missingness_uncertainty",
    }
    questionnaire = dict(pack.get("news_semantic_questionnaire") or {})
    ds_risk = _safe_number(questionnaire.get("ds_news_risk_score"))
    conflict = _safe_number(questionnaire.get("ds_news_conflict_intensity"))
    pack["news_semantic_questionnaire"] = {
        "news_semantic_questionnaire_version": questionnaire.get("news_semantic_questionnaire_version"),
        "ds_news_risk_score": ds_risk if ds_risk is not None and ds_risk >= 0.7 else 0.0,
        "ds_news_conflict_intensity": conflict if conflict is not None and conflict >= 0.7 else 0.0,
        "ds_news_self_regulatory_legal": questionnaire.get("ds_news_self_regulatory_legal"),
        "ds_news_self_material_event": questionnaire.get("ds_news_self_material_event"),
        "ds_news_official_support": questionnaire.get("ds_news_official_support"),
        "ds_news_missing_or_conflict_notes": questionnaire.get("ds_news_missing_or_conflict_notes") if (ds_risk is not None and ds_risk >= 0.7) else None,
        "hard_risk_only_policy": "semantic news can only veto/downweight on explicit hard risk or material official event; uncertainty, empty coverage, or weak mainline is soft gap only",
    }
    _hide_news_branch_case_context(pack, "news_hard_risk_only")
    _hide_nonprice_risk_overlay_context(pack, "news_hard_risk_only")
    pack["counter_evidence"] = _filter_soft_news_counter_evidence(pack.get("counter_evidence"))
    pack["news_signal_summary"] = (
        "news_hard_risk_only: keep explicit hard warning>=0.7 and material official event; "
        "hide low-warning/no-catalyst/opportunity-only/missingness so they cannot downweight small_entry alone."
    )
    pack["component_ablation_policy"] = (
        "news channel is visible only as explicit hard-risk or material official-event review; "
        "soft news gaps and news branch priors are hidden."
    )


def _apply_aggressive_small_entry_context(pack: dict[str, Any]) -> None:
    context = pack.get("operation_plan_context")
    if not isinstance(context, dict) or context.get("operation_action") != "small_buy_hold":
        pack["component_ablation_policy"] = "aggressive small-entry policy requested but no small_buy_hold operation_plan_context was present."
        return
    context["target_position"] = max(float(_safe_number(context.get("target_position")) or 0.0), 0.35)
    context["default_position_floor_if_no_hard_counter"] = 0.35
    context["default_position_ceiling"] = max(float(_safe_number(context.get("default_position_ceiling")) or 0.35), 0.5)
    context["aggressive_small_entry_policy"] = (
        "If no explicit hard counter exists, keep at least 35% target position; "
        "this is an offensive validation arm, not the conservative default."
    )
    pack["operation_plan_context"] = context
    pack["component_ablation_policy"] = (
        "aggressive validation arm: all channels visible, but small_buy_hold local plan uses 35% floor when no hard counter exists."
    )


def _filter_soft_news_counter_evidence(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "无强反证"
    hard_terms = ["明确负面", "监管", "债务", "停产", "财报质量风险", "负惊喜", "问询", "修正", "非标"]
    soft_news_terms = ["新闻覆盖不足", "新闻空窗", "新闻缺失", "无催化", "低质量新闻", "news_missing", "source_coverage", "uncertainty"]
    parts = [part.strip() for part in re.split(r"[;；]", text) if part.strip()]
    kept = []
    for part in parts:
        if any(term in part for term in soft_news_terms) and not any(term in part for term in hard_terms):
            continue
        kept.append(part)
    return ";".join(kept) if kept else "无强反证"


def _hide_channel_classifier_context(pack: dict[str, Any]) -> None:
    rows = []
    for item in pack.get("quant_tool_summaries") or []:
        if not isinstance(item, dict):
            continue
        tool_id = str(item.get("tool_id") or "")
        if not tool_id.startswith("channel_rule_outcome_classifier_v1"):
            rows.append(item)
    pack["quant_tool_summaries"] = rows
    pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
    pack["component_ablation_policy"] = "hide row-level channel rule-outcome classifier while keeping other Agent channels visible."


def _hide_single_stock_risk_review_queue_context(pack: dict[str, Any]) -> None:
    rows = []
    for item in pack.get("quant_tool_summaries") or []:
        if not isinstance(item, dict):
            continue
        tool_id = str(item.get("tool_id") or "")
        if not tool_id.startswith("single_stock_risk_calibration_v2"):
            rows.append(item)
    pack["quant_tool_summaries"] = rows
    pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
    pack["component_ablation_policy"] = "hide single-stock capped risk review queue while keeping other Agent channels visible."


def _hide_single_stock_opportunity_tool_context(pack: dict[str, Any]) -> None:
    rows = []
    for item in pack.get("quant_tool_summaries") or []:
        if not isinstance(item, dict):
            continue
        tool_id = str(item.get("tool_id") or "")
        if not tool_id.startswith("single_stock_opportunity_scorer_v2"):
            rows.append(item)
    pack["quant_tool_summaries"] = rows
    pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
    pack["component_ablation_policy"] = "hide single-stock opportunity scorer while keeping other Agent channels visible."


def _hide_single_stock_action_label_tool_context(pack: dict[str, Any]) -> None:
    rows = []
    for item in pack.get("quant_tool_summaries") or []:
        if not isinstance(item, dict):
            continue
        tool_id = str(item.get("tool_id") or "")
        if not tool_id.startswith("p0_action_label_scorer_v1"):
            rows.append(item)
    pack["quant_tool_summaries"] = rows
    pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
    context = pack.get("operation_plan_context")
    if isinstance(context, dict) and str(context.get("reason_code") or "").strip() == "p0_action_label_scorer_v1":
        pack["operation_plan_context"] = {}
    pack["component_ablation_policy"] = "hide p0_action_label_scorer_v1 while keeping other Agent channels visible."


def _hide_bookskill_strategy_id(pack: dict[str, Any], strategy_id: str, variant: str) -> None:
    target = str(strategy_id).strip()
    rows: list[dict[str, Any]] = []
    hidden = 0
    for item in pack.get("book_skill_candidates") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("strategy_id") or "").strip() == target:
            hidden += 1
            continue
        rows.append(item)
    pack["book_skill_candidates"] = rows
    pack["book_skill_requirement"] = (
        f"component ablation: Book Skill strategy_id={target} hidden from DeepSeek; "
        "all other visible Book Skill cards remain review-only evidence."
    )
    pack["component_ablation_policy"] = (
        f"hide only Book Skill strategy_id={target}; keep other BookSkill cards and all other Agent channels visible."
    )
    pack["specific_bookskill_ablation"] = {
        "variant": variant,
        "hidden_strategy_id": target,
        "hidden_cards": hidden,
        "remaining_book_skill_candidates": len(rows),
    }
    _hide_nonprice_risk_overlay_context(pack, variant)


def _hide_chip_context(pack: dict[str, Any], variant: str) -> None:
    pack["chip_features"] = {}
    pack["chip_signal_summary"] = f"component ablation: chip channel hidden for {variant}"


def _hide_financial_report_context(pack: dict[str, Any], variant: str) -> None:
    pack["financial_report_features"] = {}
    pack["financial_report_signal_summary"] = f"component ablation: financial report/as-of event channel hidden for {variant}"


def _hide_news_branch_case_context(pack: dict[str, Any], variant: str) -> None:
    pack["news_branch_case_context"] = {}
    pack["news_branch_case_requirement"] = f"component ablation: hidden for {variant}"


def _hide_analogue_case_context(pack: dict[str, Any], variant: str) -> None:
    pack["analogue_case_context"] = []
    pack["analogue_case_requirement"] = f"component ablation: hidden for {variant}"


def _hide_nonprice_risk_overlay_context(pack: dict[str, Any], variant: str) -> None:
    pack["nonprice_risk_overlay_context"] = []
    pack["nonprice_risk_overlay_requirement"] = f"component ablation: hidden for {variant}"


def _load_news_branch_case_preview(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing news branch case preview: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "pool_excess_20d",
        "rule_outcome_label",
        "gt_status",
        "gt_pass",
        "target",
        "label",
        "outcome",
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = _forbidden_json_keys(row, forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in news branch case preview line {line_number}: {sorted(leaked)}")
            date = pd.to_datetime(row.get("date"), errors="coerce")
            code = str(row.get("code") or "").zfill(6)
            if pd.isna(date) or not code:
                continue
            safe = {
                "tool_id": str(row.get("tool_id") or "news_questionnaire_branch_case_auditor"),
                "tool_version": str(row.get("tool_version") or "v1"),
                "primary_news_branch": str(row.get("primary_news_branch") or ""),
                "news_branch_tags": str(row.get("news_branch_tags") or ""),
                "branch_policy": str(row.get("branch_policy") or ""),
                "branch_rationale": str(row.get("branch_rationale") or ""),
                "prior_case_count_bucket": str(row.get("prior_case_count_bucket") or ""),
                "prior_branch_policy_status": str(row.get("prior_branch_policy_status") or ""),
                "prior_branch_policy_hint": str(row.get("prior_branch_policy_hint") or ""),
                "similar_prior_cases": row.get("similar_prior_cases") if isinstance(row.get("similar_prior_cases"), list) else [],
                "agent_use": str(row.get("agent_use") or "checklist_and_counterevidence_only_not_alpha"),
                "forbidden_use": str(row.get("forbidden_use") or "do_not_use_as_alpha"),
                "source_ref_ids": row.get("source_ref_ids") if isinstance(row.get("source_ref_ids"), list) else [],
                "research_only": True,
                "not_investment_instruction": True,
            }
            out[(date.date().isoformat(), code)] = safe
    return out


def _attach_news_branch_case_context(packs: list[dict[str, Any]], previews: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = previews.get(key)
        if not row:
            pack.setdefault("news_branch_case_context", {})
            pack.setdefault("news_branch_case_requirement", "no matched news branch case preview")
            continue
        questionnaire = pack.get("news_semantic_questionnaire")
        if not isinstance(questionnaire, dict) or not questionnaire:
            pack.setdefault("news_branch_case_context", {})
            pack.setdefault("news_branch_case_requirement", "news branch case preview hidden because questionnaire is absent")
            continue
        pack["news_branch_case_context"] = row
        pack["news_branch_case_requirement"] = (
            "news_branch_case_context is prior-only branch/case checklist; "
            "use it to separate explicit_negative_event, reversible_reversal_friction, routine_official_low_signal, "
            "soft_gap, peer_diffusion, and policy_region_direct_support. "
            "It is not alpha, not an order instruction, and contains no realized returns. "
            "Do not hard-veto a high-ranker reversal candidate from news risk alone unless explicit negative evidence is unresolved."
        )


def _load_analogue_case_preview(path: Path | None) -> list[dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing analogue case preview: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "fwd_ret_20d",
        "pool_excess_20d",
        "rank_ic",
        "positive_20d",
        "loss_gt5",
        "gt_status",
        "gt_pass",
        "target",
        "label",
        "outcome",
    }
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            leaked = _forbidden_json_keys(raw, forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in analogue case preview line {line_number}: {sorted(leaked)}")
            rows.append(_safe_analogue_case_preview_row(raw))
    rows.sort(key=_analogue_case_sort_key)
    return rows


def _safe_analogue_case_preview_row(row: dict[str, Any]) -> dict[str, Any]:
    decision_date = pd.to_datetime(row.get("date") or row.get("decision_date"), errors="coerce")
    code = str(row.get("code") or "").zfill(6) if row.get("code") not in (None, "") else ""
    task_mode = _canonical_task_mode(row.get("task_mode") or ("single_stock" if row.get("base_branch") else ""))
    analog_pos_rate = _safe_number(row.get("analog_pos_rate"))
    transfer_score = _safe_number(row.get("transfer_score"))
    return {
        "tool_id": str(row.get("tool_id") or "analogue_case_context"),
        "tool_version": str(row.get("tool_version") or "v2"),
        "date": decision_date.date().isoformat() if not pd.isna(decision_date) else "",
        "code": code,
        "time_block": str(row.get("time_block") or row.get("valid_block") or row.get("target_block") or ""),
        "task_mode": task_mode,
        "policy_profile": str(row.get("policy_profile") or "time_safe_analogue_case_context_v2"),
        "policy_status": str(row.get("policy_status") or "row_context_only"),
        "decision_frequency": str(row.get("decision_frequency") or row.get("frequency") or ""),
        "feature_group": str(row.get("feature_group") or "kline_peer_industry_case_memory"),
        "selection_mode": str(row.get("selection_mode") or "prior_matured_case_knn_context"),
        "score": _safe_number(row.get("score")) or transfer_score or analog_pos_rate,
        "confidence": _safe_number(row.get("confidence")) or analog_pos_rate,
        "risk_tier": str(row.get("risk_tier") or "review"),
        "primary_risk_branch": str(row.get("primary_risk_branch") or "analogue_case_context_checklist"),
        "risk_branch_labels": _safe_str_list(row.get("risk_branch_labels")),
        "branch_policy": str(row.get("branch_policy") or ""),
        "required_confirmation": _safe_str_list(row.get("required_confirmation")),
        "known_false_veto_risk": str(row.get("known_false_veto_risk") or ""),
        "calibration_policy": str(row.get("calibration_policy") or "walk_forward_prior_blocks_only"),
        "action_hint": str(row.get("action_hint") or "observe_checklist"),
        "usable_in_agent_default": bool(row.get("usable_in_agent_default")),
        "missing_flags": _safe_str_list(row.get("missing_flags")),
        "counter_evidence": _safe_str_list(row.get("counter_evidence")),
        "source_ref_ids": _safe_str_list(row.get("source_ref_ids")),
        "train_valid_test_blocks": str(row.get("train_valid_test_blocks") or "walk_forward_prior_only"),
        "source_variant": str(row.get("source_variant") or row.get("variant") or ""),
        "base_branch": str(row.get("base_branch") or ""),
        "analog_id": str(row.get("analog_id") or ""),
        "gate_id": str(row.get("gate_id") or ""),
        "position_cap_hint": _safe_number(row.get("position_cap_hint")),
        "transfer_score": transfer_score,
        "transfer_threshold": _safe_number(row.get("transfer_threshold")),
        "analog_neighbor_count": _safe_number(row.get("analog_neighbor_count")),
        "analog_pos_rate": analog_pos_rate,
        "analog_avg_return": _safe_number(row.get("analog_avg_return")),
        "analog_historical_tail_risk_rate": _safe_number(row.get("analog_historical_tail_risk_rate")),
        "analog_top_case_refs": str(row.get("analog_top_case_refs") or ""),
        "channel_support_count": _safe_number(row.get("channel_support_count")),
        "channel_hard_counter_count": _safe_number(row.get("channel_hard_counter_count")),
        "news_low_warning": bool(row.get("news_low_warning")) if row.get("news_low_warning") is not None else None,
        "financial_no_recent_event": bool(row.get("financial_no_recent_event")) if row.get("financial_no_recent_event") is not None else None,
        "chip_support_visible": bool(row.get("chip_support_visible")) if row.get("chip_support_visible") is not None else None,
        "agent_instruction": str(row.get("agent_instruction") or ""),
        "promotion_status": str(row.get("promotion_status") or "context_only"),
        "agent_use": "base_rate_regime_decay_failure_case_checklist_only_not_alpha",
        "forbidden_use": "do_not_use_as_standalone_alpha_or_research_grade_raise",
        "research_only": True,
        "not_investment_instruction": True,
    }


def _attach_analogue_case_context(packs: list[dict[str, Any]], previews: list[dict[str, Any]], *, max_items: int = 4) -> None:
    for pack in packs:
        task_mode = _canonical_task_mode(pack.get("task_mode"))
        pack_date = str(pack.get("decision_date") or "")
        pack_code = str(pack.get("code") or "").zfill(6)
        exact = [
            row
            for row in previews
            if _canonical_task_mode(row.get("task_mode")) in {task_mode, "all", ""}
            and str(row.get("date") or "") == pack_date
            and str(row.get("code") or "").zfill(6) == pack_code
        ]
        global_rows = [
            row
            for row in previews
            if _canonical_task_mode(row.get("task_mode")) in {task_mode, "all", ""}
            and not str(row.get("date") or "").strip()
            and not str(row.get("code") or "").strip()
        ][: max(0, int(max_items))]
        selected = exact[: max(0, int(max_items))] if exact else global_rows
        if not selected:
            pack.setdefault("analogue_case_context", [])
            pack.setdefault("analogue_case_requirement", "no applicable analogue case context preview")
            continue
        pack["analogue_case_context"] = selected
        match_scope = "row-level matched" if exact else "global checklist"
        pack["analogue_case_requirement"] = (
            f"analogue_case_context为{match_scope}的时间安全成熟历史相似案例审计，只能作为base-rate、regime衰减、"
            "失败案例和反证checklist。它不是alpha、不是当前股票收益预测，也不得单独触发买入/加仓或提高辅助分级。"
            "若analogue上下文与新闻、财报、同行、BookSkill或当前K线筹码冲突，必须说明冲突并优先使用当前多通道证据。"
        )


NONPRICE_RISK_OVERLAY_FLAG_COLUMNS = [
    "news_high_warning_any",
    "news_high_warning_official",
    "news_opportunity_with_warning",
    "news_soft_gap_missing_or_low_quality",
    "financial_high_risk_event",
    "financial_no_recent_event_soft_gap",
    "peer_industry_weak",
    "peer_area_weak",
    "bookskill_counter_high",
    "bookskill_missing_soft_gap",
    "nonprice_hard_counter_min2",
    "nonprice_soft_gap_min2",
    "nonprice_support_min2",
]
NONPRICE_RISK_OVERLAY_FLAG_SAFE_COLUMNS = [
    "date",
    "code",
    "name",
    "time_block",
    "rev_chip_score_quantile",
    "scope_high_rev_chip",
    "scope_pullback_high_rev_chip",
    *NONPRICE_RISK_OVERLAY_FLAG_COLUMNS,
    "nonprice_hard_counter_count",
    "nonprice_soft_gap_count",
    "nonprice_support_count",
    "research_only",
    "not_investment_instruction",
]


def _load_nonprice_risk_overlay_preview(path: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing nonprice risk overlay preview: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "pool_excess_20d",
        "gt_status",
        "gt_pass",
        "target",
        "label",
        "outcome",
    }
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            leaked = _forbidden_json_keys(raw, forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in nonprice risk overlay preview line {line_number}: {sorted(leaked)}")
            row = _safe_nonprice_risk_overlay_preview_row(raw)
            valid_block = _nonprice_valid_block(row)
            scope = str(row.get("feature_group") or "")
            flag = str(row.get("selection_mode") or "")
            if not valid_block or not scope or not flag:
                continue
            out[(valid_block, scope, flag)] = row
    return out


def _safe_nonprice_risk_overlay_preview_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_id": str(row.get("tool_id") or "nonprice_risk_overlay"),
        "tool_version": str(row.get("tool_version") or "nonprice_risk_overlay_v1"),
        "task_mode": _canonical_task_mode(row.get("task_mode")),
        "policy_profile": str(row.get("policy_profile") or "nonprice_risk_overlay_prior_only"),
        "policy_status": str(row.get("policy_status") or ""),
        "feature_group": str(row.get("feature_group") or ""),
        "selection_mode": str(row.get("selection_mode") or ""),
        "risk_tier": str(row.get("risk_tier") or ""),
        "primary_risk_branch": str(row.get("primary_risk_branch") or ""),
        "risk_branch_labels": _safe_str_list(row.get("risk_branch_labels")),
        "branch_policy": str(row.get("branch_policy") or ""),
        "promotion_status": str(row.get("promotion_status") or ""),
        "usable_in_agent_default": bool(row.get("usable_in_agent_default")),
        "top_features": _safe_str_list(row.get("top_features")),
        "description": str(row.get("description") or ""),
        "required_confirmation": _safe_str_list(row.get("required_confirmation")),
        "known_false_veto_risk": str(row.get("known_false_veto_risk") or ""),
        "calibration_policy": str(row.get("calibration_policy") or ""),
        "action_hint": str(row.get("action_hint") or ""),
        "counter_evidence": _safe_str_list(row.get("counter_evidence")),
        "missing_flags": _safe_str_list(row.get("missing_flags")),
        "source_ref_ids": _safe_str_list(row.get("source_ref_ids")),
        "train_valid_test_blocks": str(row.get("train_valid_test_blocks") or "walk_forward_prior_only"),
        "agent_use": "nonprice_conflict_policy_checklist_only_not_alpha",
        "forbidden_use": "do_not_use_as_standalone_alpha_or_trade_instruction",
        "research_only": True,
        "not_investment_instruction": True,
    }


def _nonprice_valid_block(row: dict[str, Any]) -> str:
    for label in _safe_str_list(row.get("risk_branch_labels")):
        if label.startswith("valid_block="):
            return label.split("=", 1)[1]
    parts = str(row.get("tool_id") or "").split(":")
    if parts:
        tail = parts[-1]
        if tail.startswith("H20"):
            return tail
    text = str(row.get("calibration_policy") or "")
    marker = "prior_only_policy_for_"
    if marker in text:
        return text.split(marker, 1)[1].split(";", 1)[0].strip()
    return ""


def _load_nonprice_risk_overlay_flags(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing nonprice risk overlay flags: {source}")
    available = pd.read_csv(source, nrows=0, encoding="utf-8-sig").columns.tolist()
    usecols = [col for col in NONPRICE_RISK_OVERLAY_FLAG_SAFE_COLUMNS if col in available]
    if "date" not in usecols or "code" not in usecols:
        return {}
    frame = pd.read_csv(source, dtype={"code": str}, low_memory=False, usecols=usecols, encoding="utf-8-sig")
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.get("date") or ""), str(row.get("code") or "").zfill(6))
        if key[0] and key[1]:
            out[key] = row.to_dict()
    return out


def _attach_nonprice_risk_overlay_context(
    packs: list[dict[str, Any]],
    previews: dict[tuple[str, str, str], dict[str, Any]],
    flags: dict[tuple[str, str], dict[str, Any]],
    *,
    max_items: int = 6,
    task_modes: set[str] | None = None,
) -> None:
    for pack in packs:
        task_mode = _canonical_task_mode(pack.get("task_mode"))
        if task_modes and task_mode not in task_modes:
            pack["nonprice_risk_overlay_context"] = []
            pack["nonprice_risk_overlay_requirement"] = (
                f"hidden for task_mode={task_mode}; nonprice overlay is default-visible only for {','.join(sorted(task_modes))}"
            )
            continue
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        flag_row = flags.get(key)
        if not flag_row:
            pack.setdefault("nonprice_risk_overlay_context", [])
            pack.setdefault("nonprice_risk_overlay_requirement", "no matched nonprice risk overlay flags")
            continue
        selected: list[dict[str, Any]] = []
        valid_block = str(pack.get("valid_block") or flag_row.get("time_block") or "")
        for scope in _active_nonprice_scopes(flag_row):
            for flag in NONPRICE_RISK_OVERLAY_FLAG_COLUMNS:
                if not _as_bool(flag_row.get(flag)):
                    continue
                preview = previews.get((valid_block, scope, flag))
                if not preview:
                    continue
                row = dict(preview)
                row["flag_active_on_current_row"] = True
                row["row_scope"] = scope
                selected.append(row)
        selected = sorted(selected, key=_nonprice_overlay_sort_key)[: max(0, int(max_items))]
        if not selected:
            pack.setdefault("nonprice_risk_overlay_context", [])
            pack.setdefault("nonprice_risk_overlay_requirement", "nonprice flags matched, but no prior-only policy preview is applicable for this block")
            continue
        pack["nonprice_risk_overlay_context"] = selected
        pack["nonprice_risk_overlay_requirement"] = (
            "nonprice_risk_overlay_context来自本地walk-forward prior-only审计，只能作为非价格冲突处理和防错杀checklist。"
            "action_hint=do_not_mechanically_veto表示该风险/软缺口历史上容易错杀，不能单独下调研究分级；"
            "action_hint=downweight_or_request_confirmation表示需要补充确认或降低置信度，但不能单独触发减仓/卖出。"
            "它不得作为独立alpha、不得替代当前新闻/财报/同行/BookSkill/K线筹码证据；"
            "nonprice_support_min2也不能单独触发继续深挖。"
        )


def _active_nonprice_scopes(row: dict[str, Any]) -> list[str]:
    scopes = ["all_evaluated"]
    if _as_bool(row.get("scope_high_rev_chip")):
        scopes.append("high_rev_chip")
    if _as_bool(row.get("scope_pullback_high_rev_chip")):
        scopes.append("pullback_high_rev_chip")
    return scopes


def _nonprice_overlay_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    action = str(row.get("action_hint") or "")
    if action == "do_not_mechanically_veto":
        action_rank = 0
    elif action == "downweight_or_request_confirmation":
        action_rank = 1
    elif action == "positive_support_but_requires_cross_channel_audit":
        action_rank = 2
    elif action == "agent_judgment_only":
        action_rank = 3
    else:
        action_rank = 4
    scope = str(row.get("row_scope") or row.get("feature_group") or "")
    scope_rank = {"pullback_high_rev_chip": 0, "high_rev_chip": 1, "all_evaluated": 2}.get(scope, 3)
    return (action_rank, scope_rank, str(row.get("selection_mode") or ""))


def _canonical_task_mode(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"portfolio_pool_optimize", "portfolio"}:
        return "portfolio_pool"
    if text in {"single_stock_watch"}:
        return "single_stock"
    return text


def _safe_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(float(number), 6)


def _analogue_case_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    status = str(row.get("promotion_status") or "")
    usable = bool(row.get("usable_in_agent_default"))
    if "relative_improvement" in status:
        status_rank = 0
    elif "latest_positive" in status:
        status_rank = 1
    elif "prior_positive" in status:
        status_rank = 2
    else:
        status_rank = 3
    return (0 if usable else 1, status_rank, -float(row.get("confidence") or 0.0), str(row.get("tool_id") or ""))


def _forbidden_json_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in forbidden or key_text.startswith("future_"):
                found.add(key_text)
            found.update(_forbidden_json_keys(item, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_json_keys(item, forbidden))
    return found


def _load_channel_classifier_scores(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing channel classifier scores: {source}")
    frame = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    if frame.empty:
        return {}
    date_col = "decision_date" if "decision_date" in frame.columns else "date"
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.get(date_col) or ""), str(row.get("code") or "").zfill(6))
        if not key[0] or not key[1]:
            continue
        out[key] = row.to_dict()
    return out


def _load_portfolio_quant_adoption_guard(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing portfolio quant adoption guard: {source}")
    available = pd.read_csv(source, nrows=0, encoding="utf-8-sig").columns.tolist()
    usecols = [col for col in PORTFOLIO_QUANT_ADOPTION_GUARD_SAFE_COLUMNS if col in available]
    if "code" not in usecols or not ({"date", "decision_date"} & set(usecols)):
        return {}
    frame = pd.read_csv(source, dtype={"code": str}, low_memory=False, usecols=usecols, encoding="utf-8-sig")
    if frame.empty:
        return {}
    date_col = "decision_date" if "decision_date" in frame.columns else "date"
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    for col in PORTFOLIO_QUANT_ADOPTION_GUARD_SAFE_COLUMNS:
        if col in {"date", "decision_date", "code", "guard_allow_raise", "quant_raise_candidate", "ml_keypoint_selected"}:
            continue
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.get(date_col) or ""), str(row.get("code") or "").zfill(6))
        if not key[0] or not key[1]:
            continue
        out[key] = row.to_dict()
    return out


def _attach_portfolio_quant_adoption_guard(packs: list[dict[str, Any]], guard: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        if str(pack.get("task_mode") or "") != "portfolio_pool":
            continue
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = guard.get(key)
        if not row:
            continue
        existing = pack.get("quant_tool_summaries")
        if isinstance(existing, list):
            rows = [item for item in existing if isinstance(item, dict)]
        elif isinstance(existing, dict):
            rows = [existing]
        else:
            rows = []
        rows.append(_portfolio_quant_adoption_guard_quant_tool_row(row, pack))
        pack["quant_tool_summaries"] = rows
        pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
        pack["quant_tool_requirement"] = (
            str(pack.get("quant_tool_requirement") or "")
            + " portfolio_quant_adoption_guard_v1 是当前股票/日期的行级采用保护上下文；"
            + "它优先于全局工具验收摘要解释当前样本，若row-level quant percentile低于0.25，"
            + "只能中和全局accepted工具的正向抬权，不能把低分位本身当作硬负面或自动降权；"
            + "若guard_probability低于阈值，也只能阻止quant来源抬权。最终仍需新闻/财报/同行/BookSkill/K线筹码共同判断。"
            + "若sampler_context为ordinary_control_midkey，量化工具仅能作为背景诊断；"
            + "除非至少两个非量化通道明确正向确认，否则不得因quant工具提高研究权重或写partially_adopted。"
            + "v4复盘规则：fresh-date面板显示partially_adopted会导致过度升权；当新闻/财报/同行/BookSkill"
            + "均缺少目标特异正向确认，或同时存在chip_overhang/data_missing软缺口束时，必须把quant工具写成"
            + "not_adopted_counter_evidence或background_diagnostic，不能写partially_adopted，且不得把模拟权重提高到0.05以上。"
        )


def _portfolio_quant_adoption_guard_quant_tool_row(row: dict[str, Any], evidence_pack: dict[str, Any] | None = None) -> dict[str, Any]:
    guard_probability = _optional_unit_num(row.get("guard_probability"))
    guard_threshold = _optional_unit_num(row.get("guard_threshold"))
    quantile = _optional_unit_num(row.get("quant_score_pct_by_date"))
    quant_score = _optional_unit_num(row.get("logistic_kline_peer_chip"))
    quant_raise_candidate = _as_bool(row.get("quant_raise_candidate"))
    guard_allow_raise = _as_bool(row.get("guard_allow_raise"))
    sampler_context = str((evidence_pack or {}).get("sampler_context") or (evidence_pack or {}).get("sample_panel_id") or "")
    ordinary_control = "ordinary_control_midkey" in sampler_context
    risk_tier = "row_quant_context_missing_percentile"
    primary_branch = "row_level_quant_context_missing"
    action_hint = "do_not_raise_from_quant_without_row_level_percentile"
    counter = ["row_level_context_required", "global_tool_summary_not_row_signal"]
    no_raise_without_nonquant = ordinary_control
    if quantile is not None and quantile < 0.25:
        risk_tier = "row_quant_low_percentile_do_not_raise"
        primary_branch = "low_row_quant_percentile"
        action_hint = "neutralize_global_quant_raise_decide_from_non_quant_channels_do_not_use_as_negative_veto"
        counter.append("cap_quant_pct_lt_0_25_best_v2_replay_but_not_default")
        no_raise_without_nonquant = True
    elif guard_probability is not None and guard_threshold is not None and not guard_allow_raise:
        risk_tier = "adoption_guard_blocks_raise"
        primary_branch = "guard_probability_below_threshold"
        action_hint = "block_quant_raise_only_decide_from_non_quant_channels_do_not_use_as_negative_veto"
        counter.append("learned_guard_observe_only")
        no_raise_without_nonquant = True
    elif quant_raise_candidate and guard_allow_raise:
        risk_tier = "row_quant_context_allows_review_not_auto_raise"
        primary_branch = "row_level_quant_candidate_requires_agent_audit"
        action_hint = "review_only_can_partially_adopt_only_if_other_channels_confirm"
        counter.append("not_standalone_positive_alpha")
    else:
        risk_tier = "row_not_quant_raise_candidate"
        primary_branch = "quant_ranker_not_row_candidate"
        action_hint = "treat_as_background_context_not_raise_signal"
        counter.append("not_quant_raise_candidate")
        no_raise_without_nonquant = True
    required_confirmation = [
        "row_level_quant_percentile_ge_0.25",
        "acceptable_reversal_friction",
        "at_least_two_target_specific_nonquant_confirmations_from_news_financial_peer_bookskill_or_announcement",
        "no_soft_gap_bundle_news_financial_peer_bookskill_chip_overhang_data_missing",
        "global_tool_summary_cannot_override_row_context",
    ]
    if ordinary_control:
        primary_branch = f"{primary_branch}_ordinary_control_midkey"
        action_hint = "ordinary_control_midkey_quant_diagnostic_only_no_partially_adopt_without_two_nonquant_confirmations"
        counter.append("ordinary_control_midkey_quant_no_raise_by_default")
        required_confirmation.extend(
            [
                "ordinary_control_midkey_requires_at_least_two_nonquant_confirmations",
                "do_not_partially_adopt_quant_on_ordinary_control_without_confirmation",
            ]
        )
    if no_raise_without_nonquant:
        counter.append("v4_no_quant_raise_without_nonquant_confirmation")
        required_confirmation.append("if_confirmation_absent_write_not_adopted_counter_evidence_and_keep_weight_le_0_05")
    score = guard_probability if guard_probability is not None else quant_score
    return sanitize_quant_tool_outcome(
        {
            "tool_id": "portfolio_quant_adoption_guard_v1_row_context",
            "tool_version": "row_level_quant_percentile_guard_v4_no_raise_without_nonquant_confirmation",
            "task_mode": "portfolio_pool",
            "policy_profile": "row_level_quant_adoption_context",
            "policy_status": "observe_not_default",
            "decision_frequency": "scheduled_keypoint_or_weekly",
            "feature_group": "kline_peer_chip_positive_channel_keypoint",
            "selection_mode": "row_level_quant_percentile_and_guard_probability",
            "score": score,
            "score_quantile": quantile,
            "confidence": score,
            "risk_tier": risk_tier,
            "primary_risk_branch": primary_branch,
            "required_confirmation": required_confirmation,
            "known_false_veto_risk": (
                "cap_quant_pct_lt_0_25 improved portfolio_keypoint_flash_v2 replay "
                "but still did not beat no-quant control; diagnostic only; "
                "low row-level quant percentile neutralizes quant adoption but is not a standalone negative veto; "
                "freshdate8_flash_v1 showed partially_adopted can over-raise when soft gaps dominate"
            ),
            "calibration_policy": (
                f"guard_probability={_fmt_optional(guard_probability)}; "
                f"guard_threshold={_fmt_optional(guard_threshold)}; "
                f"quant_score_pct_by_date={_fmt_optional(quantile)}; "
                f"quant_raise_candidate={str(quant_raise_candidate).lower()}; "
                f"guard_allow_raise={str(guard_allow_raise).lower()}"
            ),
            "action_hint": action_hint,
            "usable_in_agent_default": False,
            "top_features": [
                "guard_probability",
                "quant_score_pct_by_date",
                "logistic_kline_peer_chip",
                "logistic_kline_peer_chip_regime",
                "ml_keypoint_score",
                "logistic_channel_outcome__prob_hard_counter",
            ],
            "missing_flags": [],
            "counter_evidence": counter,
            "source_ref_ids": [
                "reports/date_generalization/portfolio_quant_adoption_guard_v1_top10_min05.md",
                "reports/date_generalization/portfolio_keypoint_flash_v2_findings.md",
            ],
            "train_valid_test_blocks": "walkforward_prior_blocks_then_H2026_replay; row-level scores only; no future labels exposed",
            "promotion_status": "observe_not_default",
            "research_only": True,
            "not_investment_instruction": True,
        }
    )


def _load_single_stock_risk_review_queue(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing single-stock risk review queue: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "pool_excess_20d",
        "rule_outcome_label",
        "single_stock_label",
        "single_stock_action",
        "gt_status",
        "gt_pass",
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(set(row) & forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in risk review queue line {line_number}: {leaked}")
            date = pd.to_datetime(row.get("date"), errors="coerce")
            code = str(row.get("code") or "").zfill(6)
            if pd.isna(date) or not code:
                continue
            out[(date.date().isoformat(), code)] = row
    return out


def _attach_single_stock_risk_review_queue(packs: list[dict[str, Any]], queue: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        if str(pack.get("task_mode") or "") != "single_stock":
            continue
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = queue.get(key)
        if not row:
            continue
        existing = pack.get("quant_tool_summaries")
        if isinstance(existing, list):
            rows = [item for item in existing if isinstance(item, dict)]
        elif isinstance(existing, dict):
            rows = [existing]
        else:
            rows = []
        rows.append(_single_stock_risk_review_quant_tool_row(row, pack))
        pack["quant_tool_summaries"] = rows
        pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
        pack["quant_tool_requirement"] = (
            str(pack.get("quant_tool_requirement") or "")
            + " single_stock_risk_calibration_v2 是单支风险复核队列，只能用于降权/补证/风险解释；"
            + "不得提高机会暴露，fixed15候选仍需下一OOT或独立DS shard验证。"
            + " risk_branch_policy_v1 会区分明确负面、软缺口、过热/反转摩擦和同行相对落后；"
            + "soft gap 或 reversal friction 不能机械压权。"
        )


def _single_stock_risk_review_quant_tool_row(row: dict[str, Any], evidence_pack: dict[str, Any] | None = None) -> dict[str, Any]:
    risk_score = _num(row.get("risk_score"))
    priority = _num(row.get("review_priority_score"))
    cap = _num(row.get("cap_pct"))
    tier = str(row.get("risk_tier") or "unknown_risk_tier")
    grade = str(row.get("research_grade") or "放入观察")
    status = str(row.get("policy_status") or "review_only")
    branch = build_single_stock_risk_branch_policy(evidence_pack or {}, row)
    action_hint = (
        f"review_only_downweight_or_request_more_evidence; never_raise_opportunity_from_this_tool; "
        f"{branch['branch_action_hint']}"
    )
    return sanitize_quant_tool_outcome(
        {
            "tool_id": "single_stock_risk_calibration_v2_review_queue",
            "tool_version": str(row.get("tool_version") or "capped_review_queue_v2"),
            "task_mode": "single_stock_watch",
            "policy_profile": "risk_review_only",
            "policy_status": status,
            "decision_frequency": str(row.get("decision_frequency") or "scheduled_twice_weekly_or_key_points"),
            "feature_group": "single_stock_channel_augmented_risk",
            "selection_mode": f"capped_review_queue_cap_{cap:.3f}",
            "cap_pct": cap,
            "tool_grade": grade,
            "score": risk_score,
            "score_quantile": priority,
            "confidence": priority,
            "risk_tier": tier,
            "primary_risk_branch": branch["primary_risk_branch"],
            "risk_branch_labels": branch["risk_branch_labels"],
            "branch_policy": branch["branch_action_hint"],
            "required_confirmation": [
                "news_quality",
                "financial_event",
                "peer_weakness",
                "bookskill_applicability",
                "kline_or_chip_risk",
                *branch["branch_required_confirmation"],
            ],
            "known_false_veto_risk": (
                "review_queue_can_miss_positive_reversal_cases; "
                f"{branch['branch_false_veto_risk']}; do_not_raise_or_remove_without_cross_channel_confirmation"
            ),
            "calibration_policy": (
                f"{status}; cap_pct={cap:.3f}; tool_grade={grade}; "
                f"primary_branch={branch['primary_risk_branch']}; next_oot_validation_required"
            ),
            "action_hint": action_hint,
            "usable_in_agent_default": False,
            "top_features": [
                "risk_score",
                "review_priority_score",
                "channel_hard_counter_prob",
                "channel_soft_gap_prob",
                "channel_positive_support_prob",
            ],
            "missing_flags": [],
            "counter_evidence": [
                "single_stock_risk_review_queue",
                tier,
                f"tool_grade={grade}",
                branch["primary_risk_branch"],
                "review_only_not_trade_instruction",
            ],
            "source_ref_ids": [
                "reports/date_generalization/single_stock_risk_calibration_v2.md",
                "reports/date_generalization/single_stock_risk_calibration_v2_review_queue.jsonl",
            ],
            "train_valid_test_blocks": "rolling validation-selected capped queue; current row score only; no future labels exposed",
            "promotion_status": "observe_review_only",
            "research_only": True,
            "not_investment_instruction": True,
        }
    )


def _load_single_stock_opportunity_preview(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing single-stock opportunity preview: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "pool_excess_20d",
        "rule_outcome_label",
        "single_stock_label",
        "single_stock_action",
        "gt_status",
        "gt_pass",
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(set(row) & forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in opportunity preview line {line_number}: {leaked}")
            date = pd.to_datetime(row.get("date"), errors="coerce")
            code = str(row.get("code") or "").zfill(6)
            if pd.isna(date) or not code:
                continue
            out[(date.date().isoformat(), code)] = row
    return out


def _attach_single_stock_opportunity_preview(packs: list[dict[str, Any]], preview: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        if str(pack.get("task_mode") or "") != "single_stock":
            continue
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = preview.get(key)
        if not row:
            continue
        existing = pack.get("quant_tool_summaries")
        if isinstance(existing, list):
            rows = [item for item in existing if isinstance(item, dict)]
        elif isinstance(existing, dict):
            rows = [existing]
        else:
            rows = []
        rows.append(_single_stock_opportunity_quant_tool_row(row))
        pack["quant_tool_summaries"] = rows
        pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
        pack["quant_tool_requirement"] = (
            str(pack.get("quant_tool_requirement") or "")
            + " single_stock_opportunity_scorer_v2 是单支机会候选摘要，可作为正向研究线索；"
            + "必须经过BookSkill、新闻/财报、同行、K线/筹码和风险队列审计，不能覆盖硬反证。"
        )


def _load_single_stock_action_label_preview(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or str(path).strip() in {"", "none", "None"}:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing single-stock action-label preview: {source}")
    forbidden = {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "pool_excess_20d",
        "rule_outcome_label",
        "single_stock_label",
        "single_stock_action",
        "entry_label",
        "strong_entry_label",
        "reduce_label",
        "gt_status",
        "gt_pass",
        "positive_20d",
        "loss_gt5",
        "loss_gt10",
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = _forbidden_json_keys(row, forbidden)
            if leaked:
                raise ValueError(f"future/result field leaked in action-label preview line {line_number}: {sorted(leaked)}")
            date = pd.to_datetime(row.get("date"), errors="coerce")
            code = str(row.get("code") or "").zfill(6)
            if pd.isna(date) or not code:
                continue
            key = (date.date().isoformat(), code)
            existing = out.get(key)
            if existing is None or _action_label_preview_preference(row) > _action_label_preview_preference(existing):
                out[key] = row
    return out


def _action_label_preview_preference(row: dict[str, Any]) -> tuple[int, int, float, float]:
    policy = str(row.get("policy_name") or "")
    model = str(row.get("model") or "")
    edge = _num(row.get("action_edge_score")) or -999.0
    target = _num(row.get("target_position")) or 0.0
    return (
        1 if policy == "precision_entry_v1" else 0,
        1 if model == "hgb" else 0,
        float(edge),
        float(target),
    )


def _attach_single_stock_action_label_preview(packs: list[dict[str, Any]], preview: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        if str(pack.get("task_mode") or "") != "single_stock":
            continue
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = preview.get(key)
        if not row:
            continue
        existing = pack.get("quant_tool_summaries")
        if isinstance(existing, list):
            rows = [item for item in existing if isinstance(item, dict)]
        elif isinstance(existing, dict):
            rows = [existing]
        else:
            rows = []
        rows.append(_single_stock_action_label_quant_tool_row(row))
        pack["quant_tool_summaries"] = rows
        pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
        pack["quant_tool_requirement"] = (
            str(pack.get("quant_tool_requirement") or "")
            + " p0_action_label_scorer_v1 是entry/reduce分动作工具，只能作为Agent审计证据；"
            + "其target_position是本地候选仓位，不是自动交易指令，必须被新闻/财报、同行、BookSkill、K线/筹码和风险反证复核。"
        )


def _single_stock_action_label_quant_tool_row(row: dict[str, Any]) -> dict[str, Any]:
    entry_prob = _num(row.get("entry_prob"))
    strong_entry_prob = _num(row.get("strong_entry_prob"))
    reduce_prob = _num(row.get("reduce_prob"))
    edge = _num(row.get("action_edge_score"))
    entry_threshold = _num(row.get("entry_threshold"))
    reduce_threshold = _num(row.get("reduce_threshold"))
    target_position = _num(row.get("target_position"))
    operation_hint = str(row.get("operation_hint") or "")
    source_refs = [item for item in str(row.get("source_ref_ids") or "").split(";") if item]
    if reduce_prob is not None and reduce_threshold is not None and reduce_prob >= reduce_threshold:
        risk_tier = "action_label_reduce_or_avoid_review"
        action_hint = "review_reduce_or_avoid_with_agent_counterevidence"
    elif target_position is not None and target_position >= 0.45:
        risk_tier = "action_label_entry_candidate_needs_audit"
        action_hint = "review_trial_buy_or_add_only_if_cross_channel_confirms"
    elif target_position is not None and target_position > 0:
        risk_tier = "action_label_support_candidate_low_weight"
        action_hint = "review_small_hold_or_watch_until_confirmation"
    else:
        risk_tier = "action_label_wait_or_background"
        action_hint = "background_only_wait_for_better_evidence"
    return sanitize_quant_tool_outcome(
        {
            "tool_id": "p0_action_label_scorer_v1",
            "tool_version": "multi_action_hgb_precision_entry_v1",
            "task_mode": "single_stock_watch",
            "policy_profile": str(row.get("policy_name") or "precision_entry_v1"),
            "policy_status": "yellow_entry_candidate_requires_confirmation",
            "decision_frequency": str(row.get("frequency") or "every_2_weeks"),
            "feature_group": str(row.get("feature_group") or ""),
            "selection_mode": str(row.get("model") or ""),
            "cap_pct": target_position,
            "tool_grade": operation_hint,
            "score": edge,
            "score_quantile": entry_prob,
            "confidence": entry_prob,
            "risk_tier": risk_tier,
            "required_confirmation": [
                "bookskill_applicability",
                "news_financial_hard_counter_check",
                "peer_relative_context",
                "kline_chip_context",
                "risk_review_queue_check",
                "do_not_use_as_standalone_buy_signal",
            ],
            "known_false_veto_risk": (
                "broad H2026 active_pos20 remains below 0.60 and panel p10 is weak; "
                "do not treat yellow status as a veto. Use it as positive entry evidence only after current hard-counter audit; "
                "soft gaps should usually downgrade position rather than zero a high-edge candidate."
            ),
            "calibration_policy": (
                f"entry_prob={_fmt_optional(entry_prob)}; strong_entry_prob={_fmt_optional(strong_entry_prob)}; "
                f"reduce_prob={_fmt_optional(reduce_prob)}; entry_threshold={_fmt_optional(entry_threshold)}; "
                f"reduce_threshold={_fmt_optional(reduce_threshold)}; target_position={_fmt_optional(target_position)}"
            ),
            "action_hint": action_hint,
            "usable_in_agent_default": True,
            "top_features": [
                "entry_prob",
                "strong_entry_prob",
                "reduce_prob",
                "action_edge_score",
                "target_position",
            ],
            "missing_flags": [],
            "counter_evidence": [
                "not_standalone_buy_signal",
                "requires_cross_channel_confirmation",
                "news_financial_only_failed_as_standalone_alpha",
            ],
            "source_ref_ids": [
                *source_refs,
                "reports/date_generalization/p0_action_label_scorer_v1_findings.md",
            ],
            "train_valid_test_blocks": "H2023_1-H2025_2 train/validation then H2026 OOT; preview row has no future labels",
            "promotion_status": "yellow_candidate_needs_flash_ablation",
            "research_only": True,
            "not_investment_instruction": True,
        }
    )


def _attach_case_memory_retrieval(packs: list[dict[str, Any]], *, mode: str, top_k: int) -> None:
    if mode in {"", "none", "memory_compact_only"}:
        for pack in packs:
            pack.setdefault("case_memory_mode", "memory_compact_only")
            pack.setdefault("retrieved_cases_context", "none")
        return
    if mode not in {"retrieved_cases_v1", "retrieved_cases_v2_applicable"}:
        raise ValueError(f"unknown case memory mode: {mode}")
    limit = max(1, int(top_k))
    for pack in packs:
        if mode == "retrieved_cases_v1":
            context = format_retrieved_cases(retrieve_cases(ROOT, pack, top_k=limit))
        else:
            context = format_applicable_retrieved_cases(retrieve_applicable_cases(ROOT, pack, top_k=limit))
        pack["case_memory_mode"] = mode
        pack["retrieved_cases_context"] = context


def _single_stock_opportunity_quant_tool_row(row: dict[str, Any]) -> dict[str, Any]:
    score = _num(row.get("opportunity_score"))
    quantile = _num(row.get("opportunity_quantile_in_date"))
    threshold = _num(row.get("opportunity_threshold"))
    status = str(row.get("tool_status") or "observe_candidate")
    grade = str(row.get("research_grade") or "放入观察")
    top_features = [item for item in str(row.get("top_feature_names") or "").split(";") if item]
    source_refs = [item for item in str(row.get("source_ref_ids") or "").split(",") if item]
    required = str(row.get("required_confirmation") or "normal_cross_channel_review")
    return sanitize_quant_tool_outcome(
        {
            "tool_id": "single_stock_opportunity_scorer_v2",
            "tool_version": str(row.get("tool_version") or "safe_orthogonal_channels_v2"),
            "task_mode": "single_stock_watch",
            "policy_profile": "opportunity_candidate_summary",
            "policy_status": status,
            "decision_frequency": "scheduled_twice_weekly_or_key_points",
            "feature_group": str(row.get("feature_group") or "baseline_existing"),
            "selection_mode": str(row.get("model_variant") or "additive_bin_baseline_existing"),
            "cap_pct": threshold,
            "tool_grade": grade,
            "score": score,
            "score_quantile": quantile,
            "confidence": quantile,
            "risk_tier": f"opportunity_{status}",
            "required_confirmation": [
                required,
                "bookskill_applicability",
                "news_or_announcement_quality",
                "financial_event_or_no_event_context",
                "peer_relative_context",
                "risk_review_queue_check",
            ],
            "known_false_veto_risk": "opportunity_summary_can_be_wrong_when_news_financial_peer_bookskill_or_risk_queue_conflicts",
            "calibration_policy": f"{status}; threshold={threshold:.4f}; grade={grade}; agent_must_audit_cross_channels",
            "action_hint": "opportunity_candidate_summary_requires_agent_audit",
            "usable_in_agent_default": status == "green_candidate",
            "top_features": top_features[:12],
            "missing_flags": [],
            "counter_evidence": [
                "not_a_standalone_decision",
                "news_financial_only_failed_lift",
                "wide_logistic_is_diagnostic_only",
            ],
            "source_ref_ids": [
                *source_refs,
                "reports/date_generalization/single_stock_opportunity_scorer_v2.md",
            ],
            "train_valid_test_blocks": "rolling train/validation with H2026 OOT preview; no future labels exposed",
            "promotion_status": "green_candidate_requires_cross_channel_audit" if status == "green_candidate" else "observe_or_reject",
            "research_only": True,
            "not_investment_instruction": True,
        }
    )


def _attach_channel_classifier_scores(packs: list[dict[str, Any]], scores: dict[tuple[str, str], dict[str, Any]]) -> None:
    for pack in packs:
        key = (str(pack.get("decision_date") or ""), str(pack.get("code") or "").zfill(6))
        row = scores.get(key)
        if not row:
            continue
        existing = pack.get("quant_tool_summaries")
        if isinstance(existing, list):
            rows = [item for item in existing if isinstance(item, dict)]
        elif isinstance(existing, dict):
            rows = [existing]
        else:
            rows = []
        rows.extend(_channel_classifier_quant_tool_rows(row, task_mode=str(pack.get("task_mode") or "")))
        pack["quant_tool_summaries"] = rows
        pack["quant_tool_signal_summary"] = quant_tool_summary_text(rows)
        pack["quant_tool_requirement"] = (
            str(pack.get("quant_tool_requirement") or "")
            + " channel_rule_outcome_classifier_v1 是按当前date/code匹配的通道裁决工具；"
            + "hard_counter必须按risk_tier校准使用，不能作为单阈值must_remove；"
            + "positive_support_v1已被拒绝，不能单独提高研究暴露。"
        )


def _channel_classifier_quant_tool_rows(row: dict[str, Any], *, task_mode: str) -> list[dict[str, Any]]:
    hard = _num(row.get("logistic_channel_outcome__prob_hard_counter"))
    soft = _num(row.get("logistic_channel_outcome__prob_soft_gap"))
    positive = _num(row.get("logistic_channel_outcome__prob_positive_support"))
    hard_policy = _hard_counter_calibration_policy(hard=hard, soft=soft, positive=positive)
    return [
        sanitize_quant_tool_outcome(
            {
                "tool_id": "channel_rule_outcome_classifier_v1_hard_counter",
                "tool_version": "channel_rule_outcome_classifier_v1",
                "task_mode": task_mode,
                "feature_group": "news_financial_peer_chip_kline_bookskill",
                "selection_mode": "hard_counter",
                "score": hard,
                "score_quantile": hard,
                "confidence": hard,
                "risk_tier": hard_policy["risk_tier"],
                "required_confirmation": hard_policy["required_confirmation"],
                "known_false_veto_risk": hard_policy["known_false_veto_risk"],
                "calibration_policy": hard_policy["calibration_policy"],
                "action_hint": hard_policy["action_hint"],
                "usable_in_agent_default": False,
                "top_features": [
                    "news_missing_rate",
                    "financial_report_missing_rate",
                    "tushare_industry_positive_breadth_20d",
                    "upper_overhang",
                    "lower_support",
                    "bookskill_available_flag",
                ],
                "missing_flags": [],
                "counter_evidence": [
                    "v1_observe_only",
                    "hard_counter_guard_not_positive_alpha",
                    hard_policy["risk_tier"],
                    hard_policy["known_false_veto_risk"],
                ],
                "source_ref_ids": [
                    "reports/date_generalization/channel_rule_outcome_classifier_v1.md",
                    "reports/date_generalization/channel_hard_counter_threshold_policy_v1.md",
                ],
                "train_valid_test_blocks": "walk_forward_prior_blocks; current row score only, no future labels exposed",
                "promotion_status": "accepted_guard_candidate",
                "research_only": True,
                "not_investment_instruction": True,
            }
        ),
        sanitize_quant_tool_outcome(
            {
                "tool_id": "channel_rule_outcome_classifier_v1_soft_gap",
                "tool_version": "channel_rule_outcome_classifier_v1",
                "task_mode": task_mode,
                "feature_group": "news_financial_peer_chip_kline_bookskill",
                "selection_mode": "soft_gap",
                "score": soft,
                "score_quantile": soft,
                "confidence": soft,
                "risk_tier": "soft_gap_missing_as_confidence_discount",
                "required_confirmation": ["target_news_or_financial_or_peer_or_bookskill_confirmation"],
                "known_false_veto_risk": "soft_gap_can_be_positive_do_not_zero_by_missingness",
                "calibration_policy": "soft_gap is hygiene/uncertainty only; not positive alpha",
                "action_hint": "treat_missing_as_confidence_discount_observe",
                "usable_in_agent_default": False,
                "top_features": ["news_missing_rate", "financial_report_missing_rate", "bookskill_available_flag"],
                "missing_flags": [],
                "counter_evidence": ["v1_observe_latest_weak", "soft_gap_not_positive_alpha"],
                "source_ref_ids": ["reports/date_generalization/channel_rule_outcome_classifier_v1.md"],
                "train_valid_test_blocks": "walk_forward_prior_blocks; current row score only, no future labels exposed",
                "promotion_status": "observe_soft_gap_prior_ok_latest_weak",
                "research_only": True,
                "not_investment_instruction": True,
            }
        ),
        sanitize_quant_tool_outcome(
            {
                "tool_id": "channel_rule_outcome_classifier_v1_positive_support_rejected",
                "tool_version": "channel_rule_outcome_classifier_v1",
                "task_mode": task_mode,
                "feature_group": "news_financial_peer_chip_kline_bookskill",
                "selection_mode": "positive_support",
                "score": positive,
                "score_quantile": positive,
                "confidence": positive,
                "risk_tier": "positive_support_rejected_for_promotion",
                "required_confirmation": ["ignore_positive_support_v1_for_upgrade"],
                "known_false_veto_risk": "not_applicable_positive_support_v1_rejected",
                "calibration_policy": "positive_support_v1 failed prior stability; diagnostic only",
                "action_hint": "do_not_promote_from_v1",
                "usable_in_agent_default": False,
                "top_features": ["positive_support_v1_rejected"],
                "missing_flags": [],
                "counter_evidence": ["prior_positive_support_unstable"],
                "source_ref_ids": ["reports/date_generalization/channel_rule_outcome_classifier_v1.md"],
                "train_valid_test_blocks": "walk_forward_prior_blocks; current row score only, no future labels exposed",
                "promotion_status": "rejected_or_diagnostic_only",
                "research_only": True,
                "not_investment_instruction": True,
            }
        ),
    ]


def _hard_counter_calibration_policy(*, hard: float, soft: float, positive: float) -> dict[str, Any]:
    if hard >= 0.95:
        return {
            "risk_tier": "hard_counter_high_risk_review_ge_0.95",
            "required_confirmation": [
                "negative_or_missing_news_quality",
                "financial_event_or_disclosure_gap",
                "peer_or_industry_weakness",
                "bookskill_applicability_or_failure_condition",
                "chip_overhang_or_overheat",
            ],
            "known_false_veto_risk": "medium_false_veto_risk_soft_gap_possible",
            "calibration_policy": "high-risk review tier; strong downweight only when cross-channel conflicts confirm",
            "action_hint": "strong_downweight_only_if_cross_channel_conflicts_confirmed",
        }
    if hard >= 0.80:
        return {
            "risk_tier": "hard_counter_yellow_review_0.80_0.95",
            "required_confirmation": [
                "target_specific_confirmation_required",
                "do_not_zero_without_news_financial_peer_bookskill_or_chip_confirmation",
            ],
            "known_false_veto_risk": "high_false_veto_risk_soft_gap_and_reversal_samples",
            "calibration_policy": "yellow review tier; probability alone is not a veto",
            "action_hint": "do_not_zero_without_cross_channel_confirmation",
        }
    if soft >= max(hard, positive) and soft >= 0.25:
        return {
            "risk_tier": "soft_gap_dominant_low_hard",
            "required_confirmation": ["treat_missingness_as_uncertainty_not_direction"],
            "known_false_veto_risk": "high_false_veto_risk_if_missingness_is_treated_as_negative",
            "calibration_policy": "soft-gap dominant; no hard downweight from classifier alone",
            "action_hint": "observe_or_hold_until_confirmation",
        }
    return {
        "risk_tier": "low_hard_counter_probability",
        "required_confirmation": ["do_not_use_as_primary_counterevidence"],
        "known_false_veto_risk": "high_false_veto_risk_if_used_as_veto",
        "calibration_policy": "low hard-counter probability; classifier is diagnostic only",
        "action_hint": "do_not_downweight_from_hard_counter_alone",
    }


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _optional_unit_num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return max(0.0, min(1.0, number))


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and not pd.isna(number):
        return number != 0.0
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _build_base_packs(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    blocks: list[str],
    quant_tool_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    block_order = list(TIME_BLOCKS)
    memory_context = load_compact_memory_context(ROOT)
    packs: list[dict[str, Any]] = []
    panel_size = max(1, int(args.limit_per_mode))
    panel_count = max(1, int(args.panel_count))
    for block in blocks:
        step = block_order.index(block)
        train_blocks = block_order[:step]
        block_packs = build_dual_mode_evidence_packs(
            frame,
            limit_per_mode=panel_size * panel_count,
            agent_policy_version=args.agent_policy_version,
            step=step,
            train_blocks=train_blocks,
            valid_block=block,
            memory_context=memory_context,
            portfolio_preset=args.portfolio_preset,
            portfolio_date_gate=args.portfolio_date_gate,
            portfolio_row_gate=args.portfolio_row_gate,
            decision_frequency=args.decision_frequency,
        )
        selected = assign_sample_panels(block_packs, panel_size=panel_size, panel_count=panel_count)
        for pack in selected:
            _attach_quant_tool_context(pack, quant_tool_summaries, max_items=args.quant_tool_max_items)
        packs.extend(selected)
    return packs


def _build_sample_plan_base_packs(
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    quant_tool_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plan = _load_safe_sample_plan(args.sample_plan)
    if plan.empty:
        return []
    memory_context = load_compact_memory_context(ROOT)
    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    source["code"] = source["code"].astype(str).str.zfill(6)
    if _is_quant_tool_ranker_preset(args.portfolio_preset):
        ranker = _portfolio_ranker_details(
            source,
            preset=args.portfolio_preset,
            valid_block="sample_plan",
            decision_frequency=args.decision_frequency,
        )
        source["quant_tool_summaries"] = ranker["quant_tool_summaries"]
    block_order = list(TIME_BLOCKS)
    packs: list[dict[str, Any]] = []
    for index, plan_row in plan.iterrows():
        date = str(plan_row["date"])
        code = str(plan_row["code"]).zfill(6)
        task_mode = str(plan_row.get("task_mode") or "portfolio_pool")
        valid_block = str(plan_row.get("valid_block") or _block_for_date(date) or "")
        if valid_block not in TIME_BLOCKS:
            continue
        matches = source[source["date"].astype(str).eq(date) & source["code"].astype(str).eq(code)]
        if matches.empty:
            continue
        row = matches.iloc[0].copy()
        if task_mode == "single_stock" and "quant_tool_summaries" in row.index:
            row = row.drop(labels=["quant_tool_summaries"])
        step = block_order.index(valid_block)
        train_blocks = block_order[:step]
        pack = build_evidence_pack(
            row,
            agent_policy_version=args.agent_policy_version,
            step=step,
            train_blocks=train_blocks,
            valid_block=valid_block,
            task_mode=task_mode,
            variant="deepseek_agent",
            python_candidate=(
                f"sample_plan_{task_mode}:{args.portfolio_preset}:{args.decision_frequency}"
                if task_mode == "portfolio_pool"
                else "sample_plan_single_stock:single_stock_risk_watch"
            ),
            memory_context=memory_context,
        )
        pack["sample_panel_id"] = str(plan_row.get("sample_panel_id") or "sample_plan")
        pack["sample_rank_in_panel"] = int(plan_row.get("sample_rank_in_panel") or index + 1)
        pack["sampler_context"] = str(plan_row.get("sampler_context") or plan_row.get("stratum") or "hard_counter_stratified_sample")
        operation_context = _sample_plan_operation_context(plan_row)
        if operation_context:
            pack["operation_plan_context"] = operation_context
        _attach_quant_tool_context(pack, quant_tool_summaries, max_items=args.quant_tool_max_items)
        packs.append(pack)
    return packs


def _sample_plan_operation_context(plan_row: pd.Series) -> dict[str, Any]:
    action = str(plan_row.get("operation_action") or plan_row.get("operation_action_cn") or "").strip()
    action_cn = str(plan_row.get("operation_action_cn") or "").strip()
    user_action, default_target = _operation_action_defaults(action, action_cn)
    target = _safe_plan_float(plan_row.get("local_target_position", plan_row.get("target_position")))
    if target is None:
        target = default_target
    reason = str(plan_row.get("local_reason_code") or plan_row.get("operation_reason_code") or "").strip()
    if not action and target is None and not reason:
        return {}
    return {
        "tool_id": "local_user_operation_plan_context_v1",
        "operation_action": action,
        "user_operation_suggestion": user_action,
        "target_position": target,
        "reason_code": reason or action,
        "decision_frequency": str(plan_row.get("decision_frequency") or plan_row.get("frequency") or "").strip(),
        "period": str(plan_row.get("period") or plan_row.get("valid_block") or "").strip(),
        "local_validation_status": (
            "yellow_small_entry_candidate_for_ds_confirmation"
            if action == "small_buy_hold"
            else "yellow_action_label_entry_candidate_for_ds_confirmation"
            if action == "buy_add" and reason == "p0_action_label_scorer_v1"
            else "local_operation_candidate_for_ds_audit"
        ),
        "soft_gap_policy": (
            "news_missing、financial_no_event_in_window、peer_weak、bookskill_observe_only属于软缺口；"
            "soft gap只能压低仓位和置信度，不能单独把small_buy_hold归零。"
            if action == "small_buy_hold"
            else "action-label强候选中，news_missing、financial_no_event_in_window、peer_weak、chip_overhang、bookskill_observe_only属于软缺口；"
            "soft gap可以把buy/add降为小仓试探或持有复核，但不能在没有明确硬反证时直接归零。"
            if action == "buy_add" and reason == "p0_action_label_scorer_v1"
            else ""
        ),
        "hard_counter_policy": (
            "只有明确负面新闻/监管债务停产、财报质量风险或负惊喜、极端过热、筹码强上压、"
            "同行显著走弱且目标持续落后、或RAG相似失败等硬反证，才覆盖为等待不买/卖出。"
            if action == "small_buy_hold"
            else "只有明确负面新闻或监管/债务/停产、财报质量风险或负惊喜、新闻warning高且无对冲证据、"
            "极端过热放量、筹码强上压叠加当前价格无法修复、同行持续显著走弱且目标没有反转证据时，才把action-label强候选归零。"
            if action == "buy_add" and reason == "p0_action_label_scorer_v1"
            else ""
        ),
        "default_position_floor_if_no_hard_counter": 0.10 if action in {"small_buy_hold", "buy_add"} else None,
        "default_position_ceiling": 0.35 if action == "small_buy_hold" else (target if action == "buy_add" else None),
        "agent_instruction": (
            "这是本地确定性工作流给出的用户操作草案；DeepSeek需要审计并决定承接、降级或覆盖。"
            "若operation_action=small_buy_hold，含义是小仓试探/继续持有，不是模糊观察；"
            "若operation_action=buy_add且reason_code=p0_action_label_scorer_v1，含义是强entry候选；"
            "只有硬反证才可归零，软缺口应优先降为低仓位试探或持有复核。"
            "若当前证据没有硬反证，优先保留低仓位动作。若覆盖，必须写明证据原因和替代动作。"
        ),
        "forbidden_use": "not_future_label_not_mandatory_copy",
        "research_only": True,
    }


def _operation_action_defaults(action: str, action_cn: str) -> tuple[str, float | None]:
    text = f"{action} {action_cn}".strip()
    if "small_buy_hold" in text or "小仓" in text:
        return "试探买入/持有", 0.25
    if "buy_add" in text or "买入" in text or "加仓" in text:
        return "试探买入/加仓", 0.5
    if "reduce_review" in text or "复核" in text:
        return "减仓/卖出复核", 0.15
    if "reduce_sell" in text or "卖出" in text or "回避" in text:
        return "卖出/不买", 0.0
    if action == "wait" or "等待" in text or "不操作" in text:
        return "等待不买", 0.0
    return action_cn or action, None


def _safe_plan_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return max(0.0, min(1.0, round(float(number), 4)))


def _load_safe_sample_plan(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing sample plan: {source}")
    frame = pd.read_csv(source, dtype={"code": str}, low_memory=False)
    if frame.empty:
        return frame
    if "date" not in frame or "code" not in frame:
        raise ValueError("sample plan must contain date and code columns")
    forbidden = {"return_5d", "return_10d", "return_20d", "future_return_5d", "future_return_10d", "future_return_20d", "gt_status", "gt_pass", "rule_outcome_label", "pool_excess_20d"}
    frame = frame.drop(columns=[col for col in forbidden if col in frame.columns])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "task_mode" not in frame:
        frame["task_mode"] = "portfolio_pool"
    if "valid_block" not in frame:
        frame["valid_block"] = frame["date"].map(_block_for_date)
    return frame.dropna(subset=["date", "code"]).reset_index(drop=True)


def _block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def _attach_quant_tool_context(pack: dict[str, Any], summaries: list[dict[str, Any]], *, max_items: int) -> None:
    existing = pack.get("quant_tool_summaries")
    if isinstance(existing, dict):
        existing_rows = [existing]
    elif isinstance(existing, list):
        existing_rows = [item for item in existing if isinstance(item, dict)]
    else:
        existing_rows = []
    selected = select_quant_tool_summaries(
        [*existing_rows, *summaries],
        task_mode=str(pack.get("task_mode") or ""),
        max_items=max_items,
    )
    pack["quant_tool_summaries"] = selected
    pack["quant_tool_signal_summary"] = quant_tool_summary_text(selected)
    pack["quant_tool_requirement"] = (
        "量化工具是训练/验证后的辅助层；若usable_in_agent_default=false或promotion_status未通过，"
        "只能作为灰色参考或反证，不能单独提高研究分级或研究暴露。"
        "若存在usable_in_agent_default=true的default/accepted工具，新闻空窗、财报近窗口无事件、"
        "BookSkill需grounding等属于软缺口，应优先partially_adopted并保持低权重观察；"
        "只有明确负面新闻/财报风险/同行显著弱/过热高波动/筹码强上压/RAG失败等硬反证覆盖工具时，"
        "才写not_adopted_counter_evidence。"
    )


def assign_sample_panels(packs: list[dict[str, Any]], *, panel_size: int, panel_count: int) -> list[dict[str, Any]]:
    if not packs:
        return []
    panel_size = max(1, int(panel_size))
    panel_count = max(1, int(panel_count))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for pack in packs:
        key = (pack.get("step"), pack.get("valid_block"), pack.get("task_mode"))
        grouped.setdefault(key, []).append(pack)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(part) for part in item)):
        for index, pack in enumerate(grouped[key]):
            panel_index = index // panel_size
            if panel_index >= panel_count:
                continue
            row = deepcopy(pack)
            row["sample_panel_id"] = f"panel_{panel_index + 1:02d}"
            row["sample_rank_in_panel"] = int(index % panel_size + 1)
            rows.append(row)
    return rows


def write_questionnaire_sample_plan(packs: list[dict[str, Any]], path: Path) -> None:
    rows = []
    seen: set[tuple[str, str]] = set()
    for pack in packs:
        date = str(pack.get("decision_date") or "")
        code = str(pack.get("code") or "").zfill(6)
        if not date or not code or (date, code) in seen:
            continue
        seen.add((date, code))
        rows.append(
            {
                "date": date,
                "code": code,
                "name": pack.get("name"),
                "valid_block": pack.get("valid_block"),
                "task_mode": pack.get("task_mode"),
                "sample_panel_id": pack.get("sample_panel_id", "panel_01"),
                "sample_rank_in_panel": pack.get("sample_rank_in_panel", 1),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _load_questionnaire_scores() -> pd.DataFrame:
    files = sorted(OUTPUT.glob("news_questionnaire_flash*_scores.csv"))
    frames = []
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if frame.empty:
            continue
        frame["source_score_file"] = path.name
        frame["_source_mtime"] = path.stat().st_mtime
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["decision_date", "code"])
    data = pd.concat(frames, ignore_index=True)
    keep = _questionnaire_keep_columns(data)
    data = data[keep].copy()
    data["_source_rank"] = data["source_score_file"].map(_source_rank)
    if "_source_mtime" not in data:
        data["_source_mtime"] = 0
    data = data.sort_values(["decision_date", "code", "_source_rank", "_source_mtime", "source_score_file"]).drop_duplicates(["decision_date", "code"], keep="last")
    return data.drop(columns=["_source_rank", "_source_mtime"])


def _questionnaire_keep_columns(data: pd.DataFrame) -> list[str]:
    config = load_news_questionnaire(ROOT / "config" / "news_deepseek_questionnaire.yaml")
    question_fields = questionnaire_output_fields(config)
    candidates = [
        "decision_date",
        "code",
        "questionnaire_version",
        "mainline_summary",
        "missing_or_conflict_notes",
        "source_score_file",
        "_source_mtime",
        *DERIVED_QUESTIONNAIRE_FIELDS,
        *question_fields,
    ]
    return [field for field in candidates if field in data]


def _merge_questionnaire_scores(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    if scores.empty:
        return data
    mapped = scores.copy()
    mapped = mapped.rename(
        columns={
            "decision_date": "date",
            "questionnaire_version": "news_semantic_questionnaire_version",
            "mainline_summary": "ds_news_mainline_summary",
            "missing_or_conflict_notes": "ds_news_missing_or_conflict_notes",
        }
    )
    safe_cols = [
        col
        for col in mapped.columns
        if col in {"date", "code", "news_semantic_questionnaire_version", "ds_news_mainline_summary", "ds_news_missing_or_conflict_notes", "source_score_file"}
        or col.startswith("ds_news_")
    ]
    mapped = mapped[safe_cols].copy()
    for field in safe_cols:
        if field.startswith("ds_news_") and field not in {"ds_news_mainline_summary", "ds_news_missing_or_conflict_notes"}:
            mapped[field] = pd.to_numeric(mapped[field], errors="coerce")
    return data.merge(mapped, on=["date", "code"], how="left", suffixes=("", "_questionnaire"))


def _source_rank(path_name: str) -> int:
    if "retry" in path_name:
        return 3
    if "spread_panel_v2" in path_name:
        return 2
    if "spread_panel" in path_name:
        return 1
    return 0


def _write_result_tables(
    metrics_path: Path,
    step_metrics_path: Path,
    report_path: Path,
    cards: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    usage: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    base_count: int,
    pack_count: int,
    called: bool,
    reused: bool,
    questionnaire_plan_path: Path,
) -> None:
    metrics = variant_metrics(cards, invalid, frame, portfolio_preset=args.portfolio_preset)
    step_metrics = variant_step_metrics(cards, invalid, frame, portfolio_preset=args.portfolio_preset)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
    _write_summary(report_path, args=args, base_count=base_count, pack_count=pack_count, called=called, reused=reused, metrics=metrics, step_metrics=step_metrics, usage=usage, invalid_count=len(invalid), questionnaire_plan_path=questionnaire_plan_path, cards=cards)


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    base_count: int,
    pack_count: int,
    called: bool,
    reused: bool,
    metrics: pd.DataFrame,
    step_metrics: pd.DataFrame,
    usage: pd.DataFrame,
    invalid_count: int,
    questionnaire_plan_path: Path,
    cards: list[dict[str, Any]],
) -> None:
    token_total = int(pd.to_numeric(usage.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not usage.empty else 0
    adoption = quant_tool_adoption_summary(cards)
    lines = [
        "# Full Channel Ablation Small Round",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Run",
        "",
        f"- agent_policy_version: `{args.agent_policy_version}`",
        f"- model: `{args.model}`",
        f"- called_deepseek: `{called}`",
        f"- reused_decision_ledger: `{reused}`",
        f"- base_packs: `{base_count}`",
        f"- ablation_packs: `{pack_count}`",
        f"- panel_count: `{args.panel_count}`",
        f"- invalid_outputs: `{invalid_count}`",
        f"- total_tokens: `{token_total}`",
        f"- variants: `{args.variants}`",
        f"- valid_blocks: `{args.valid_blocks}`",
        f"- quant_tool_rule_outcomes: `{args.quant_tool_rule_outcomes}`",
        f"- quant_tool_max_items: `{args.quant_tool_max_items}`",
        f"- portfolio_quant_adoption_guard: `{getattr(args, 'portfolio_quant_adoption_guard', None)}`",
        f"- questionnaire_sample_plan: `{questionnaire_plan_path}`",
        "",
        "## Metrics By Variant",
        "",
        _table(metrics),
        "",
        "## Step Metrics",
        "",
        _table(step_metrics),
        "",
        "## Quant Tool Adoption",
        "",
        _table(adoption),
        "",
        "## Notes",
        "",
        "- `full_agent` 是全通道候选；其余 variant 是灰色对照或组件消融。",
        "- `full_agent_with_quant_tools/full_agent_without_quant_tools/quant_tool_summary_only` 用于检查训练型量化工具是否能给Agent带来可复现增益。",
        "- 当前量化工具若标记为 `usable_in_agent_default=false` 或 latest block failed，只能作为灰色参考或反证。",
        "- `python_only` 只保留 Python gate 和基础字段，用于判断 Agent 多通道是否真的增加价值。",
        "- 扩大前必须先跑 leakage audit 和 channel coverage audit；coverage 只证明输入完整，不证明收益贡献。",
        "- 若某个 variant 的 exposure_cards=0，只能解释为防守/研究权重路径，不能宣称选股能力。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quant_tool_adoption_summary(cards: list[dict[str, Any]]) -> pd.DataFrame:
    if not cards:
        return pd.DataFrame()
    frame = pd.DataFrame(cards)
    required = ["variant", "task_mode", "accepted_quant_tool_ids", "quant_tool_adoption_decision", "quant_tool_override_reasons"]
    for col in required:
        if col not in frame:
            frame[col] = "missing"
    rows = []
    for values, group in frame.groupby(["variant", "task_mode", "quant_tool_adoption_decision", "quant_tool_override_reasons"], dropna=False, sort=True):
        accepted_non_none = group["accepted_quant_tool_ids"].astype(str).ne("none").sum()
        row = {
            "variant": values[0],
            "task_mode": values[1],
            "quant_tool_adoption_decision": values[2],
            "quant_tool_override_reasons": values[3],
            "decision_cards": int(len(group)),
            "accepted_tool_cards": int(accepted_non_none),
            "research_only": True,
            "not_investment_instruction": True,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_blocks(raw: str) -> list[str]:
    blocks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [block for block in blocks if block not in TIME_BLOCKS]
    if unknown:
        raise ValueError(f"unknown valid blocks: {unknown}")
    return blocks


def _parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; allowed={ALLOWED_VARIANTS}")
    return variants


def _parse_task_modes(raw: str) -> set[str] | None:
    task_modes = {_canonical_task_mode(item) for item in str(raw or "").split(",") if item.strip()}
    task_modes.discard("")
    if not task_modes or task_modes == {"all"}:
        return None
    unknown = task_modes - {"portfolio_pool", "single_stock"}
    if unknown:
        raise ValueError(f"unknown task modes for nonprice overlay: {sorted(unknown)}")
    return task_modes


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "full_channel_ablation_small_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pack_key(pack: dict[str, Any]) -> tuple[Any, ...]:
    return (
        pack.get("agent_policy_version"),
        pack.get("variant"),
        pack.get("step"),
        pack.get("valid_block"),
        pack.get("decision_date"),
        pack.get("code"),
        pack.get("task_mode"),
        pack.get("sample_panel_id") or "panel_01",
    )


def _card_key(card: dict[str, Any]) -> tuple[Any, ...]:
    return (
        card.get("agent_policy_version"),
        card.get("variant"),
        card.get("step"),
        card.get("valid_block"),
        card.get("decision_date"),
        card.get("code"),
        card.get("task_mode"),
        card.get("sample_panel_id") or "panel_01",
    )


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for card in cards:
        seen[_card_key(card)] = card
    return [seen[key] for key in sorted(seen, key=lambda item: tuple(str(part) for part in item))]


def _apply_posthoc_guardrails(cards: list[dict[str, Any]], packs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_pack_key(pack): pack for pack in packs}
    updated: list[dict[str, Any]] = []
    for card in cards:
        row = deepcopy(card)
        pack = by_key.get(_card_key(row))
        if pack is not None:
            row = card_from_evidence_pack(pack, row)
        else:
            apply_decision_guardrails(row, {})
        updated.append(row)
    return updated


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
