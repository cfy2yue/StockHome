from __future__ import annotations

import argparse
import csv
import json
import math
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
from src.agent_training.evidence_pack import KLINE_FEATURE_FIELDS, apply_decision_guardrails, build_evidence_pack
from src.agent_training.case_memory_retriever import (
    format_applicable_retrieved_cases,
    format_retrieved_cases,
    retrieve_applicable_cases,
    retrieve_cases,
)
from src.agent_training.memory_context import load_compact_memory_context
from src.agent_training.dual_mode_round import (
    BANK_ANNUAL_RATE,
    TIME_BLOCKS,
    build_dual_mode_evidence_packs,
    load_ground_truth,
)
from src.agent_training.preflight import run_preflight, write_preflight_reports
from src.world_model.news_questionnaire import load_news_questionnaire, questionnaire_output_fields


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_VARIANTS = [
    "no_news",
    "keyword_only",
    "questionnaire_only",
    "keyword_plus_questionnaire",
    "keyword_plus_questionnaire_guarded",
    "risk_only_questionnaire",
]
EXPERIMENTAL_VARIANTS = [
    "uncertainty_only_questionnaire",
    "quality_only_questionnaire",
    "semantic_risk_only_questionnaire",
    "risk_uncertainty_questionnaire",
]
FINANCIAL_REPORT_VARIANTS = [
    "no_financial_report_channel",
    "financial_report_only",
    "news_plus_financial_report",
    "news_plus_financial_report_guarded",
]
KLINE_VARIANTS = [
    "no_kline",
    "kline_weak_prompt",
]
KNOWN_VARIANTS = DEFAULT_VARIANTS + EXPERIMENTAL_VARIANTS + FINANCIAL_REPORT_VARIANTS + KLINE_VARIANTS
KNOWN_CASE_MEMORY_MODES = ["no_rag", "memory_compact_only", "retrieved_cases_v1", "retrieved_cases_v2_applicable"]
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
ROUTINE_POSITIVE_CAP_RULE_ID = "news_questionnaire_routine_announcement_positive_cap_v1"
ROUTINE_POSITIVE_CAP_OPPORTUNITY_THRESHOLD = 0.7
ROUTINE_POSITIVE_CAP_MAINLINE_THRESHOLD = 0.5
ROUTINE_POSITIVE_CAP_RELEVANCE_THRESHOLD = 0.5
ROUTINE_POSITIVE_CAP_REPETITION_THRESHOLD = 0.6
ROUTINE_POSITIVE_CAP_OPPORTUNITY_SCORE = 0.2
ROUTINE_POSITIVE_CAP_NET_SCORE = 0.0
RISK_ONLY_QUESTION_FIELDS = {
    "ds_news_source_coverage",
    "ds_news_timestamp_confidence",
    "ds_news_self_regulatory_legal",
    "ds_news_self_capital_financing",
    "ds_news_peer_risk_diffusion",
    "ds_news_policy_headwind",
    "ds_news_region_risk",
    "ds_news_conflict_intensity",
    "ds_news_consensus_crowding",
    "ds_news_repetition_lag",
    "ds_news_decision_relevance",
}
RISK_ONLY_DERIVED_FIELDS = {
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_risk_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_missing_or_conflict_notes",
}
UNCERTAINTY_ONLY_QUESTION_FIELDS = {
    "ds_news_mainline_clarity",
    "ds_news_source_coverage",
    "ds_news_timestamp_confidence",
    "ds_news_peer_silent_gap",
    "ds_news_conflict_intensity",
    "ds_news_consensus_crowding",
    "ds_news_novelty",
    "ds_news_repetition_lag",
    "ds_news_decision_relevance",
}
UNCERTAINTY_ONLY_DERIVED_FIELDS = {
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_missing_or_conflict_notes",
}
QUALITY_ONLY_QUESTION_FIELDS = {
    "ds_news_mainline_clarity",
    "ds_news_source_coverage",
    "ds_news_official_support",
    "ds_news_timestamp_confidence",
    "ds_news_cross_stock_confirmation",
    "ds_news_decision_relevance",
}
QUALITY_ONLY_DERIVED_FIELDS = {
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_quality_score",
    "ds_news_missing_or_conflict_notes",
}
SEMANTIC_RISK_ONLY_QUESTION_FIELDS = {
    "ds_news_self_capital_financing",
    "ds_news_self_regulatory_legal",
    "ds_news_peer_risk_diffusion",
    "ds_news_policy_headwind",
    "ds_news_region_risk",
    "ds_news_conflict_intensity",
    "ds_news_consensus_crowding",
    "ds_news_repetition_lag",
    "ds_news_decision_relevance",
}
SEMANTIC_RISK_ONLY_DERIVED_FIELDS = {
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_risk_score",
    "ds_news_uncertainty_score",
    "ds_news_missing_or_conflict_notes",
}
RISK_UNCERTAINTY_QUESTION_FIELDS = (
    SEMANTIC_RISK_ONLY_QUESTION_FIELDS
    | UNCERTAINTY_ONLY_QUESTION_FIELDS
    | {
        "ds_news_source_coverage",
        "ds_news_timestamp_confidence",
    }
)
RISK_UNCERTAINTY_DERIVED_FIELDS = {
    "news_semantic_questionnaire_version",
    "ds_news_mainline_summary",
    "ds_news_risk_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_missing_or_conflict_notes",
}
KEYWORD_OPPORTUNITY_FIELDS = {
    "news_opportunity_score",
    "policy_background_score",
    "region_background_score",
    "official_confirmation_score",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-agent DeepSeek news-channel ablation on questionnaire-matched rows.")
    parser.add_argument("--blocks", default="H2025_1,H2026_1")
    parser.add_argument("--limit-per-mode", type=int, default=2)
    parser.add_argument("--portfolio-preset", default="peer_confirmed_pullback")
    parser.add_argument("--portfolio-date-gate", default="pool_pullback")
    parser.add_argument("--portfolio-row-gate", default="news_risk_low")
    parser.add_argument("--decision-frequency", default="every_2_weeks")
    parser.add_argument("--agent-policy-version", default="deepseek_news_ablation_v1")
    parser.add_argument("--output-prefix", default="deepseek_news_ablation_round")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--matched-questionnaire-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--matched-financial-report-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-plan", default="", help="Optional CSV with date/code rows to use exactly as the base sample before ablation.")
    parser.add_argument("--sample-plan-max-rows", type=int, default=0, help="Optional cap for sample-plan rows before expanding task modes and variants.")
    parser.add_argument("--sample-plan-per-rule", type=int, default=0, help="Optional per-candidate_rule cap before sample-plan-max-rows.")
    parser.add_argument("--sample-plan-rules", default="", help="Optional comma-separated candidate_rule filter for --sample-plan.")
    parser.add_argument("--sample-plan-task-modes", default="both", choices=["both", "portfolio_pool", "single_stock"])
    parser.add_argument(
        "--case-memory-modes",
        default="memory_compact_only",
        help="Comma-separated: no_rag,memory_compact_only,retrieved_cases_v1,retrieved_cases_v2_applicable",
    )
    parser.add_argument("--case-memory-top-k", type=int, default=3)
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--reuse-decision-ledger", action="store_true")
    parser.add_argument("--resume-missing", action="store_true", help="When calling DeepSeek, keep existing valid cards and call only missing pack keys.")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--user-id", default="stock_agent_news_ablation")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)
    preflight = run_preflight(ROOT)
    write_preflight_reports(preflight, OUTPUT)
    if not preflight["ok"]:
        raise SystemExit("preflight failed; see reports/date_generalization/preflight_check.md")

    source_frame = load_ground_truth(GT_SOURCES)
    questionnaire = _load_questionnaire_scores()
    enriched = _merge_questionnaire_scores(source_frame, questionnaire)
    blocks = _parse_blocks(args.blocks)
    variants = _parse_variants(args.variants)
    case_memory_modes = _parse_case_memory_modes(args.case_memory_modes)
    if args.sample_plan:
        base_packs = _build_sample_plan_packs(enriched, args=args)
        if base_packs:
            blocks = sorted({str(pack.get("valid_block")) for pack in base_packs}, key=_block_sort_key)
    else:
        base_packs = _build_base_packs(enriched, args=args, blocks=blocks)
    ablation_packs = _apply_case_memory_modes(_ablation_packs(base_packs, variants), case_memory_modes, top_k=args.case_memory_top_k)

    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    metrics_path = OUTPUT / f"{prefix}_metrics.csv"
    step_metrics_path = OUTPUT / f"{prefix}_step_metrics.csv"
    action_diagnostics_path = OUTPUT / f"{prefix}_action_diagnostics.csv"
    summary_path = OUTPUT / f"{prefix}_summary.md"
    write_jsonl(str(evidence_path), ablation_packs)

    if args.reuse_decision_ledger:
        cards = _read_jsonl(decision_path)
        cards = _apply_posthoc_guardrails(cards, ablation_packs)
        write_jsonl(str(decision_path), cards)
        invalid = _read_jsonl(invalid_path)
        usage = pd.read_csv(usage_path) if usage_path.exists() else pd.DataFrame()
        metrics = _metrics(cards, invalid, enriched)
        step_metrics = _step_metrics(cards, invalid, enriched)
        action_diagnostics = _action_diagnostics(cards, invalid)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
        action_diagnostics.to_csv(action_diagnostics_path, index=False, encoding="utf-8-sig")
        _write_summary(summary_path, args=args, blocks=blocks, variants=variants, called_deepseek=True, reused=True, base_packs=base_packs, ablation_packs=ablation_packs, metrics=metrics, step_metrics=step_metrics, action_diagnostics=action_diagnostics, usage=usage, invalid_count=len(invalid))
        _update_usage_index(prefix, usage_path, usage)
        print("A股研究Agent")
        print(f"reused_decision_ledger=True base_packs={len(base_packs)} ablation_packs={len(ablation_packs)} cards={len(cards)} invalid={len(invalid)}")
        print(f"wrote: {summary_path}")
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
        cards = _dedupe_cards([*existing_cards, *result.ok_cards])
        write_jsonl(str(decision_path), cards)
        write_jsonl(str(invalid_path), result.invalid_outputs)
        new_usage = pd.DataFrame(result.usage_rows)
        previous_usage = pd.read_csv(usage_path) if args.resume_missing and usage_path.exists() else pd.DataFrame()
        usage = pd.concat([previous_usage, new_usage], ignore_index=True) if not previous_usage.empty else new_usage
        usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
        metrics = _metrics(cards, result.invalid_outputs, enriched)
        step_metrics = _step_metrics(cards, result.invalid_outputs, enriched)
        action_diagnostics = _action_diagnostics(cards, result.invalid_outputs)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
        action_diagnostics.to_csv(action_diagnostics_path, index=False, encoding="utf-8-sig")
        _write_summary(summary_path, args=args, blocks=blocks, variants=variants, called_deepseek=True, reused=False, base_packs=base_packs, ablation_packs=ablation_packs, metrics=metrics, step_metrics=step_metrics, action_diagnostics=action_diagnostics, usage=usage, invalid_count=len(result.invalid_outputs))
        _update_usage_index(prefix, usage_path, usage)
        print("A股研究Agent")
        print(f"called_deepseek=True base_packs={len(base_packs)} ablation_packs={len(ablation_packs)} called_packs={len(packs_to_call)} ok_cards={len(cards)} invalid={len(result.invalid_outputs)}")
        print(f"wrote: {summary_path}")
        return

    write_jsonl(str(decision_path), [])
    write_jsonl(str(invalid_path), [])
    usage = pd.DataFrame(columns=["model", "status", "total_tokens", "requested_max_workers", "effective_workers", "model_concurrency_limit"])
    usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
    planned = _planned_metrics(ablation_packs)
    planned.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(action_diagnostics_path, index=False, encoding="utf-8-sig")
    _write_summary(summary_path, args=args, blocks=blocks, variants=variants, called_deepseek=False, reused=False, base_packs=base_packs, ablation_packs=ablation_packs, metrics=pd.DataFrame(), step_metrics=planned, action_diagnostics=pd.DataFrame(), usage=usage, invalid_count=0)
    print("A股研究Agent")
    print(f"called_deepseek=False base_packs={len(base_packs)} ablation_packs={len(ablation_packs)}")
    print(f"wrote: {summary_path}")


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
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["decision_date", "code"])
    data = pd.concat(frames, ignore_index=True)
    keep = _questionnaire_keep_columns(data)
    data = data[keep].copy()
    data["_source_rank"] = data["source_score_file"].map(_source_rank)
    data = data.sort_values(["decision_date", "code", "_source_rank"]).drop_duplicates(["decision_date", "code"], keep="last")
    data = data.drop(columns=["_source_rank"])
    return data


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
    rename = {
        "decision_date": "date",
        "questionnaire_version": "news_semantic_questionnaire_version",
        "mainline_summary": "ds_news_mainline_summary",
        "missing_or_conflict_notes": "ds_news_missing_or_conflict_notes",
    }
    mapped = mapped.rename(columns=rename)
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
    merged = data.merge(mapped, on=["date", "code"], how="left", suffixes=("", "_questionnaire"))
    return merged


def _build_base_packs(frame: pd.DataFrame, *, args: argparse.Namespace, blocks: list[str]) -> list[dict[str, Any]]:
    block_order = list(TIME_BLOCKS)
    packs: list[dict[str, Any]] = []
    for block in blocks:
        if block not in TIME_BLOCKS:
            raise ValueError(f"unknown time block: {block}")
        scoped_frame = frame.copy()
        if args.matched_questionnaire_only:
            block_dates = _block_selector(scoped_frame, block)
            has_questionnaire = scoped_frame.get("ds_news_risk_score", pd.Series(index=scoped_frame.index, dtype=float)).notna()
            scoped_frame = scoped_frame[(~block_dates) | has_questionnaire].copy()
        if args.matched_financial_report_only:
            block_dates = _block_selector(scoped_frame, block)
            has_financial_report = _financial_report_matched_selector(scoped_frame)
            scoped_frame = scoped_frame[(~block_dates) | has_financial_report].copy()
        step = block_order.index(block)
        train_blocks = block_order[: block_order.index(block)]
        packs.extend(
            build_dual_mode_evidence_packs(
                scoped_frame,
                limit_per_mode=args.limit_per_mode,
                agent_policy_version=args.agent_policy_version,
                step=step,
                train_blocks=train_blocks,
                valid_block=block,
                memory_context=_load_memory_context(),
                portfolio_preset=args.portfolio_preset,
                portfolio_date_gate=args.portfolio_date_gate,
                portfolio_row_gate=args.portfolio_row_gate,
                decision_frequency=args.decision_frequency,
            )
        )
    return packs


def _build_sample_plan_packs(frame: pd.DataFrame, *, args: argparse.Namespace) -> list[dict[str, Any]]:
    plan_path = Path(args.sample_plan)
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    plan = pd.read_csv(plan_path, dtype={"code": str}, low_memory=False)
    future_cols = sorted(set(plan.columns) & {"return_5d", "return_10d", "return_20d", "future_return_5d", "future_return_10d", "future_return_20d", "gt_status"})
    if future_cols:
        raise ValueError(f"sample plan contains future/result fields: {future_cols}")
    required = {"date", "code"}
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"sample plan missing required columns: {missing}")
    plan = plan.copy()
    plan["code"] = plan["code"].astype(str).str.zfill(6)
    plan["date"] = pd.to_datetime(plan["date"], errors="coerce").dt.date.astype(str)
    if args.sample_plan_rules:
        allowed_rules = {item.strip() for item in str(args.sample_plan_rules).split(",") if item.strip()}
        if "candidate_rule" not in plan:
            raise ValueError("--sample-plan-rules requires candidate_rule column")
        plan = plan[plan["candidate_rule"].astype(str).isin(allowed_rules)].copy()
    if args.sample_plan_per_rule and args.sample_plan_per_rule > 0:
        if "candidate_rule" not in plan:
            raise ValueError("--sample-plan-per-rule requires candidate_rule column")
        plan = (
            plan.groupby(plan["candidate_rule"].astype(str), sort=False, group_keys=False)
            .head(args.sample_plan_per_rule)
            .copy()
        )
    if args.sample_plan_max_rows and args.sample_plan_max_rows > 0:
        plan = plan.head(args.sample_plan_max_rows).copy()

    source = frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    plan_cols = [
        col
        for col in [
            "date",
            "code",
            "candidate_rule",
            "reason_to_test",
            "sample_stock_concentration_note",
            *KLINE_FEATURE_FIELDS,
        ]
        if col in plan
    ]
    merged = source.merge(plan[plan_cols], on=["date", "code"], how="inner", suffixes=("", "_sample_plan"))
    if merged.empty and not plan.empty:
        examples = plan[["date", "code"]].head(5).to_dict("records")
        raise ValueError(f"sample plan matched zero source rows; first_keys={examples}")
    merged = _order_like_plan(merged, plan)
    task_modes = ["portfolio_pool", "single_stock"] if args.sample_plan_task_modes == "both" else [args.sample_plan_task_modes]
    block_order = list(TIME_BLOCKS)
    packs: list[dict[str, Any]] = []
    memory_context = _load_memory_context()
    for _, row in merged.iterrows():
        valid_block = str(row.get("time_block") or _date_to_time_block(row.get("date")))
        if valid_block not in TIME_BLOCKS:
            valid_block = _date_to_time_block(row.get("date"))
        step = block_order.index(valid_block) if valid_block in TIME_BLOCKS else 0
        train_blocks = block_order[:step]
        candidate_rule = str(row.get("candidate_rule") or "sample_plan")
        reason = str(row.get("reason_to_test") or "")
        concentration = str(row.get("sample_stock_concentration_note") or "")
        for mode in task_modes:
            packs.append(
                build_evidence_pack(
                    row,
                    agent_policy_version=args.agent_policy_version,
                    step=step,
                    train_blocks=train_blocks,
                    valid_block=valid_block,
                    task_mode=mode,
                    variant="deepseek_agent",
                    python_candidate=(
                        f"sample_plan_rule={candidate_rule}; task_mode={mode}; "
                        f"reason={reason}; concentration={concentration}"
                    ),
                    memory_context=memory_context,
                )
            )
    return packs


def _order_like_plan(merged: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    order = plan[["date", "code"]].drop_duplicates().reset_index(drop=True).reset_index()
    order = order.rename(columns={"index": "_sample_plan_order"})
    data = merged.merge(order, on=["date", "code"], how="left")
    return data.sort_values(["_sample_plan_order", "date", "code"]).drop(columns=["_sample_plan_order"]).reset_index(drop=True)


def _date_to_time_block(value: Any) -> str:
    ts = pd.Timestamp(value)
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    raise ValueError(f"date outside known time blocks: {value}")


def _block_sort_key(block: str) -> int:
    try:
        return list(TIME_BLOCKS).index(block)
    except ValueError:
        return len(TIME_BLOCKS)


def _block_selector(frame: pd.DataFrame, block: str) -> pd.Series:
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _financial_report_matched_selector(frame: pd.DataFrame) -> pd.Series:
    status = frame.get("financial_report_join_status", pd.Series("", index=frame.index)).fillna("").astype(str)
    count = pd.to_numeric(frame.get("financial_report_event_count", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    return status.eq("event_window_matched") | count.gt(0)


def _ablation_packs(base_packs: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for base in base_packs:
        for variant in variants:
            pack = _apply_variant(base, variant)
            pack["variant"] = variant
            pack["news_ablation_variant"] = variant
            rows.append(pack)
    return rows


def _apply_case_memory_modes(packs: list[dict[str, Any]], modes: list[str], *, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in packs:
        original_memory = str(base.get("memory_context") or "none")
        for mode in modes:
            pack = deepcopy(base)
            pack["case_memory_mode"] = mode
            if mode == "no_rag":
                pack["memory_context"] = "none"
                pack["retrieved_cases_context"] = "none"
            elif mode == "memory_compact_only":
                pack["memory_context"] = original_memory
                pack["retrieved_cases_context"] = "none"
            elif mode == "retrieved_cases_v1":
                pack["memory_context"] = original_memory
                cases = retrieve_cases(ROOT, _case_memory_query(pack), top_k=top_k)
                pack["retrieved_cases_context"] = format_retrieved_cases(cases, max_chars=1400)
            elif mode == "retrieved_cases_v2_applicable":
                pack["memory_context"] = original_memory
                cases = retrieve_applicable_cases(ROOT, pack, top_k=top_k)
                pack["retrieved_cases_context"] = format_applicable_retrieved_cases(cases, max_chars=1400)
            else:
                raise ValueError(f"unknown case memory mode: {mode}")
            rows.append(pack)
    return rows


def _case_memory_query(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": pack.get("variant"),
        "task_mode": pack.get("task_mode"),
        "valid_block": pack.get("valid_block"),
        "python": pack.get("python_signal_summary"),
        "news": pack.get("news_signal_summary"),
        "financial_report": pack.get("financial_report_signal_summary"),
        "counter_evidence": pack.get("counter_evidence"),
        "data_missing_flags": pack.get("data_missing_flags"),
        "book_skill": ";".join(
            str(item.get("strategy_id", ""))
            for item in pack.get("book_skill_candidates", [])
            if isinstance(item, dict)
        ),
    }


def _apply_variant(base_pack: dict[str, Any], variant: str) -> dict[str, Any]:
    pack = deepcopy(base_pack)
    if variant == "keyword_plus_questionnaire":
        pack["news_ablation_policy"] = "keep keyword/event statistics and full DeepSeek semantic questionnaire."
    elif variant == "keyword_plus_questionnaire_guarded":
        pack["news_semantic_questionnaire"] = _routine_positive_cap_questionnaire(pack.get("news_semantic_questionnaire", {}))
        pack["news_signal_summary"] = _guarded_news_summary(pack)
        if pack.get("news_semantic_questionnaire", {}).get("ds_news_positive_capped_by_rule") is True:
            pack["news_ablation_policy"] = "positive news opportunity/net score is capped by routine/weak-mainline guard for this pack; do not use capped opportunity as positive evidence."
        else:
            pack["news_ablation_policy"] = "same as keyword_plus_questionnaire for this pack; routine/weak-mainline positive cap did not trigger."
    elif variant == "no_news":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = {}
        pack["news_signal_summary"] = "news ablation: no news fields visible"
        pack["news_ablation_policy"] = "all news keyword and questionnaire evidence hidden from DeepSeek."
    elif variant == "keyword_only":
        pack["news_semantic_questionnaire"] = {}
        pack["news_ablation_policy"] = "keyword/event statistics visible; DeepSeek semantic questionnaire hidden."
    elif variant == "questionnaire_only":
        pack["news_features"] = {}
        pack["news_ablation_policy"] = "DeepSeek semantic questionnaire visible; keyword/event statistics hidden."
    elif variant == "risk_only_questionnaire":
        pack["news_features"] = _risk_only_keyword_features(pack.get("news_features", {}))
        pack["news_semantic_questionnaire"] = _field_subset_by_allowed(
            pack.get("news_semantic_questionnaire", {}),
            set(RISK_ONLY_DERIVED_FIELDS) | RISK_ONLY_QUESTION_FIELDS,
        )
        pack["news_signal_summary"] = _risk_only_summary(pack.get("news_semantic_questionnaire", {}))
        pack["news_ablation_policy"] = "only risk, uncertainty, quality and conflict questionnaire fields visible; positive opportunity/net fields hidden."
    elif variant == "uncertainty_only_questionnaire":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = _field_subset_by_allowed(
            pack.get("news_semantic_questionnaire", {}),
            set(UNCERTAINTY_ONLY_DERIVED_FIELDS) | UNCERTAINTY_ONLY_QUESTION_FIELDS,
        )
        pack["news_signal_summary"] = _uncertainty_only_summary(pack.get("news_semantic_questionnaire", {}))
        pack["news_ablation_policy"] = "only source/timestamp/conflict/repetition/mainline uncertainty evidence visible; risk, opportunity and keyword fields hidden."
    elif variant == "quality_only_questionnaire":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = _field_subset_by_allowed(
            pack.get("news_semantic_questionnaire", {}),
            set(QUALITY_ONLY_DERIVED_FIELDS) | QUALITY_ONLY_QUESTION_FIELDS,
        )
        pack["news_signal_summary"] = _quality_only_summary(pack.get("news_semantic_questionnaire", {}))
        pack["news_ablation_policy"] = "only evidence quality, source coverage, official support and timestamp confidence visible; risk/opportunity fields hidden."
    elif variant == "semantic_risk_only_questionnaire":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = _field_subset_by_allowed(
            pack.get("news_semantic_questionnaire", {}),
            set(SEMANTIC_RISK_ONLY_DERIVED_FIELDS) | SEMANTIC_RISK_ONLY_QUESTION_FIELDS,
        )
        pack["news_signal_summary"] = _semantic_risk_only_summary(pack.get("news_semantic_questionnaire", {}))
        pack["news_ablation_policy"] = "only DeepSeek semantic risk fields visible; keyword/event statistics and positive opportunity/net fields hidden."
    elif variant == "risk_uncertainty_questionnaire":
        pack["news_features"] = _risk_only_keyword_features(pack.get("news_features", {}))
        pack["news_semantic_questionnaire"] = _field_subset_by_allowed(
            pack.get("news_semantic_questionnaire", {}),
            set(RISK_UNCERTAINTY_DERIVED_FIELDS) | RISK_UNCERTAINTY_QUESTION_FIELDS,
        )
        pack["news_signal_summary"] = _risk_uncertainty_summary(pack.get("news_semantic_questionnaire", {}))
        pack["news_ablation_policy"] = "risk keyword fields plus semantic risk/uncertainty/quality evidence visible; positive opportunity/net fields hidden."
    elif variant == "no_financial_report_channel":
        pack["financial_report_features"] = {}
        pack["financial_report_signal_summary"] = "financial report ablation: no financial report fields visible"
        pack["financial_report_ablation_policy"] = "all financial report and earnings-announcement evidence hidden from DeepSeek; ordinary news remains visible."
    elif variant == "financial_report_only":
        pack["news_features"] = {}
        pack["news_semantic_questionnaire"] = {}
        pack["news_signal_summary"] = "news ablation: only financial report channel visible"
        pack["news_ablation_policy"] = "ordinary news keyword and questionnaire fields hidden; financial report channel remains visible."
        pack["financial_report_ablation_policy"] = "only financial report / earnings-announcement evidence is visible among external event channels."
    elif variant == "news_plus_financial_report":
        pack["news_ablation_policy"] = "ordinary news keyword/questionnaire evidence visible."
        pack["financial_report_ablation_policy"] = "financial report / earnings-announcement evidence visible as separate high-confidence event channel."
    elif variant == "news_plus_financial_report_guarded":
        pack["news_ablation_policy"] = "ordinary news keyword/questionnaire evidence visible."
        pack["financial_report_guardrail"] = _financial_report_risk_to_zero_guard(pack)
        pack["financial_report_signal_summary"] = _financial_report_guarded_summary(pack)
        pack["financial_report_ablation_policy"] = (
            "financial report / earnings-announcement evidence visible; "
            "apply financial_risk_to_zero_guard_v1 when high financial quality risk or negative surprise plus overheat lacks cross-channel confirmation."
        )
    elif variant == "no_kline":
        pack["kline_features"] = {}
        pack["kline_signal_summary"] = "kline ablation: no K-line fields visible"
        pack["kline_ablation_policy"] = "all multiscale K-line and peer K-line evidence hidden from DeepSeek."
    elif variant == "kline_weak_prompt":
        pack["kline_ablation_policy"] = (
            "multiscale K-line evidence visible as weak quantitative context only; "
            "20d pullback is observe-level evidence, while 60d deep drawdown and weak peer breadth are not positive gates."
        )
    else:
        raise ValueError(f"unknown news ablation variant: {variant}")
    return pack


def _financial_report_risk_to_zero_guard(pack: dict[str, Any]) -> dict[str, Any]:
    features = pack.get("financial_report_features", {}) if isinstance(pack.get("financial_report_features"), dict) else {}
    python_text = str(pack.get("python_signal_summary") or "")
    triggered = _financial_risk_guard_triggered(pack)
    return {
        "rule_id": "financial_risk_to_zero_guard_v1",
        "status": "observe_candidate",
        "triggered_for_pack": triggered,
        "high_quality_risk_threshold": 0.6,
        "negative_surprise_threshold": -0.4,
        "overheat_prior_return_20d_threshold": 20,
        "overheat_rsi14_threshold": 70,
        "policy": "若触发且缺少Book Skill/同行/新闻恢复确认，不得增加研究暴露；优先转入0权重研究状态或信息不足不动作。",
        "financial_quality_risk_score": features.get("financial_quality_risk_score"),
        "financial_surprise_score": features.get("financial_surprise_score"),
        "python_signal_summary": python_text[:240],
    }


def _financial_report_guarded_summary(pack: dict[str, Any]) -> str:
    base = str(pack.get("financial_report_signal_summary") or "financial_report_channel_not_collected")
    guard = pack.get("financial_report_guardrail", {})
    if guard.get("triggered_for_pack") is True:
        return f"{base}; guardrail=financial_risk_to_zero_guard_v1_triggered"
    return f"{base}; guardrail=financial_risk_to_zero_guard_v1_not_triggered"


def _financial_risk_guard_triggered(pack: dict[str, Any]) -> bool:
    features = pack.get("financial_report_features", {}) if isinstance(pack.get("financial_report_features"), dict) else {}
    quality_risk = _safe(features.get("financial_quality_risk_score"))
    surprise = _safe(features.get("financial_surprise_score"))
    prior = _safe(_extract_python_signal_value(pack, "prior_return_20d"))
    rsi = _safe(_extract_python_signal_value(pack, "rsi14"))
    high_quality_risk = not math.isnan(quality_risk) and quality_risk >= 0.6
    negative_surprise = not math.isnan(surprise) and surprise <= -0.4
    overheat = (not math.isnan(prior) and prior >= 20) or (not math.isnan(rsi) and rsi >= 70)
    if not (high_quality_risk or (negative_surprise and overheat)):
        return False
    return _financial_guard_lacks_confirmation(pack)


def _financial_guard_lacks_confirmation(pack: dict[str, Any]) -> bool:
    if not pack.get("book_skill_candidates"):
        return True
    for item in pack.get("book_skill_candidates", []):
        if isinstance(item, dict) and str(item.get("source_status", "")).lower() in {
            "must_resolve_before_strong_evidence",
            "needs_grounding",
            "weak_until_grounded",
        }:
            return True
    news = pack.get("news_features", {}) if isinstance(pack.get("news_features"), dict) else {}
    questionnaire = pack.get("news_semantic_questionnaire", {}) if isinstance(pack.get("news_semantic_questionnaire"), dict) else {}
    news_missing = _safe(news.get("news_missing_rate"))
    news_risk = _safe(questionnaire.get("ds_news_risk_score"))
    news_uncertainty = _safe(questionnaire.get("ds_news_uncertainty_score"))
    if (not math.isnan(news_missing) and news_missing >= 0.8) or (not math.isnan(news_risk) and news_risk >= 0.6):
        return True
    if not math.isnan(news_uncertainty) and news_uncertainty >= 0.6:
        return True
    return False


def _extract_python_signal_value(pack: dict[str, Any], field: str) -> Any:
    features = pack.get("python_features", {}) if isinstance(pack.get("python_features"), dict) else {}
    if field in features:
        return features.get(field)
    text = str(pack.get("python_signal_summary") or "")
    needle = f"{field}="
    if needle not in text:
        return None
    raw = text.split(needle, 1)[1].split(";", 1)[0].strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _risk_only_keyword_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key not in KEYWORD_OPPORTUNITY_FIELDS}


def _field_subset_by_allowed(fields: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if key in allowed}


def _risk_only_summary(fields: dict[str, Any]) -> str:
    risk = fields.get("ds_news_risk_score", "NA")
    uncertainty = fields.get("ds_news_uncertainty_score", "NA")
    quality = fields.get("ds_news_quality_score", "NA")
    return f"news risk-only questionnaire: risk={risk}; uncertainty={uncertainty}; quality={quality}"


def _uncertainty_only_summary(fields: dict[str, Any]) -> str:
    uncertainty = fields.get("ds_news_uncertainty_score", "NA")
    quality = fields.get("ds_news_quality_score", "NA")
    mainline = fields.get("ds_news_mainline_clarity", "NA")
    repetition = fields.get("ds_news_repetition_lag", "NA")
    return f"news uncertainty-only questionnaire: uncertainty={uncertainty}; quality={quality}; mainline={mainline}; repetition={repetition}"


def _quality_only_summary(fields: dict[str, Any]) -> str:
    quality = fields.get("ds_news_quality_score", "NA")
    coverage = fields.get("ds_news_source_coverage", "NA")
    official = fields.get("ds_news_official_support", "NA")
    timestamp = fields.get("ds_news_timestamp_confidence", "NA")
    return f"news quality-only questionnaire: quality={quality}; coverage={coverage}; official={official}; timestamp={timestamp}"


def _semantic_risk_only_summary(fields: dict[str, Any]) -> str:
    risk = fields.get("ds_news_risk_score", "NA")
    uncertainty = fields.get("ds_news_uncertainty_score", "NA")
    legal = fields.get("ds_news_self_regulatory_legal", "NA")
    peer_risk = fields.get("ds_news_peer_risk_diffusion", "NA")
    return f"news semantic-risk-only questionnaire: risk={risk}; uncertainty={uncertainty}; legal={legal}; peer_risk={peer_risk}"


def _risk_uncertainty_summary(fields: dict[str, Any]) -> str:
    risk = fields.get("ds_news_risk_score", "NA")
    uncertainty = fields.get("ds_news_uncertainty_score", "NA")
    quality = fields.get("ds_news_quality_score", "NA")
    conflict = fields.get("ds_news_conflict_intensity", "NA")
    return f"news risk-uncertainty questionnaire: risk={risk}; uncertainty={uncertainty}; quality={quality}; conflict={conflict}"


def _routine_positive_cap_questionnaire(fields: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(fields or {})
    opportunity = _safe(guarded.get("ds_news_opportunity_score"))
    mainline = _safe(guarded.get("ds_news_mainline_clarity"))
    relevance = _safe(guarded.get("ds_news_decision_relevance"))
    repetition = _safe(guarded.get("ds_news_repetition_lag"))
    should_cap = (
        not math.isnan(opportunity)
        and opportunity >= ROUTINE_POSITIVE_CAP_OPPORTUNITY_THRESHOLD
        and (
            (not math.isnan(mainline) and mainline < ROUTINE_POSITIVE_CAP_MAINLINE_THRESHOLD)
            or (not math.isnan(relevance) and relevance < ROUTINE_POSITIVE_CAP_RELEVANCE_THRESHOLD)
            or (not math.isnan(repetition) and repetition >= ROUTINE_POSITIVE_CAP_REPETITION_THRESHOLD)
        )
    )
    guarded["ds_news_positive_capped_by_rule"] = bool(should_cap)
    guarded["ds_news_positive_cap_rule_id"] = ROUTINE_POSITIVE_CAP_RULE_ID if should_cap else ""
    if not should_cap:
        guarded["ds_news_positive_cap_reason"] = ""
        return guarded
    guarded["ds_news_original_opportunity_score"] = guarded.get("ds_news_opportunity_score")
    guarded["ds_news_original_net_score"] = guarded.get("ds_news_net_score")
    guarded["ds_news_opportunity_score"] = min(opportunity, ROUTINE_POSITIVE_CAP_OPPORTUNITY_SCORE)
    net_score = _safe(guarded.get("ds_news_net_score"))
    if not math.isnan(net_score):
        guarded["ds_news_net_score"] = min(net_score, ROUTINE_POSITIVE_CAP_NET_SCORE)
    reasons = []
    if not math.isnan(mainline) and mainline < ROUTINE_POSITIVE_CAP_MAINLINE_THRESHOLD:
        reasons.append(f"mainline={mainline:.2f}<0.50")
    if not math.isnan(relevance) and relevance < ROUTINE_POSITIVE_CAP_RELEVANCE_THRESHOLD:
        reasons.append(f"relevance={relevance:.2f}<0.50")
    if not math.isnan(repetition) and repetition >= ROUTINE_POSITIVE_CAP_REPETITION_THRESHOLD:
        reasons.append(f"repetition={repetition:.2f}>=0.60")
    guarded["ds_news_positive_cap_reason"] = "; ".join(reasons) or "weak_or_routine_news_mainline"
    notes = str(guarded.get("ds_news_missing_or_conflict_notes") or "").strip()
    cap_note = f"{ROUTINE_POSITIVE_CAP_RULE_ID}: positive news score capped because {guarded['ds_news_positive_cap_reason']}."
    guarded["ds_news_missing_or_conflict_notes"] = f"{notes} | {cap_note}" if notes else cap_note
    return guarded


def _guarded_news_summary(pack: dict[str, Any]) -> str:
    fields = pack.get("news_semantic_questionnaire", {}) if isinstance(pack.get("news_semantic_questionnaire"), dict) else {}
    capped = fields.get("ds_news_positive_capped_by_rule", False)
    if not capped:
        return str(pack.get("news_signal_summary") or "guarded news: no positive cap triggered")
    risk = fields.get("ds_news_risk_score", "NA")
    opportunity = fields.get("ds_news_opportunity_score", "NA")
    net = fields.get("ds_news_net_score", "NA")
    reason = fields.get("ds_news_positive_cap_reason", "")
    return f"guarded news: risk={risk}; capped_opportunity={opportunity}; net={net}; cap_reason={reason}"


def _metrics(cards: list[dict[str, Any]], invalid_outputs: list[dict[str, Any]], source_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    card_frame = pd.DataFrame(cards)
    invalid_rows = _invalid_rows(invalid_outputs)
    variants = sorted(set(card_frame.get("variant", pd.Series(dtype=str)).dropna().astype(str)) | {row["variant"] for row in invalid_rows})
    task_modes = sorted(set(card_frame.get("task_mode", pd.Series(dtype=str)).dropna().astype(str)) | {row["task_mode"] for row in invalid_rows})
    cash = _bank_return_20d()
    if not variants:
        variants = DEFAULT_VARIANTS
    if not task_modes:
        task_modes = ["portfolio_pool", "single_stock"]
    for variant in variants:
        for task_mode in task_modes:
            case_modes = sorted(
                set(cards_subset_modes(card_frame, variant=variant, task_mode=task_mode))
                | {row["case_memory_mode"] for row in invalid_rows if row["variant"] == variant and row["task_mode"] == task_mode}
            ) or ["memory_compact_only"]
            for case_memory_mode in case_modes:
                cards_subset = _card_subset(card_frame, variant=variant, task_mode=task_mode, case_memory_mode=case_memory_mode)
                invalid_subset = [
                    row
                    for row in invalid_rows
                    if row["variant"] == variant and row["task_mode"] == task_mode and row["case_memory_mode"] == case_memory_mode
                ]
                joined = _join_cards(cards_subset, source_frame)
                exposure = joined[joined["is_exposure"]].copy() if not joined.empty else joined
                rows.append(
                    {
                        "variant": variant,
                        "case_memory_mode": case_memory_mode,
                        "task_mode": task_mode,
                        "decision_cards": int(len(cards_subset)),
                        "invalid_outputs": int(len(invalid_subset)),
                        "schema_pass_rate": _rate(len(cards_subset), len(cards_subset) + len(invalid_subset)),
                        "exposure_cards": int(len(exposure)),
                        "avg_return_20d_exposure": _mean(exposure.get("return_20d", pd.Series(dtype=float))),
                        "positive_20d_rate_exposure": _positive(exposure.get("return_20d", pd.Series(dtype=float))),
                        "loss_20d_over_5_rate_exposure": _loss5(exposure.get("return_20d", pd.Series(dtype=float))),
                        "cash_adjusted_avg_return_20d": _mean(_cash_adjusted(joined, cash)),
                        "cash_adjusted_positive_20d_rate": _positive(_cash_adjusted(joined, cash)),
                        "cash_adjusted_loss_20d_over_5_rate": _loss5(_cash_adjusted(joined, cash)),
                        "avg_confidence": _mean(cards_subset.get("confidence_level", pd.Series(dtype=float))) if not cards_subset.empty else None,
                        "research_only": True,
                        "not_investment_instruction": True,
                    }
                )
    return pd.DataFrame(rows)


def _step_metrics(cards: list[dict[str, Any]], invalid_outputs: list[dict[str, Any]], source_frame: pd.DataFrame) -> pd.DataFrame:
    card_frame = pd.DataFrame(cards)
    invalid_rows = _invalid_rows(invalid_outputs)
    groups = set()
    keys = ["variant", "case_memory_mode", "valid_block", "task_mode"]
    if not card_frame.empty:
        groups.update(tuple(row.get(key) for key in keys) for _, row in card_frame.iterrows())
    groups.update((row["variant"], row["case_memory_mode"], row["valid_block"], row["task_mode"]) for row in invalid_rows)
    cash = _bank_return_20d()
    rows = []
    for variant, case_memory_mode, valid_block, task_mode in sorted(groups):
        cards_subset = _card_subset(card_frame, variant=str(variant), task_mode=str(task_mode), valid_block=str(valid_block), case_memory_mode=str(case_memory_mode))
        invalid_subset = [
            row
            for row in invalid_rows
            if row["variant"] == variant
            and row["case_memory_mode"] == case_memory_mode
            and row["valid_block"] == valid_block
            and row["task_mode"] == task_mode
        ]
        joined = _join_cards(cards_subset, source_frame)
        exposure = joined[joined["is_exposure"]].copy() if not joined.empty else joined
        rows.append(
            {
                "variant": variant,
                "case_memory_mode": case_memory_mode,
                "valid_block": valid_block,
                "task_mode": task_mode,
                "decision_cards": int(len(cards_subset)),
                "invalid_outputs": int(len(invalid_subset)),
                "schema_pass_rate": _rate(len(cards_subset), len(cards_subset) + len(invalid_subset)),
                "exposure_cards": int(len(exposure)),
                "avg_return_20d_exposure": _mean(exposure.get("return_20d", pd.Series(dtype=float))),
                "positive_20d_rate_exposure": _positive(exposure.get("return_20d", pd.Series(dtype=float))),
                "cash_adjusted_avg_return_20d": _mean(_cash_adjusted(joined, cash)),
                "cash_adjusted_positive_20d_rate": _positive(_cash_adjusted(joined, cash)),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def _planned_metrics(packs: list[dict[str, Any]]) -> pd.DataFrame:
    if not packs:
        return pd.DataFrame()
    frame = pd.DataFrame(packs)
    rows = []
    if "case_memory_mode" not in frame:
        frame["case_memory_mode"] = "memory_compact_only"
    for (variant, case_memory_mode, valid_block, task_mode), group in frame.groupby(["variant", "case_memory_mode", "valid_block", "task_mode"], sort=True):
        rows.append(
            {
                "variant": variant,
                "case_memory_mode": case_memory_mode,
                "valid_block": valid_block,
                "task_mode": task_mode,
                "planned_evidence_packs": int(len(group)),
                "decision_cards": 0,
                "invalid_outputs": 0,
                "called_deepseek": False,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def _action_diagnostics(cards: list[dict[str, Any]], invalid_outputs: list[dict[str, Any]]) -> pd.DataFrame:
    card_frame = pd.DataFrame(cards)
    invalid_rows = _invalid_rows(invalid_outputs)
    groups = set()
    keys = ["variant", "case_memory_mode", "valid_block", "task_mode"]
    if not card_frame.empty:
        groups.update(tuple(row.get(key) for key in keys) for _, row in card_frame.iterrows())
    groups.update((row["variant"], row["case_memory_mode"], row["valid_block"], row["task_mode"]) for row in invalid_rows)
    rows = []
    for variant, case_memory_mode, valid_block, task_mode in sorted(groups):
        subset = _card_subset(card_frame, variant=str(variant), task_mode=str(task_mode), valid_block=str(valid_block), case_memory_mode=str(case_memory_mode))
        invalid_count = len(
            [
                row
                for row in invalid_rows
                if row["variant"] == variant
                and row["case_memory_mode"] == case_memory_mode
                and row["valid_block"] == valid_block
                and row["task_mode"] == task_mode
            ]
        )
        action_counts = subset.get("simulated_action", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not subset.empty else {}
        grade_counts = subset.get("research_grade", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not subset.empty else {}
        weights = pd.to_numeric(subset.get("simulated_weight_change", pd.Series(dtype=float)), errors="coerce") if not subset.empty else pd.Series(dtype=float)
        confidences = pd.to_numeric(subset.get("confidence_level", pd.Series(dtype=float)), errors="coerce") if not subset.empty else pd.Series(dtype=float)
        decision_cards = int(len(subset))
        rows.append(
            {
                "variant": variant,
                "case_memory_mode": case_memory_mode,
                "valid_block": valid_block,
                "task_mode": task_mode,
                "decision_cards": decision_cards,
                "invalid_outputs": int(invalid_count),
                "increase_cards": int(action_counts.get("增加研究暴露", 0)),
                "observe_cards": int(action_counts.get("保持观察", 0)),
                "reduce_cards": int(action_counts.get("降低研究暴露", 0)),
                "cash_cards": int(action_counts.get("转入现金", 0)),
                "insufficient_no_action_cards": int(action_counts.get("信息不足不动作", 0)),
                "continue_research_cards": int(grade_counts.get("继续深挖", 0)),
                "watch_cards": int(grade_counts.get("放入观察", 0)),
                "exclude_cards": int(grade_counts.get("暂时剔除", 0)),
                "insufficient_grade_cards": int(grade_counts.get("信息不足", 0)),
                "increase_rate": _rate(int(action_counts.get("增加研究暴露", 0)), decision_cards),
                "avg_weight": _mean(weights),
                "avg_confidence": _mean(confidences),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def cards_subset_modes(frame: pd.DataFrame, *, variant: str, task_mode: str) -> set[str]:
    if frame.empty:
        return set()
    subset = frame[frame["variant"].astype(str).eq(str(variant)) & frame["task_mode"].astype(str).eq(str(task_mode))].copy()
    if subset.empty:
        return set()
    if "case_memory_mode" not in subset:
        return {"memory_compact_only"}
    return set(subset["case_memory_mode"].fillna("memory_compact_only").astype(str))


def _card_subset(
    frame: pd.DataFrame,
    *,
    variant: str,
    task_mode: str,
    valid_block: str | None = None,
    case_memory_mode: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    selector = frame["variant"].astype(str).eq(str(variant)) & frame["task_mode"].astype(str).eq(str(task_mode))
    if valid_block is not None:
        selector &= frame["valid_block"].astype(str).eq(str(valid_block))
    if case_memory_mode is not None and "case_memory_mode" in frame:
        selector &= frame["case_memory_mode"].fillna("memory_compact_only").astype(str).eq(str(case_memory_mode))
    return frame[selector].copy()


def _join_cards(cards: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    if cards.empty:
        return pd.DataFrame(columns=["return_20d", "weight", "is_exposure"])
    source = source_frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    lookup = source.set_index(["date", "code"])
    rows = []
    for _, card in cards.iterrows():
        key = (str(card.get("decision_date")), str(card.get("code")).zfill(6))
        if key not in lookup.index:
            continue
        ret = _safe(lookup.loc[key].get("return_20d"))
        if math.isnan(ret):
            continue
        action = str(card.get("simulated_action", ""))
        weight = max(0.0, min(1.0, _safe(card.get("simulated_weight_change"))))
        rows.append({"return_20d": ret, "weight": weight, "is_exposure": action == "增加研究暴露"})
    return pd.DataFrame(rows)


def _cash_adjusted(joined: pd.DataFrame, cash: float) -> pd.Series:
    if joined.empty:
        return pd.Series(dtype=float)
    return joined["weight"].astype(float) * joined["return_20d"].astype(float) + (1 - joined["weight"].astype(float)) * cash


def _invalid_rows(invalid_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in invalid_outputs:
        pack = item.get("evidence_pack") if isinstance(item, dict) else None
        if not isinstance(pack, dict):
            continue
        rows.append(
            {
                "variant": str(pack.get("variant")),
                "case_memory_mode": str(pack.get("case_memory_mode") or "memory_compact_only"),
                "valid_block": str(pack.get("valid_block")),
                "task_mode": str(pack.get("task_mode")),
            }
        )
    return rows


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    blocks: list[str],
    variants: list[str],
    called_deepseek: bool,
    reused: bool,
    base_packs: list[dict[str, Any]],
    ablation_packs: list[dict[str, Any]],
    metrics: pd.DataFrame,
    step_metrics: pd.DataFrame,
    action_diagnostics: pd.DataFrame,
    usage: pd.DataFrame,
    invalid_count: int,
) -> None:
    model_limit = model_concurrency_limit(args.model)
    effective_workers = max(1, min(model_limit if args.max_workers <= 0 else args.max_workers, max(len(ablation_packs), 1), model_limit))
    total_tokens = _sum(usage, "total_tokens")
    lines = [
        _summary_title(args),
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 目的",
        "",
        _summary_purpose(args),
        "",
        "## 配置",
        "",
        f"- called_deepseek: `{called_deepseek}`",
        f"- reused_decision_ledger: `{reused}`",
        f"- model: `{args.model}`",
        f"- requested_max_workers: `{args.max_workers}`",
        f"- effective_workers: `{effective_workers}`",
        f"- model_concurrency_limit: `{model_limit}`",
        f"- blocks: `{','.join(blocks)}`",
        f"- variants: `{','.join(variants)}`",
        f"- case_memory_modes: `{','.join(_parse_case_memory_modes(args.case_memory_modes))}`",
        f"- case_memory_top_k: `{args.case_memory_top_k}`",
        f"- limit_per_mode: `{args.limit_per_mode}`",
        f"- matched_questionnaire_only: `{args.matched_questionnaire_only}`",
        f"- matched_financial_report_only: `{args.matched_financial_report_only}`",
        f"- sample_plan: `{args.sample_plan or ''}`",
        f"- sample_plan_max_rows: `{args.sample_plan_max_rows}`",
        f"- sample_plan_per_rule: `{args.sample_plan_per_rule}`",
        f"- sample_plan_rules: `{args.sample_plan_rules}`",
        f"- sample_plan_task_modes: `{args.sample_plan_task_modes}`",
        f"- base_packs: `{len(base_packs)}`",
        f"- ablation_packs: `{len(ablation_packs)}`",
        f"- guarded_positive_cap_packs: `{_guarded_positive_cap_count(ablation_packs)}`",
        f"- financial_guardrail_triggered_packs: `{_financial_guardrail_trigger_count(ablation_packs)}`",
        f"- invalid_outputs: `{invalid_count}`",
        f"- total_tokens: `{total_tokens}`",
        "",
        "## Variant 定义",
        "",
        "- `no_news`：DeepSeek 看不到关键词统计和语义问卷。",
        "- `keyword_only`：只看可复现关键词/事件统计。",
        "- `questionnaire_only`：只看 DeepSeek 新闻语义问卷。",
        "- `keyword_plus_questionnaire`：关键词统计和完整问卷同时可见。",
        "- `keyword_plus_questionnaire_guarded`：关键词统计和完整问卷同时可见，但常规/弱主线高机会分会被封顶。",
        "- `risk_only_questionnaire`：隐藏机会/净分，只保留风险、不确定性、质量、冲突类新闻证据。",
        "- `uncertainty_only_questionnaire`：隐藏关键词、机会和风险字段，只保留来源、时间戳、主线、冲突、重复滞后和不确定性。",
        "- `quality_only_questionnaire`：隐藏关键词和方向性字段，只保留来源覆盖、官方确认、时间戳、主线和证据质量。",
        "- `semantic_risk_only_questionnaire`：隐藏关键词统计，只保留 DeepSeek 语义风险字段。",
        "- `risk_uncertainty_questionnaire`：保留风险关键词和语义风险/不确定性字段，继续隐藏正向机会/净分。",
        "- `no_financial_report_channel`：隐藏财报/业绩公告事件通道，普通新闻仍可见。",
        "- `financial_report_only`：隐藏普通新闻关键词和问卷，只保留财报/业绩公告事件通道。",
        "- `news_plus_financial_report`：普通新闻和财报/业绩公告事件通道同时可见，财报仍作为独立证据块。",
        "- `news_plus_financial_report_guarded`：普通新闻和财报同时可见，并显式加入 `financial_risk_to_zero_guard_v1`；高财务风险或负惊喜过热且缺少交叉确认时，执行层不得增加研究暴露。",
        "- `no_kline`：隐藏多尺度 K 线和同组 K 线证据。",
        "- `kline_weak_prompt`：显示多尺度 K 线弱提示；20 日回撤只能作为观察提示，60 日深跌和弱 peer 广度不得作为正向 gate。",
        "",
        "## Case Memory 定义",
        "",
        "- `no_rag`：隐藏 compact memory 和 retrieved cases，用作灰色参考。",
        "- `memory_compact_only`：只使用结构化 compact ledger memory，是当前默认方式。",
        "- `retrieved_cases_v1`：在 compact memory 基础上加入本地案例检索结果；只允许 case_id、规则状态、失败条件和下一步动作，不允许后验收益/GT 字段。",
        "- `retrieved_cases_v2_applicable`：在 v1 基础上增加适用性 checklist；只有命中相同失败条件的 case 可作 counter-evidence，partial case 只能观察，不得单独升级或降级。",
        "",
        "## 汇总指标",
        "",
        _table(metrics),
        "",
        "## 分块指标",
        "",
        _table(step_metrics),
        "",
        "## 行动诊断",
        "",
        _table(action_diagnostics),
        "",
        "## 解释边界",
        "",
        "- 这是训练/验证阶段，只能用于优化策略和记录经验。",
        "- 若 `keyword_plus_questionnaire` 或 `questionnaire_only` 没有稳定优于 `no_news`，不得宣称新闻 alpha。",
        "- 若 `keyword_plus_questionnaire_guarded` 稳定优于 `keyword_plus_questionnaire`，说明主线清晰度/决策相关性保护规则值得进入下一轮默认策略。",
        "- 若 `risk_only_questionnaire` 优于完整问卷，说明新闻更适合作反证而不是正向催化。",
        "- 财报通道当前只证明工程可用和时间安全；若 `financial_report_only`、`news_plus_financial_report` 或 `news_plus_financial_report_guarded` 未稳定优于 `no_financial_report_channel`，不得提高财报权重。",
        _sample_scope_note(args),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_title(args: argparse.Namespace) -> str:
    variants = set(_parse_variants(args.variants))
    if args.sample_plan and variants <= set(FINANCIAL_REPORT_VARIANTS):
        return "# DeepSeek Financial Report Channel Full-Agent Ablation"
    return "# DeepSeek News Channel Full-Agent Ablation"


def _summary_purpose(args: argparse.Namespace) -> str:
    variants = set(_parse_variants(args.variants))
    if args.sample_plan and variants <= set(FINANCIAL_REPORT_VARIANTS):
        return "把财报/业绩公告通道放进真实 Agent 决策，比较同一批样本在隐藏财报、仅财报、新闻+财报、新闻+财报+风险护栏输入下的表现，避免把本地后验分层误当成策略能力。"
    return "把新闻关键词层和 DeepSeek 语义问卷层放进真实 Agent 决策，比较同一批样本在不同新闻输入下的表现，避免只用问卷分数推断策略能力。"


def _sample_scope_note(args: argparse.Namespace) -> str:
    if args.sample_plan:
        return "- 样本来自外部 sample plan，只能验证该计划覆盖的候选规则，不能替代跨 2023/2024/2025/2026 的最终日期泛化验收。"
    return "- 样本仍来自 matched-news 窗口，不能替代跨 2023/2024/2025/2026 的最终日期泛化验收。"


def _load_memory_context() -> str:
    return load_compact_memory_context(ROOT)


def _apply_posthoc_guardrails(cards: list[dict[str, Any]], packs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(pack.get("variant")),
            str(pack.get("case_memory_mode") or "memory_compact_only"),
            str(pack.get("valid_block")),
            str(pack.get("task_mode")),
            str(pack.get("decision_date")),
            str(pack.get("code")).zfill(6),
        ): pack
        for pack in packs
    }
    updated = []
    for card in cards:
        item = dict(card)
        key = (
            str(item.get("variant")),
            str(item.get("case_memory_mode") or "memory_compact_only"),
            str(item.get("valid_block")),
            str(item.get("task_mode")),
            str(item.get("decision_date")),
            str(item.get("code")).zfill(6),
        )
        pack = lookup.get(key)
        if pack:
            apply_decision_guardrails(item, pack)
        updated.append(item)
    return updated


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for card in cards:
        keyed[_card_key(card)] = card
    return [keyed[key] for key in sorted(keyed)]


def _card_key(card: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(card.get("variant")),
        str(card.get("case_memory_mode") or "memory_compact_only"),
        str(card.get("valid_block")),
        str(card.get("task_mode")),
        str(card.get("decision_date")),
        str(card.get("code")).zfill(6),
    )


def _pack_key(pack: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(pack.get("variant")),
        str(pack.get("case_memory_mode") or "memory_compact_only"),
        str(pack.get("valid_block")),
        str(pack.get("task_mode")),
        str(pack.get("decision_date")),
        str(pack.get("code")).zfill(6),
    )


def _guarded_positive_cap_count(packs: list[dict[str, Any]]) -> int:
    return int(
        sum(
            1
            for pack in packs
            if pack.get("variant") == "keyword_plus_questionnaire_guarded"
            and isinstance(pack.get("news_semantic_questionnaire"), dict)
            and pack["news_semantic_questionnaire"].get("ds_news_positive_capped_by_rule") is True
        )
    )


def _financial_guardrail_trigger_count(packs: list[dict[str, Any]]) -> int:
    return int(
        sum(
            1
            for pack in packs
            if pack.get("variant") == "news_plus_financial_report_guarded"
            and isinstance(pack.get("financial_report_guardrail"), dict)
            and pack["financial_report_guardrail"].get("triggered_for_pack") is True
        )
    )


def _update_usage_index(prefix: str, usage_path: Path, usage: pd.DataFrame) -> None:
    path = OUTPUT / "deepseek_usage_summary.csv"
    rows = []
    if path.exists():
        try:
            rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
        except csv.Error:
            rows = []
    rows = [row for row in rows if row.get("source_file") != usage_path.name]
    rows.append(
        {
            "source_file": usage_path.name,
            "rows": str(len(usage)),
            "columns": ";".join(usage.columns),
            "experiment_prefix": prefix,
            "total_tokens": str(_sum(usage, "total_tokens")),
            "invalid_rows": str(int((usage.get("status", pd.Series(dtype=str)).astype(str) != "ok").sum())) if not usage.empty and "status" in usage else "0",
        }
    )
    fieldnames = ["source_file", "rows", "columns", "experiment_prefix", "total_tokens", "invalid_rows"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _source_rank(name: str) -> int:
    if "retry" in name:
        return 80
    if "spread_panel_v2" in name:
        return 70
    if "spread_panel" in name:
        return 40
    if "compressed_panel" in name:
        return 30
    if "compact" in name:
        return 20
    return 10


def _parse_blocks(raw: str) -> list[str]:
    blocks = [item.strip() for item in raw.split(",") if item.strip()]
    return blocks or ["H2025_1", "H2026_1"]


def _parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in KNOWN_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return variants or DEFAULT_VARIANTS


def _parse_case_memory_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = [item for item in modes if item not in KNOWN_CASE_MEMORY_MODES]
    if unknown:
        raise ValueError(f"unknown case memory modes: {unknown}")
    return modes or ["memory_compact_only"]


def _bank_return_20d() -> float:
    return ((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100


def _sum(frame: pd.DataFrame, field: str) -> int:
    if frame.empty or field not in frame:
        return 0
    return int(pd.to_numeric(frame[field], errors="coerce").fillna(0).sum())


def _mean(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.mean()), 4)


def _positive(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float((series > 0).mean()), 4)


def _loss5(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float((series <= -5).mean()), 4)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "deepseek_news_ablation_round"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
