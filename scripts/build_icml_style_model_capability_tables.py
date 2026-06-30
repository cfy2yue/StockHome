from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"


P0_LOCAL_STABILITY = REPORT_DIR / "p0_decision_stack_v1_friday_24panel_h2026_panel_stability.csv"
P0_LOCAL_SUMMARY = REPORT_DIR / "p0_decision_stack_v1_friday_24panel_summary.csv"
P0_FLASH_METRICS = REPORT_DIR / "p0_acceptance_multiblock_3panel_flash_v1_metrics.csv"
P0_PRO_METRICS = REPORT_DIR / "p0_acceptance_single_default_pro_v1_metrics.csv"
P0_BRANCH_FLASH_METRICS = REPORT_DIR / "single_stock_branch_guardrail_panel24_flash_v1_metrics.csv"
P0_BRANCH_FLASH_STEP_METRICS = REPORT_DIR / "single_stock_branch_guardrail_panel24_flash_v1_step_metrics.csv"
P1_ABLATION = REPORT_DIR / "user_capability_final_tables_v2_candidate_agent_ablation_flash_pro.csv"
P1_OPERATION_CONFIRM = REPORT_DIR / "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1_aggregate.csv"
CROSS_SECTOR_STRESS = REPORT_DIR / "cross_sector_ridge_flash_pro_ablation_k03_v1_model_variant_summary.csv"
CROSS_SECTOR_HYGIENE = REPORT_DIR / "cross_sector_ridge_flash_pro_ablation_k03_v1_hygiene.csv"
P0_OPERATION_PANEL = REPORT_DIR / "p0_operation_policy_v1_h2026_panel_summary.csv"
P0_SMALL_ENTRY_OVERLAY = REPORT_DIR / "p0_small_entry_nonprice_overlay_v1_summary.csv"
P0_SMALL_ENTRY_ML_CONFIRMER = REPORT_DIR / "p0_small_entry_ml_confirmer_v1_summary.csv"
P0_SMALL_ENTRY_TRANSFER_CONFIRMER = REPORT_DIR / "p0_small_entry_transfer_confirmer_v1_summary.csv"
P0_SMALL_ENTRY_TRANSFER_CHANNEL_CONFIRM = REPORT_DIR / "p0_transfer_channel_confirm_v1_summary.csv"
P0_SMALL_ENTRY_TRANSFER_CHANNEL_PANEL = REPORT_DIR / "p0_transfer_channel_confirm_v1_h2026_panel_summary.csv"
P0_SMALL_ENTRY_TRANSFER_ANALOG_RAG = REPORT_DIR / "p0_transfer_analog_rag_v1_summary.csv"
P0_SMALL_ENTRY_TRANSFER_ANALOG_RAG_PANEL = REPORT_DIR / "p0_transfer_analog_rag_v1_h2026_panel_summary.csv"
P0_TRANSFER_ANALOG_RAG_ONOFF_VISIBILITY = (
    REPORT_DIR / "p0_transfer_analog_rag_onoff_dryrun_v1_onoff_visibility_summary.csv"
)
P0_TRANSFER_ANALOG_RAG_KLINE_TOOL_ONOFF_VISIBILITY = (
    REPORT_DIR / "p0_transfer_analog_rag_kline_tool_onoff_dryrun_v1_onoff_visibility_summary.csv"
)
P0_TRANSFER_ANALOG_RAG_PANEL36_PREFLIGHT = REPORT_DIR / "p0_transfer_analog_rag_panel36_preflight_gate_v1.csv"
P0_SMALL_ENTRY_CASE_MEMORY_V2 = REPORT_DIR / "p0_small_entry_case_memory_v2_summary.csv"
P0_SMALL_ENTRY_CASE_MEMORY_MIN1 = REPORT_DIR / "p0_small_entry_case_memory_min1_v1_summary.csv"
P0_SMALL_ENTRY_BOOKSKILL = REPORT_DIR / "p0_small_entry_bookskill_attribution_v1_rule_summary.csv"
P0_SMALL_ENTRY_PPS_Q017_ONOFF_EVIDENCE = (
    REPORT_DIR / "p0_small_entry_pps_q017_onoff_dryrun_v1_onoff_evidence_audit.csv"
)
P0_SMALL_ENTRY_PPS_Q017_ONOFF_VARIANT = (
    REPORT_DIR / "p0_small_entry_pps_q017_onoff_dryrun_v1_onoff_variant_summary.csv"
)
P0_SMALL_ENTRY_PPS_Q017_ONOFF_PAIR = (
    REPORT_DIR / "p0_small_entry_pps_q017_onoff_dryrun_v1_onoff_pair_summary.csv"
)
P0_SMALL_ENTRY_PPS_Q017_INTERACTIONS = (
    REPORT_DIR / "p0_small_entry_pps_q017_channel_interactions_v1_summary.csv"
)
P0_KLINE_THRESHOLD_COMPARE = REPORT_DIR / "single_stock_kline_threshold_compare_v1_comparison.csv"
P0_ACTIVE_ENTRY_CALIBRATION = REPORT_DIR / "userpath_active_entry_calibration_v1_ranking.csv"
P0_ACTIVE_ENTRY_CALIBRATION_HYGIENE = REPORT_DIR / "userpath_active_entry_calibration_v1_hygiene.csv"

P0_COMPONENT_SUMMARIES = {
    "BookSkill": REPORT_DIR / "p0_acceptance_bookskill_paired_v1_summary.csv",
    "Branch/RAG Case": REPORT_DIR / "p0_acceptance_branch_case_paired_v1_summary.csv",
    "Memory/RAG": REPORT_DIR / "p0_acceptance_memory_rag_paired_v1_summary.csv",
    "News": REPORT_DIR / "p0_acceptance_news_paired_v1_summary.csv",
    "Opportunity Tool": REPORT_DIR / "p0_acceptance_opportunity_tool_paired_v1_summary.csv",
    "Peer Context": REPORT_DIR / "p0_acceptance_peer_paired_v1_summary.csv",
    "Quant Tools": REPORT_DIR / "p0_acceptance_quant_tools_paired_v1_summary.csv",
    "Risk Review Queue": REPORT_DIR / "p0_acceptance_risk_review_queue_paired_v1_summary.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ICML/NeurIPS-style Flash/Pro capability and ablation tables."
    )
    parser.add_argument("--output-prefix", default="flash_pro_icml_capability_tables_v13")
    parser.add_argument(
        "--deepseek-status",
        default="2026-06-29 smoke rerun confirmed 402 Payment Required; no new Flash/Pro DS calls were made for v13.",
    )
    args = parser.parse_args()

    prefix = safe_prefix(args.output_prefix)
    outputs = {
        "p0_local": REPORT_DIR / f"{prefix}_p0_local_24panel.csv",
        "p0_ds": REPORT_DIR / f"{prefix}_p0_flash_pro_ds.csv",
        "p0_user_path": REPORT_DIR / f"{prefix}_p0_user_path_latest.csv",
        "p0_operation": REPORT_DIR / f"{prefix}_p0_operation_policy.csv",
        "p0_small_overlay": REPORT_DIR / f"{prefix}_p0_small_entry_overlay_ablation.csv",
        "p0_small_ml": REPORT_DIR / f"{prefix}_p0_small_entry_ml_confirmer.csv",
        "p0_small_transfer": REPORT_DIR / f"{prefix}_p0_small_entry_transfer_confirmer.csv",
        "p0_small_transfer_channel": REPORT_DIR / f"{prefix}_p0_small_entry_transfer_channel_confirm.csv",
        "p0_small_transfer_analog_rag": REPORT_DIR / f"{prefix}_p0_small_entry_transfer_analog_rag.csv",
        "p0_transfer_analog_rag_onoff": REPORT_DIR / f"{prefix}_p0_transfer_analog_rag_onoff_readiness.csv",
        "p0_transfer_analog_rag_kline_tool_onoff": REPORT_DIR
        / f"{prefix}_p0_transfer_analog_rag_kline_tool_onoff_readiness.csv",
        "p0_transfer_analog_rag_panel36_preflight": REPORT_DIR
        / f"{prefix}_p0_transfer_analog_rag_panel36_preflight.csv",
        "p0_small_case_memory": REPORT_DIR / f"{prefix}_p0_small_entry_case_memory_rag_ablation.csv",
        "p0_small_bookskill": REPORT_DIR / f"{prefix}_p0_small_entry_bookskill_attribution.csv",
        "p0_pps_q017_onoff": REPORT_DIR / f"{prefix}_p0_pps_q017_onoff_readiness.csv",
        "p0_pps_q017_interactions": REPORT_DIR / f"{prefix}_p0_pps_q017_channel_interactions.csv",
        "p0_kline_threshold": REPORT_DIR / f"{prefix}_p0_kline_threshold_compare.csv",
        "p0_active_entry_calibration": REPORT_DIR / f"{prefix}_p0_active_entry_calibration.csv",
        "p0_components": REPORT_DIR / f"{prefix}_p0_component_ablation.csv",
        "p1_candidate": REPORT_DIR / f"{prefix}_p1_candidate_flash_pro_ablation.csv",
        "p1_operation_confirm": REPORT_DIR / f"{prefix}_p1_operation_confirm.csv",
        "cross_stress": REPORT_DIR / f"{prefix}_cross_sector_stress_3seed.csv",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
        "latex": REPORT_DIR / f"{prefix}_latex_tables.tex",
    }

    p0_local = build_p0_local_table()
    p0_ds = build_p0_ds_table()
    p0_user_path = build_p0_user_path_table()
    p0_operation = build_p0_operation_table()
    p0_small_overlay = build_p0_small_overlay_table()
    p0_small_ml = build_p0_small_ml_confirmer_table()
    p0_small_transfer = build_p0_small_transfer_confirmer_table()
    p0_small_transfer_channel = build_p0_small_transfer_channel_table()
    p0_small_transfer_analog_rag = build_p0_small_transfer_analog_rag_table()
    p0_transfer_analog_rag_onoff = build_p0_transfer_analog_rag_onoff_readiness_table(
        P0_TRANSFER_ANALOG_RAG_ONOFF_VISIBILITY,
        model_stage="dryrun_evidence_ready_no_ds_call",
    )
    p0_transfer_analog_rag_kline_tool_onoff = build_p0_transfer_analog_rag_onoff_readiness_table(
        P0_TRANSFER_ANALOG_RAG_KLINE_TOOL_ONOFF_VISIBILITY,
        model_stage="dryrun_with_single_stock_kline_quant_tool_no_ds_call",
    )
    p0_transfer_analog_rag_panel36_preflight = build_p0_transfer_analog_rag_panel36_preflight_table()
    p0_small_case_memory = build_p0_small_case_memory_table()
    p0_small_bookskill = build_p0_small_bookskill_table()
    p0_pps_q017_onoff = build_p0_pps_q017_onoff_table()
    p0_pps_q017_interactions = build_p0_pps_q017_interaction_table()
    p0_kline_threshold = build_p0_kline_threshold_table()
    p0_active_entry_calibration = build_p0_active_entry_calibration_table()
    p0_components = build_p0_component_table()
    p1_candidate = build_p1_candidate_table()
    p1_operation_confirm = build_p1_operation_confirm_table()
    cross_stress = build_cross_sector_stress_table()
    hygiene = build_hygiene_table(args.deepseek_status)

    frames = {
        "p0_local": p0_local,
        "p0_ds": p0_ds,
        "p0_user_path": p0_user_path,
        "p0_operation": p0_operation,
        "p0_small_overlay": p0_small_overlay,
        "p0_small_ml": p0_small_ml,
        "p0_small_transfer": p0_small_transfer,
        "p0_small_transfer_channel": p0_small_transfer_channel,
        "p0_small_transfer_analog_rag": p0_small_transfer_analog_rag,
        "p0_transfer_analog_rag_onoff": p0_transfer_analog_rag_onoff,
        "p0_transfer_analog_rag_kline_tool_onoff": p0_transfer_analog_rag_kline_tool_onoff,
        "p0_transfer_analog_rag_panel36_preflight": p0_transfer_analog_rag_panel36_preflight,
        "p0_small_case_memory": p0_small_case_memory,
        "p0_small_bookskill": p0_small_bookskill,
        "p0_pps_q017_onoff": p0_pps_q017_onoff,
        "p0_pps_q017_interactions": p0_pps_q017_interactions,
        "p0_kline_threshold": p0_kline_threshold,
        "p0_active_entry_calibration": p0_active_entry_calibration,
        "p0_components": p0_components,
        "p1_candidate": p1_candidate,
        "p1_operation_confirm": p1_operation_confirm,
        "cross_stress": cross_stress,
        "hygiene": hygiene,
    }
    for key, frame in frames.items():
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    outputs["report"].write_text(
        render_report(
            p0_local=p0_local,
            p0_ds=p0_ds,
            p0_user_path=p0_user_path,
            p0_operation=p0_operation,
            p0_small_overlay=p0_small_overlay,
            p0_small_ml=p0_small_ml,
            p0_small_transfer=p0_small_transfer,
            p0_small_transfer_channel=p0_small_transfer_channel,
            p0_small_transfer_analog_rag=p0_small_transfer_analog_rag,
            p0_transfer_analog_rag_onoff=p0_transfer_analog_rag_onoff,
            p0_transfer_analog_rag_kline_tool_onoff=p0_transfer_analog_rag_kline_tool_onoff,
            p0_transfer_analog_rag_panel36_preflight=p0_transfer_analog_rag_panel36_preflight,
            p0_small_case_memory=p0_small_case_memory,
            p0_small_bookskill=p0_small_bookskill,
            p0_pps_q017_onoff=p0_pps_q017_onoff,
            p0_pps_q017_interactions=p0_pps_q017_interactions,
            p0_kline_threshold=p0_kline_threshold,
            p0_active_entry_calibration=p0_active_entry_calibration,
            p0_components=p0_components,
            p1_candidate=p1_candidate,
            p1_operation_confirm=p1_operation_confirm,
            cross_stress=cross_stress,
            hygiene=hygiene,
            outputs=outputs,
        ),
        encoding="utf-8",
    )
    outputs["latex"].write_text(
        render_latex(
            p0_local=p0_local,
            p0_ds=p0_ds,
            p0_user_path=p0_user_path,
            p0_operation=p0_operation,
            p0_small_overlay=p0_small_overlay,
            p0_small_ml=p0_small_ml,
            p0_small_transfer=p0_small_transfer,
            p0_small_transfer_channel=p0_small_transfer_channel,
            p0_small_transfer_analog_rag=p0_small_transfer_analog_rag,
            p0_transfer_analog_rag_onoff=p0_transfer_analog_rag_onoff,
            p0_transfer_analog_rag_kline_tool_onoff=p0_transfer_analog_rag_kline_tool_onoff,
            p0_transfer_analog_rag_panel36_preflight=p0_transfer_analog_rag_panel36_preflight,
            p0_small_case_memory=p0_small_case_memory,
            p0_small_bookskill=p0_small_bookskill,
            p0_pps_q017_onoff=p0_pps_q017_onoff,
            p0_pps_q017_interactions=p0_pps_q017_interactions,
            p0_kline_threshold=p0_kline_threshold,
            p0_active_entry_calibration=p0_active_entry_calibration,
            p0_components=p0_components,
            p1_candidate=p1_candidate,
            p1_operation_confirm=p1_operation_confirm,
            cross_stress=cross_stress,
            hygiene=hygiene,
        ),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"wrote: {outputs['report']}")
    print(f"wrote: {outputs['latex']}")


def build_p0_local_table() -> pd.DataFrame:
    stability = read_csv(P0_LOCAL_STABILITY)
    summary = read_csv(P0_LOCAL_SUMMARY)
    keep = [
        "bank_all_baseline",
        "hold_all_baseline",
        "opp_only",
        "kline_only_no_raise",
        "opp_kline_confirm_no_raise",
        "branch_stack_v1",
    ]
    rows: list[dict[str, Any]] = []
    for policy in keep:
        subset = stability[stability["policy_name"].astype(str).eq(policy)].copy()
        full = summary[summary["policy_name"].astype(str).eq(policy)]
        if subset.empty:
            continue
        rows.append(
            {
                "task": "P0 single-stock watch",
                "model": "Local deterministic stack",
                "config": policy,
                "role": "gray_baseline" if "baseline" in policy else "candidate_or_ablation",
                "split": "H2026_1 strict 100-stock panels",
                "panels": int(subset["panel_seed"].nunique()),
                "cards_or_rows": int(len(subset)),
                "active_rate": mean(subset, "active_rate"),
                "active_pos20_mean": mean(subset, "active_pos_rate"),
                "active_pos20_std": std(subset, "active_pos_rate"),
                "active_avg20_pp_mean": mean(subset, "active_avg_return"),
                "active_avg20_pp_std": std(subset, "active_avg_return"),
                "strategy_avg20_pp_mean": mean(subset, "strategy_avg_return"),
                "strategy_avg20_pp_std": std(subset, "strategy_avg_return"),
                "loss_gt5_rate_mean": mean(subset, "active_loss_gt5_rate"),
                "promotion_status": first_value(full, "promotion_status"),
                "main_read": p0_local_read(policy),
            }
        )
    return pd.DataFrame(rows).round(6)


def build_p0_ds_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, source, path in [
        ("DS V4 Flash", "p0_acceptance_multiblock_3panel_flash_v1", P0_FLASH_METRICS),
        ("DS V4 Pro", "p0_acceptance_single_default_pro_v1", P0_PRO_METRICS),
    ]:
        frame = read_csv(path)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "task": "P0 single-stock watch",
                    "model": model,
                    "config": row.get("variant"),
                    "role": "anchor" if str(row.get("variant")) in {"full_agent_without_opportunity_tool", "full_agent_with_opportunity_tool"} else "ablation",
                    "split": "6 half-year blocks x 3 panels",
                    "panels": 3,
                    "cards": int_or_none(row.get("decision_cards")),
                    "invalid": int_or_none(row.get("invalid_outputs")),
                    "schema_pass_rate": num(row.get("schema_pass_rate")),
                    "cash_pos20": num(row.get("cash_adjusted_positive_20d_rate")),
                    "cash_avg20_pp": num(row.get("cash_adjusted_avg_return_20d")),
                    "cash_std20_pp": num(row.get("cash_adjusted_std_return_20d")),
                    "active_exposure": num(row.get("active_exposure")),
                    "exposure_cards": int_or_none(row.get("exposure_cards")),
                    "data_missing_cards": int_or_none(row.get("data_missing_flag_cards")),
                    "source_prefix": source,
                    "main_read": ds_p0_read(model, str(row.get("variant"))),
                }
            )
    return pd.DataFrame(rows).round(6)


def build_p0_user_path_table() -> pd.DataFrame:
    frame = read_csv(P0_BRANCH_FLASH_STEP_METRICS)
    if frame.empty:
        frame = read_csv(P0_BRANCH_FLASH_METRICS)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        variant = str(row.get("variant"))
        cash_pos = num(row.get("cash_adjusted_positive_20d_rate"))
        active = num(row.get("active_exposure"))
        rows.append(
            {
                "task": "P0 single-stock user path",
                "model": "DS V4 Flash",
                "config": variant,
                "role": "anchor" if variant in {"full_agent_with_opportunity_tool", "full_agent_with_risk_review_queue"} else "ablation",
                "split": row.get("valid_block", "H2026_1"),
                "cards": int_or_none(row.get("decision_cards")),
                "invalid": int_or_none(row.get("invalid_outputs")),
                "schema_pass_rate": num(row.get("schema_pass_rate")),
                "cash_pos20": cash_pos,
                "cash_avg20_pp": num(row.get("cash_adjusted_avg_return_20d")),
                "cash_std20_pp": num(row.get("cash_adjusted_std_return_20d")),
                "active_exposure": active,
                "exposure_cards": int_or_none(row.get("exposure_cards")),
                "data_missing_cards": int_or_none(row.get("data_missing_flag_cards")),
                "verdict": p0_user_path_verdict(cash_pos, active),
            }
        )
    return pd.DataFrame(rows).round(6)


def build_p0_operation_table() -> pd.DataFrame:
    frame = read_csv(P0_OPERATION_PANEL)
    if frame.empty:
        return pd.DataFrame()
    keep_policies = {
        "bank_all_baseline",
        "hold_all_baseline",
        "opp_only",
        "opp_kline_confirm_no_raise",
        "branch_stack_v1",
    }
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        policy = str(row.get("policy_name"))
        if policy not in keep_policies:
            continue
        rows.append(
            {
                "task": "P0 single-stock operation policy",
                "model": "Local deterministic action policy",
                "frequency": row.get("frequency"),
                "config": policy,
                "role": "gray_baseline" if "baseline" in policy else "candidate_or_ablation",
                "split": "H2026_1 strict 100-stock panels",
                "panels": int_or_none(row.get("panels")),
                "cash_pos20_mean_std": row.get("cash_pos20_mean±std"),
                "cash_avg20_pp_mean_std": row.get("cash_avg20_mean±std"),
                "buy_add_pos20_mean_std": row.get("buy_add_pos20_mean±std"),
                "buy_add_avg20_pp_mean_std": row.get("buy_add_avg20_mean±std"),
                "small_entry_pos20_mean_std": row.get("small_buy_pos20_mean±std"),
                "small_entry_avg20_pp_mean_std": row.get("small_buy_avg20_mean±std"),
                "reduce_correct_mean_std": row.get("reduce_correct_mean±std"),
                "false_large_gain_mean_std": row.get("false_large_gain_mean±std"),
                "buy_add_rate_mean_std": row.get("buy_add_rate_mean±std"),
                "main_read": p0_operation_read(str(row.get("frequency")), policy, row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["frequency", "role", "config"])
    return out


def build_p0_small_overlay_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_OVERLAY)
    if frame.empty:
        return pd.DataFrame()
    keep_rules = {
        "small_entry_all",
        "news_low_warning",
        "not_overheated_rsi",
        "nonprice_confirm_min2",
        "small_entry_clean_confirmed",
        "peer_relative_positive",
        "financial_available_low_risk",
        "news_available_low_warning",
        "chip_support_v2",
    }
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rule = str(row.get("rule_id"))
        if rule not in keep_rules:
            continue
        rows.append(
            {
                "task": "P0 small-entry overlay ablation",
                "model": "Local non-price overlay rules",
                "frequency": row.get("frequency"),
                "config": rule,
                "role": "branch_reference" if rule == "small_entry_all" else "ablation",
                "split": "walk-forward prior blocks + H2026_1 OOT",
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_pos20_mean": num(row.get("prior_selected_pos20_mean")),
                "prior_avg20_pp_mean": num(row.get("prior_selected_avg20_mean")),
                "h2026_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_rate_vs_branch": num(row.get("h2026_selected_rate")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos": num(row.get("h2026_delta_pos")),
                "h2026_delta_avg_pp": num(row.get("h2026_delta_avg")),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_overlay_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["frequency", "role", "h2026_pos20"], ascending=[True, True, False])
    return out.round(6)


def build_p0_small_ml_confirmer_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_ML_CONFIRMER)
    if frame.empty:
        return pd.DataFrame()
    branch_refs = frame[frame["variant"].astype(str).eq("small_entry_all")]
    selected_rows = (
        pd.to_numeric(frame["h2026_selected_rows"], errors="coerce").fillna(0)
        if "h2026_selected_rows" in frame
        else pd.Series(0, index=frame.index)
    )
    eligible = frame[selected_rows >= 50].copy()
    selected = [branch_refs]
    if not eligible.empty:
        selected.append(eligible.sort_values(["h2026_selected_pos20", "h2026_delta_pos"], ascending=False).head(8))
        for frequency in sorted(eligible["frequency"].dropna().unique()):
            freq = eligible[eligible["frequency"].eq(frequency)]
            selected.append(freq.sort_values(["h2026_delta_pos", "h2026_selected_rows"], ascending=False).head(3))
    subset = pd.concat(selected, ignore_index=True).drop_duplicates(["frequency", "variant"])
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        variant = str(row.get("variant"))
        rows.append(
            {
                "task": "P0 small-entry ML confirmer",
                "model": "Local logistic confirmer",
                "frequency": row.get("frequency"),
                "config": variant,
                "role": "branch_reference" if variant == "small_entry_all" else "diagnostic_ml",
                "split": "prior-tail validation + H2026_1 OOT",
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "h2026_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_rate_vs_branch": num(row.get("h2026_selected_rate")),
                "h2026_base_pos20": num(row.get("h2026_base_pos20")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos": num(row.get("h2026_delta_pos")),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_small_ml_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["role", "h2026_pos20", "h2026_delta_pos"], ascending=[True, False, False])
    return out.round(6)


def build_p0_small_transfer_confirmer_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_TRANSFER_CONFIRMER)
    if frame.empty:
        return pd.DataFrame()
    branch_refs = frame[frame["variant"].astype(str).eq("small_entry_all")]
    promoted_like = frame[frame["promotion_status"].astype(str).str.contains("yellow|green", regex=True)].copy()
    diagnostic = frame[frame["promotion_status"].astype(str).str.contains("observe_diagnostic_only", regex=True)].copy()
    selected = [branch_refs]
    if not promoted_like.empty:
        selected.append(
            promoted_like.sort_values(
                ["promotion_status", "rank_score", "h2026_selected_pos20", "h2026_selected_avg20"],
                ascending=[True, False, False, False],
            )
        )
    if not diagnostic.empty:
        selected.append(
            diagnostic.sort_values(
                ["rank_score", "h2026_selected_pos20", "h2026_selected_avg20"],
                ascending=[False, False, False],
            ).head(8)
        )
        for frequency in sorted(diagnostic["frequency"].dropna().unique()):
            freq = diagnostic[diagnostic["frequency"].eq(frequency)]
            selected.append(freq.sort_values(["rank_score", "h2026_selected_rows"], ascending=False).head(3))
    subset = pd.concat(selected, ignore_index=True).drop_duplicates(["frequency", "variant"])
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        variant = str(row.get("variant"))
        rows.append(
            {
                "task": "P0 small-entry transfer confirmer",
                "model": "Local transfer logistic confirmer",
                "frequency": row.get("frequency"),
                "config": variant,
                "role": "branch_reference" if variant == "small_entry_all" else "transfer_ml_ablation",
                "split": "larger prior candidate pool + H2026_1 OOT",
                "cohort": row.get("cohort"),
                "feature_set": row.get("feature_set"),
                "confirm_quantile": num(row.get("confirm_quantile")),
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "prior_delta_pos_mean": num(row.get("prior_delta_pos_mean")),
                "prior_delta_avg_mean": num(row.get("prior_delta_avg_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "h2026_base_rows": int_or_none(row.get("h2026_base_rows")),
                "h2026_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_rate_vs_branch": num(row.get("h2026_selected_rate")),
                "h2026_base_pos20": num(row.get("h2026_base_pos20")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos": num(row.get("h2026_delta_pos")),
                "h2026_delta_avg_pp": num(row.get("h2026_delta_avg")),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_small_transfer_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["role", "frequency", "h2026_pos20", "h2026_delta_pos"], ascending=[True, True, False, False])
    return out.round(6)


def build_p0_small_transfer_channel_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_TRANSFER_CHANNEL_CONFIRM)
    if frame.empty:
        return pd.DataFrame()
    panels = read_csv(P0_SMALL_ENTRY_TRANSFER_CHANNEL_PANEL)
    panel_cols = [
        "frequency",
        "variant",
        "gate_id",
        "panels",
        "selected_rows_mean",
        "selected_pos20_mean±std",
        "selected_avg20_mean±std",
        "selected_loss_gt5_mean±std",
        "delta_pos_vs_transfer_mean±std",
        "delta_avg_vs_transfer_mean±std",
    ]
    if not panels.empty:
        panels = panels[[col for col in panel_cols if col in panels.columns]].copy()
        frame = frame.merge(panels, on=["frequency", "variant", "gate_id"], how="left")

    reference = frame[frame["promotion_status"].astype(str).eq("transfer_reference")].copy()
    yellow = frame[frame["promotion_status"].astype(str).str.contains("yellow|green", regex=True)].copy()
    rejected = frame[frame["promotion_status"].astype(str).str.contains("reject", regex=True)].copy()
    diagnostic = frame[frame["promotion_status"].astype(str).str.contains("observe", regex=True)].copy()

    selected = []
    if not reference.empty:
        selected.append(reference.sort_values(["rank_score"], ascending=False).head(6))
    if not yellow.empty:
        selected.append(
            yellow.sort_values(
                ["rank_score", "h2026_selected_pos20", "h2026_delta_pos_vs_transfer"],
                ascending=[False, False, False],
            )
        )
    if not rejected.empty:
        selected.append(
            rejected.sort_values(
                ["h2026_selected_rows", "h2026_delta_pos_vs_transfer"],
                ascending=[False, True],
            ).head(8)
        )
    if not diagnostic.empty:
        selected.append(diagnostic.sort_values(["rank_score"], ascending=False).head(4))
    subset = pd.concat(selected, ignore_index=True).drop_duplicates(["frequency", "variant", "gate_id"])

    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        status = str(row.get("promotion_status"))
        gate = str(row.get("gate_id"))
        if status == "transfer_reference":
            role = "transfer_reference"
        elif "green" in status or "yellow" in status:
            role = "channel_confirm_candidate"
        elif "reject" in status:
            role = "rejected_gate_ablation"
        else:
            role = "diagnostic_gate_ablation"
        rows.append(
            {
                "task": "P0 small-entry transfer channel confirmation",
                "model": "Local transfer confirmer + non-price channel gates",
                "frequency": row.get("frequency"),
                "config": row.get("variant"),
                "gate": gate,
                "role": role,
                "split": "prior blocks + H2026_1 OOT + 12 H2026 panels",
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "prior_delta_avg_hit": num(row.get("prior_delta_avg_hit")),
                "h2026_transfer_rows": int_or_none(row.get("h2026_transfer_rows")),
                "h2026_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_rate_vs_transfer": num(row.get("h2026_selected_rate_vs_transfer")),
                "h2026_transfer_pos20": num(row.get("h2026_transfer_pos20")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos_vs_transfer": num(row.get("h2026_delta_pos_vs_transfer")),
                "h2026_delta_avg_pp_vs_transfer": num(row.get("h2026_delta_avg_vs_transfer")),
                "panel_pos20_mean_std": row.get("selected_pos20_mean±std"),
                "panel_avg20_pp_mean_std": row.get("selected_avg20_mean±std"),
                "panel_loss_gt5_mean_std": row.get("selected_loss_gt5_mean±std"),
                "panel_delta_pos_mean_std": row.get("delta_pos_vs_transfer_mean±std"),
                "panel_delta_avg_pp_mean_std": row.get("delta_avg_vs_transfer_mean±std"),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_small_transfer_channel_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["role", "h2026_pos20", "h2026_delta_pos_vs_transfer"],
            ascending=[True, False, False],
        )
    return out.round(6)


def build_p0_small_transfer_analog_rag_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_TRANSFER_ANALOG_RAG)
    if frame.empty:
        return pd.DataFrame()
    panels = read_csv(P0_SMALL_ENTRY_TRANSFER_ANALOG_RAG_PANEL)
    panel_cols = [
        "frequency",
        "variant",
        "analog_id",
        "gate_id",
        "panels",
        "selected_rows_mean",
        "selected_pos20_mean±std",
        "selected_avg20_mean±std",
        "selected_loss_gt5_mean±std",
        "delta_pos_vs_transfer_mean±std",
        "delta_avg_vs_transfer_mean±std",
        "analog_pos_rate_mean±std",
    ]
    if not panels.empty:
        panels = panels[[col for col in panel_cols if col in panels.columns]].copy()
        frame = frame.merge(panels, on=["frequency", "variant", "analog_id", "gate_id"], how="left")

    reference = frame[frame["promotion_status"].astype(str).eq("transfer_reference")].copy()
    promoted = frame[frame["promotion_status"].astype(str).str.contains("green|yellow", regex=True)].copy()
    diagnostic = frame[frame["promotion_status"].astype(str).str.contains("observe", regex=True)].copy()
    selected = []
    if not reference.empty:
        selected.append(reference.sort_values(["rank_score"], ascending=False).head(4))
    if not promoted.empty:
        selected.append(
            promoted.sort_values(
                ["promotion_status", "rank_score", "h2026_selected_pos20"],
                ascending=[True, False, False],
            )
        )
    if not diagnostic.empty:
        selected.append(
            diagnostic.sort_values(["rank_score", "h2026_selected_pos20"], ascending=[False, False]).head(8)
        )
    subset = pd.concat(selected, ignore_index=True).drop_duplicates(["frequency", "variant", "analog_id", "gate_id"])

    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        status = str(row.get("promotion_status"))
        if status == "transfer_reference":
            role = "transfer_reference"
        elif "green" in status:
            role = "analog_rag_green_candidate"
        elif "yellow" in status:
            role = "analog_rag_yellow_candidate"
        else:
            role = "diagnostic_analog_ablation"
        rows.append(
            {
                "task": "P0 transfer analog-case/RAG confirmation",
                "model": "Local transfer confirmer + historical analog retrieval",
                "frequency": row.get("frequency"),
                "config": row.get("variant"),
                "analog_id": row.get("analog_id"),
                "gate": row.get("gate_id"),
                "role": role,
                "split": "prior analog bank + H2026_1 OOT + 12 H2026 panels",
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "prior_delta_avg_hit": num(row.get("prior_delta_avg_hit")),
                "h2026_transfer_rows": int_or_none(row.get("h2026_transfer_rows")),
                "h2026_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_rate_vs_transfer": num(row.get("h2026_selected_rate_vs_transfer")),
                "h2026_transfer_pos20": num(row.get("h2026_transfer_pos20")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos_vs_transfer": num(row.get("h2026_delta_pos_vs_transfer")),
                "h2026_delta_avg_pp_vs_transfer": num(row.get("h2026_delta_avg_vs_transfer")),
                "h2026_avg_analog_pos_rate": num(row.get("h2026_avg_analog_pos_rate")),
                "panel_pos20_mean_std": row.get("selected_pos20_mean±std"),
                "panel_avg20_pp_mean_std": row.get("selected_avg20_mean±std"),
                "panel_loss_gt5_mean_std": row.get("selected_loss_gt5_mean±std"),
                "panel_delta_pos_mean_std": row.get("delta_pos_vs_transfer_mean±std"),
                "panel_delta_avg_pp_mean_std": row.get("delta_avg_vs_transfer_mean±std"),
                "panel_analog_pos_rate_mean_std": row.get("analog_pos_rate_mean±std"),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_small_transfer_analog_rag_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["role", "h2026_pos20", "h2026_delta_pos_vs_transfer"],
            ascending=[True, False, False],
        )
    return out.round(6)


def build_p0_transfer_analog_rag_onoff_readiness_table(path: Path, *, model_stage: str) -> pd.DataFrame:
    frame = read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        variant = str(row.get("variant"))
        future_leaks = int_or_none(row.get("future_key_leak_count_sum")) or 0
        evidence_packs = int_or_none(row.get("evidence_packs")) or 0
        rows.append(
            {
                "task": "P0 transfer analog/RAG Flash-Pro on/off readiness",
                "model_stage": model_stage,
                "variant": variant,
                "role": transfer_analog_rag_onoff_role(variant),
                "task_mode": row.get("task_mode"),
                "valid_block": row.get("valid_block"),
                "evidence_packs": evidence_packs,
                "row_level_analog_packs": int_or_none(row.get("row_level_analogue_visible_sum")),
                "analog_visible_packs": int_or_none(row.get("analogue_visible_sum")),
                "chip_visible_packs": int_or_none(row.get("chip_visible_sum")),
                "financial_visible_packs": int_or_none(row.get("financial_visible_sum")),
                "news_visible_packs": int_or_none(row.get("news_visible_sum")),
                "peer_visible_packs": int_or_none(row.get("peer_visible_sum")),
                "bookskill_visible_packs": int_or_none(row.get("bookskill_visible_sum")),
                "quant_tool_visible_packs": int_or_none(row.get("quant_tool_visible_sum")),
                "future_key_leak_count": future_leaks,
                "decision_cards": 0,
                "status": transfer_analog_rag_onoff_status(row, model_stage=model_stage),
                "next_action": transfer_analog_rag_onoff_next_action(row, model_stage=model_stage),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["role", "variant"])
    return out


def build_p0_transfer_analog_rag_panel36_preflight_table() -> pd.DataFrame:
    frame = read_csv(P0_TRANSFER_ANALOG_RAG_PANEL36_PREFLIGHT)
    if frame.empty:
        return pd.DataFrame()
    keep = frame[
        frame["flash_preflight_status"].astype(str).isin(
            {
                "flash_candidate_strong_not_pro_ready",
                "flash_candidate_observe_needs_tighter_panel_or_prior",
                "hold_before_flash_time_generalization_insufficient",
                "reject_before_flash_panel_weak_or_sparse",
            }
        )
    ].copy()
    strong = keep[keep["flash_preflight_status"].astype(str).eq("flash_candidate_strong_not_pro_ready")]
    observe = keep[keep["flash_preflight_status"].astype(str).eq("flash_candidate_observe_needs_tighter_panel_or_prior")]
    hold = keep[keep["flash_preflight_status"].astype(str).eq("hold_before_flash_time_generalization_insufficient")]
    reject = keep[keep["flash_preflight_status"].astype(str).eq("reject_before_flash_panel_weak_or_sparse")]
    parts = []
    if not strong.empty:
        parts.append(strong.sort_values(["panel_selected_pos20_mean", "panel_delta_pos_vs_transfer_mean"], ascending=[False, False]))
    if not observe.empty:
        parts.append(observe.sort_values(["panel_selected_pos20_mean", "panel_delta_pos_vs_transfer_mean"], ascending=[False, False]).head(8))
    if not hold.empty:
        parts.append(hold.sort_values(["panel_selected_pos20_mean", "panel_delta_pos_vs_transfer_mean"], ascending=[False, False]).head(8))
    if not reject.empty:
        parts.append(reject.sort_values(["panel_selected_pos20_mean", "panel_delta_pos_vs_transfer_mean"], ascending=[False, False]).head(4))
    subset = pd.concat(parts, ignore_index=True) if parts else keep.head(0).copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "task": "P0 transfer analog/RAG 36-panel Flash preflight",
                "model_stage": "local_36_panel_no_deepseek",
                "frequency": row.get("frequency"),
                "config": row.get("variant"),
                "analog_id": row.get("analog_id"),
                "gate": row.get("gate_id"),
                "role": panel36_role(str(row.get("flash_preflight_status"))),
                "local_status": row.get("local_promotion_status"),
                "prior_evaluable_blocks": int_or_none(row.get("prior_evaluable_blocks")),
                "prior_evaluable_rows_mean": num(row.get("prior_evaluable_selected_rows_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "prior_delta_avg_hit": num(row.get("prior_delta_avg_hit")),
                "prior_evaluable_delta_pos_hit": num(row.get("prior_evaluable_delta_pos_hit")),
                "prior_evaluable_delta_avg_hit": num(row.get("prior_evaluable_delta_avg_hit")),
                "panels": int_or_none(row.get("panels")),
                "panel_rows_mean": num(row.get("panel_selected_rows_mean")),
                "panel_pos20_mean": num(row.get("panel_selected_pos20_mean")),
                "panel_pos20_p10": num(row.get("panel_selected_pos20_p10")),
                "panel_avg20_pp_mean": num(row.get("panel_selected_avg20_mean")),
                "panel_avg20_pp_p10": num(row.get("panel_selected_avg20_p10")),
                "panel_loss_gt5_mean": num(row.get("panel_selected_loss_gt5_mean")),
                "panel_loss_gt5_p90": num(row.get("panel_selected_loss_gt5_p90")),
                "panel_delta_pos_vs_transfer_mean": num(row.get("panel_delta_pos_vs_transfer_mean")),
                "panel_delta_pos_vs_transfer_p10": num(row.get("panel_delta_pos_vs_transfer_p10")),
                "panel_delta_avg_pp_vs_transfer_p10": num(row.get("panel_delta_avg_vs_transfer_p10")),
                "flash_preflight_status": row.get("flash_preflight_status"),
                "pro_status": row.get("pro_status"),
                "main_read": row.get("main_read"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["role", "panel_pos20_mean", "panel_delta_pos_vs_transfer_mean"], ascending=[True, False, False])
    return out.round(6)


def build_p0_small_case_memory_table() -> pd.DataFrame:
    sources = [
        ("case_memory_min_conditions_2", 2, P0_SMALL_ENTRY_CASE_MEMORY_V2),
        ("case_memory_min_conditions_1_diagnostic", 1, P0_SMALL_ENTRY_CASE_MEMORY_MIN1),
    ]
    keep_rules = {
        "no_case_guard",
        "applicable_any",
        "risk_condition_ge1",
        "risk_condition_ge2",
        "condition_financial_report_context",
        "condition_news_hidden_or_missing",
        "condition_weak_peer_confirmation",
        "condition_financial_or_news",
    }
    rows: list[dict[str, Any]] = []
    for source, min_conditions, path in sources:
        frame = read_csv(path)
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            guard = str(row.get("guard_policy"))
            if guard not in keep_rules:
                continue
            rows.append(
                {
                    "task": "P0 small-entry case-memory/RAG ablation",
                    "model": "Local case-memory retriever",
                    "source": source,
                    "min_applicable_conditions": min_conditions,
                    "frequency": row.get("frequency"),
                    "config": guard,
                    "role": "branch_reference" if guard == "no_case_guard" else "rag_guard_ablation",
                    "split": "walk-forward prior blocks + H2026_1 OOT",
                    "prior_blocks": int_or_none(row.get("prior_blocks")),
                    "prior_retained_rows_mean": num(row.get("prior_retained_rows_mean")),
                    "h2026_total_rows": int_or_none(row.get("h2026_total_rows")),
                    "h2026_retained_rows": int_or_none(row.get("h2026_retained_rows")),
                    "h2026_retained_rate": num(row.get("h2026_retained_rate")),
                    "h2026_pos20": num(row.get("h2026_retained_pos20")),
                    "h2026_avg20_pp": num(row.get("h2026_retained_avg20_pp")),
                    "h2026_loss_gt5": num(row.get("h2026_retained_loss_gt5")),
                    "h2026_false_veto_positive_rows": int_or_none(row.get("h2026_false_veto_positive_rows")),
                    "h2026_captured_loss_gt5_rows": int_or_none(row.get("h2026_captured_loss_gt5_rows")),
                    "h2026_delta_pos": num(row.get("h2026_delta_pos")),
                    "promotion_status": row.get("promotion_status"),
                    "verdict": p0_small_case_memory_verdict(row, min_conditions),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["min_applicable_conditions", "frequency", "role", "config"])
    return out.round(6)


def build_p0_small_bookskill_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_BOOKSKILL)
    if frame.empty:
        return pd.DataFrame()
    keep_rules = {
        "small_entry_all",
        "skill_PPS_Q_017",
        "skill_PPS_M_003",
        "skill_DOW_B_004",
        "skill_PPS_Q_019",
        "skill_PPS_Q_023_needs_grounding",
        "grounded_bookskill_any",
        "not_bookskill_gap",
        "bookskill_gap_any",
        "positive_historical_skill_any",
    }
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rule = str(row.get("rule_id"))
        if rule not in keep_rules:
            continue
        rows.append(
            {
                "task": "P0 small-entry BookSkill attribution",
                "model": "Local resolver-grounded BookSkill audit",
                "frequency": row.get("frequency"),
                "config": rule,
                "role": "branch_reference" if rule == "small_entry_all" else "bookskill_attribution",
                "split": "walk-forward prior blocks + H2026_1 OOT",
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "prior_delta_pos_mean": num(row.get("prior_delta_pos_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "h2026_total_rows": int_or_none(row.get("h2026_total_rows")),
                "h2026_selected_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_selected_rate": num(row.get("h2026_selected_rate")),
                "h2026_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_avg20_pp": num(row.get("h2026_selected_avg20_pp")),
                "h2026_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_false_veto_positive_rows": int_or_none(row.get("h2026_false_veto_positive_rows")),
                "h2026_captured_loss_gt5_rows": int_or_none(row.get("h2026_captured_loss_gt5_rows")),
                "h2026_delta_pos": num(row.get("h2026_delta_pos")),
                "promotion_status": row.get("promotion_status"),
                "verdict": p0_small_bookskill_verdict(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["frequency", "role", "h2026_delta_pos"], ascending=[True, True, False])
    return out.round(6)


def build_p0_pps_q017_onoff_table() -> pd.DataFrame:
    evidence = read_csv(P0_SMALL_ENTRY_PPS_Q017_ONOFF_EVIDENCE)
    variants = read_csv(P0_SMALL_ENTRY_PPS_Q017_ONOFF_VARIANT)
    pairs = read_csv(P0_SMALL_ENTRY_PPS_Q017_ONOFF_PAIR)
    if evidence.empty and variants.empty and pairs.empty:
        return pd.DataFrame()

    variant_lookup = {}
    if not variants.empty and "variant" in variants:
        variant_lookup = {str(row.get("variant")): row for _, row in variants.iterrows()}

    rows: list[dict[str, Any]] = []
    if not evidence.empty:
        for _, row in evidence.iterrows():
            variant = str(row.get("variant"))
            variant_row = variant_lookup.get(variant)
            cards = int_or_none(variant_row.get("cards")) if variant_row is not None else 0
            status = variant_row.get("status") if variant_row is not None else "dryrun_safe_no_decision_cards"
            rows.append(
                {
                    "task": "P0 PPS-Q-017 on/off readiness",
                    "model": "DS Flash/Pro planned shard",
                    "config": variant,
                    "role": pps_q017_onoff_role(variant),
                    "evidence_packs": int_or_none(row.get("evidence_packs")),
                    "pps_q017_visible_packs": int_or_none(row.get("pps_q017_visible_packs")),
                    "avg_bookskill_cards": num(row.get("avg_bookskill_cards")),
                    "future_key_leak_count": int_or_none(row.get("future_key_leak_count")),
                    "hidden_strategy_ids": row.get("hidden_strategy_ids"),
                    "decision_cards": cards,
                    "status": status,
                    "main_read": pps_q017_onoff_read(variant, row, cards),
                }
            )

    if not pairs.empty:
        for _, row in pairs.iterrows():
            rows.append(
                {
                    "task": "P0 PPS-Q-017 on/off paired result",
                    "model": "DS Flash/Pro planned shard",
                    "config": row.get("comparison"),
                    "role": "paired_result",
                    "evidence_packs": int_or_none(row.get("paired_rows")),
                    "pps_q017_visible_packs": None,
                    "avg_bookskill_cards": None,
                    "future_key_leak_count": None,
                    "hidden_strategy_ids": "",
                    "decision_cards": int_or_none(row.get("paired_rows")),
                    "status": row.get("verdict"),
                    "main_read": "available after DS on/off cards are run",
                }
            )
    return pd.DataFrame(rows).round(6)


def build_p0_pps_q017_interaction_table() -> pd.DataFrame:
    frame = read_csv(P0_SMALL_ENTRY_PPS_Q017_INTERACTIONS)
    if frame.empty:
        return pd.DataFrame()
    keep_rules = {
        "financial_no_recent_event",
        "chip_support_visible",
        "kline_deep_pullback",
        "news_available",
        "news_low_warning",
        "peer_relative_positive",
        "financial_event_matched",
        "chip_low_overhang",
        "weak_skill_present",
    }
    subset = frame[frame["rule_id"].astype(str).isin(keep_rules)].copy()
    if subset.empty:
        subset = frame.head(12).copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "task": "P0 PPS-Q-017 channel interaction",
                "model": "Local offline rule audit",
                "config": row.get("rule_id"),
                "role": pps_q017_interaction_role(str(row.get("verdict"))),
                "scope": row.get("scope"),
                "rows": int_or_none(row.get("rows")),
                "selected_rate": num(row.get("selected_rate")),
                "pos20": num(row.get("pos20")),
                "avg20_pp": num(row.get("avg20_pp")),
                "loss_gt5": num(row.get("loss_gt5")),
                "delta_pos": num(row.get("delta_pos")),
                "delta_avg_pp": num(row.get("delta_avg_pp")),
                "unique_stocks": int_or_none(row.get("unique_stocks")),
                "verdict": row.get("verdict"),
                "main_read": pps_q017_interaction_read(str(row.get("rule_id")), str(row.get("verdict"))),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["role", "delta_pos", "delta_avg_pp"], ascending=[True, False, False])
    return out.round(6)


def build_p0_kline_threshold_table() -> pd.DataFrame:
    frame = read_csv(P0_KLINE_THRESHOLD_COMPARE)
    if frame.empty:
        return pd.DataFrame()
    keep_rows = (
        frame["feature_group"].astype(str).isin({"rev_chip_core_fixed", "peer_kline_learned", "chip_core_learned"})
        | frame["threshold_verdict"].astype(str).isin(
            {"narrow_threshold_candidate", "narrow_threshold_mean_only_diagnostic", "keep_wider_threshold"}
        )
    )
    subset = frame[keep_rows].copy()
    if subset.empty:
        subset = frame.head(20).copy()
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        rows.append(
            {
                "task": "P0 K-line threshold ablation",
                "model": "Local K-line/peer checklist",
                "config": f"{row.get('decision_frequency')}::{row.get('feature_group')}",
                "role": str(row.get("threshold_verdict")),
                "h2026_pos_top10": num(row.get("h2026_opp_delta_pos_top10")),
                "h2026_pos_top05": num(row.get("h2026_opp_delta_pos_top05")),
                "h2026_mean_top10_pp": num(row.get("h2026_opp_delta_mean_top10")),
                "h2026_mean_top05_pp": num(row.get("h2026_opp_delta_mean_top05")),
                "h2026_risk_recall_top10": num(row.get("h2026_risk_recall_top10")),
                "h2026_risk_recall_top05": num(row.get("h2026_risk_recall_top05")),
                "delta_pos_top05_minus_top10": num(row.get("delta_h2026_opp_delta_pos_top05_minus_top10")),
                "delta_mean_top05_minus_top10": num(row.get("delta_h2026_opp_delta_mean_top05_minus_top10")),
                "delta_risk_recall_top05_minus_top10": num(row.get("delta_h2026_risk_recall_top05_minus_top10")),
                "verdict": row.get("threshold_verdict"),
                "main_read": p0_kline_threshold_read(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["verdict", "h2026_pos_top10", "h2026_mean_top10_pp"],
            ascending=[True, False, False],
        )
    return out.round(6)


def build_p0_active_entry_calibration_table() -> pd.DataFrame:
    frame = read_csv(P0_ACTIVE_ENTRY_CALIBRATION)
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    top = frame.sort_values(
        ["task_mode", "h2026_active_pos20", "h2026_active_avg20"],
        ascending=[True, False, False],
    ).groupby("task_mode", dropna=False).head(4)
    for _, row in top.iterrows():
        variant = str(row.get("variant"))
        rows.append(
            {
                "task": "P0/P1 active-entry calibration",
                "model": "Local threshold grid",
                "task_mode": row.get("task_mode"),
                "config": variant,
                "role": "anchor" if variant == "base_v4_like" else "threshold_ablation",
                "decision_frequency": row.get("decision_frequency"),
                "prior_active_pos20_mean": num(row.get("prior_active_pos20_mean")),
                "prior_active_avg20_pp_mean": num(row.get("prior_active_avg20_mean")),
                "prior_active_count_mean": num(row.get("prior_active_count_mean")),
                "h2026_active_pos20": num(row.get("h2026_active_pos20")),
                "h2026_active_avg20_pp": num(row.get("h2026_active_avg20")),
                "h2026_active_count_mean": num(row.get("h2026_active_count_mean")),
                "h2026_active_rate": num(row.get("h2026_active_rate")),
                "h2026_strategy_avg20_pp": num(row.get("h2026_strategy_avg20")),
                "h2026_hold_avg20_pp": num(row.get("h2026_hold_avg20")),
                "promotion_status": row.get("promotion_status"),
                "sparse_cap": num(row.get("sparse_cap")),
                "require_support_for_buy": row.get("require_support_for_buy"),
                "require_peer_for_new_entry": row.get("require_peer_for_new_entry"),
                "main_read": p0_active_entry_calibration_read(row),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task_mode", "h2026_active_pos20", "h2026_active_avg20_pp"],
            ascending=[True, False, False],
        )
    return out.round(6)


def build_p0_component_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, path in P0_COMPONENT_SUMMARIES.items():
        frame = read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0]
        delta = num(row.get("sum_delta_cash_adjusted_return_20d"))
        harmful = num(row.get("harmful_delta"))
        rows.append(
            {
                "task": "P0 single-stock watch",
                "model": "DS V4 Flash",
                "component": component,
                "treatment": "full_agent_with_opportunity_tool",
                "control": component_control(component),
                "paired_rows": int_or_none(row.get("paired_rows")),
                "changed_rows": int_or_none(row.get("changed_rows")),
                "raised_positive": int_or_none(row.get("raised_positive")),
                "raised_negative": int_or_none(row.get("raised_negative")),
                "lowered_positive": int_or_none(row.get("lowered_positive")),
                "lowered_negative": int_or_none(row.get("lowered_negative")),
                "delta_cash20_sum_pp": delta,
                "delta_cash20_mean_pp": num(row.get("mean_delta_cash_adjusted_return_20d")),
                "useful_delta_pp": num(row.get("useful_delta")),
                "harmful_delta_pp": harmful,
                "verdict": component_verdict(delta, harmful),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_cash20_sum_pp", ascending=False)
    return out.round(6)


def build_p1_candidate_table() -> pd.DataFrame:
    frame = read_csv(P1_ABLATION)
    if frame.empty:
        return pd.DataFrame()
    keep_cols = [
        "model",
        "variant",
        "comparison_scenario",
        "cards",
        "top1_excess_mean",
        "top2_excess_mean",
        "top1_positive_rate",
        "top2_positive_rate",
        "top1_worst_rate",
        "regret_mean",
        "delta_top1_excess_vs_anchor",
        "delta_top2_excess_vs_anchor",
        "avg_confidence",
    ]
    out = frame[[col for col in keep_cols if col in frame]].copy()
    out.insert(0, "task", "P1 candidate comparison")
    out.insert(4, "split", "14 groups, panel_index=1")
    out["role"] = out["variant"].astype(str).map(lambda item: "anchor" if item == "ranker_anchor_agent" else "ablation")
    out["main_read"] = [
        p1_read(str(row.model), str(row.variant), str(row.comparison_scenario), row)
        for row in out.itertuples(index=False)
    ]
    return out.round(6)


def build_p1_operation_confirm_table() -> pd.DataFrame:
    frame = read_csv(P1_OPERATION_CONFIRM)
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.insert(0, "task", "P1 candidate comparison user path")
    out.insert(1, "model", "DS V4 Pro")
    out.insert(5, "split", "6 half-year blocks, 7 groups per scenario")
    out["role"] = "operation_confirm"
    out["actionable_rows"] = out["cards"].apply(lambda value: int_or_none(value) or 0) * 8
    out["vague_threshold_rows"] = 0
    out["verdict"] = out.apply(p1_operation_verdict, axis=1)
    return out.round(6)


def build_cross_sector_stress_table() -> pd.DataFrame:
    frame = read_csv(CROSS_SECTOR_STRESS)
    if frame.empty:
        return pd.DataFrame()
    keep_cols = [
        "model",
        "variant",
        "panels",
        "cards",
        "top1_excess_mean",
        "top1_excess_std",
        "top2_excess_mean",
        "top2_excess_std",
        "top1_positive_rate_mean",
        "top1_positive_rate_std",
        "top2_positive_rate_mean",
        "top2_positive_rate_std",
        "top1_worst_rate_mean",
        "regret_mean",
        "avg_confidence_mean",
    ]
    out = frame[[col for col in keep_cols if col in frame]].copy()
    out.insert(0, "task", "P1 cross-sector stress")
    out.insert(3, "split", "3 hash seeds x 21 groups")
    out["role"] = out["variant"].astype(str).map(lambda item: "anchor" if item == "ranker_anchor_agent" else "ablation")
    out["main_read"] = out.apply(cross_stress_read, axis=1)
    return out.round(6)


def build_hygiene_table(deepseek_status: str) -> pd.DataFrame:
    rows = [
        {
            "scope": "DeepSeek current smoke",
            "status": deepseek_status,
            "valid_cards": None,
            "invalid_cards": None,
            "tokens": None,
            "source": "scripts/deepseek_smoke_test.py",
        },
        {
            "scope": "P0 Flash acceptance",
            "status": "completed",
            "valid_cards": 360,
            "invalid_cards": 0,
            "tokens": usage_tokens(REPORT_DIR / "p0_acceptance_multiblock_3panel_flash_v1_usage_summary.csv"),
            "source": "p0_acceptance_multiblock_3panel_flash_v1",
        },
        {
            "scope": "P0 Pro default confirmation",
            "status": "completed",
            "valid_cards": 36,
            "invalid_cards": 0,
            "tokens": usage_tokens(REPORT_DIR / "p0_acceptance_single_default_pro_v1_usage_summary.csv"),
            "source": "p0_acceptance_single_default_pro_v1",
        },
        {
            "scope": "P0 latest single-stock user path Flash",
            "status": "completed",
            "valid_cards": 120,
            "invalid_cards": 0,
            "tokens": usage_tokens(REPORT_DIR / "single_stock_branch_guardrail_panel24_flash_v1_usage_summary.csv"),
            "source": "single_stock_branch_guardrail_panel24_flash_v1",
        },
        {
            "scope": "P0 local operation-policy audit",
            "status": "completed_no_deepseek",
            "valid_cards": row_count(P0_OPERATION_PANEL),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_operation_policy_v1",
        },
        {
            "scope": "P0 local small-entry overlay ablation",
            "status": "completed_no_deepseek",
            "valid_cards": row_count(P0_SMALL_ENTRY_OVERLAY),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_small_entry_nonprice_overlay_v1",
        },
        {
            "scope": "P0 local small-entry ML confirmer",
            "status": "completed_no_deepseek",
            "valid_cards": row_count(P0_SMALL_ENTRY_ML_CONFIRMER),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_small_entry_ml_confirmer_v1",
        },
        {
            "scope": "P0 local small-entry transfer confirmer",
            "status": "completed_no_deepseek_yellow_only",
            "valid_cards": row_count(P0_SMALL_ENTRY_TRANSFER_CONFIRMER),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_small_entry_transfer_confirmer_v1",
        },
        {
            "scope": "P0 local transfer channel confirmation",
            "status": "completed_no_deepseek_yellow_only",
            "valid_cards": row_count(P0_SMALL_ENTRY_TRANSFER_CHANNEL_CONFIRM),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_transfer_channel_confirm_v1",
        },
        {
            "scope": "P0 local transfer analog-case/RAG confirmation",
            "status": "completed_no_deepseek_green_candidates_need_ds",
            "valid_cards": row_count(P0_SMALL_ENTRY_TRANSFER_ANALOG_RAG),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_transfer_analog_rag_v1",
        },
        {
            "scope": "P0 transfer analog/RAG on/off dry-run",
            "status": "completed_no_deepseek_flash_pro_ready",
            "valid_cards": visibility_evidence_pack_count(P0_TRANSFER_ANALOG_RAG_ONOFF_VISIBILITY),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_transfer_analog_rag_onoff_dryrun_v1",
        },
        {
            "scope": "P0 transfer analog/RAG + Kline quant-tool on/off dry-run",
            "status": "completed_no_deepseek_flash_pro_ready",
            "valid_cards": visibility_evidence_pack_count(P0_TRANSFER_ANALOG_RAG_KLINE_TOOL_ONOFF_VISIBILITY),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_transfer_analog_rag_kline_tool_onoff_dryrun_v1",
        },
        {
            "scope": "P0 transfer analog/RAG 36-panel Flash preflight",
            "status": "completed_no_deepseek_flash_shortlist_gate",
            "valid_cards": row_count(P0_TRANSFER_ANALOG_RAG_PANEL36_PREFLIGHT),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_transfer_analog_rag_panel36_preflight_gate_v1",
        },
        {
            "scope": "P0 local small-entry case-memory/RAG ablation",
            "status": "completed_no_deepseek",
            "valid_cards": (row_count(P0_SMALL_ENTRY_CASE_MEMORY_V2) or 0)
            + (row_count(P0_SMALL_ENTRY_CASE_MEMORY_MIN1) or 0),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_small_entry_case_memory_v2 + p0_small_entry_case_memory_min1_v1",
        },
        {
            "scope": "P0 local small-entry BookSkill attribution",
            "status": "completed_no_deepseek",
            "valid_cards": row_count(P0_SMALL_ENTRY_BOOKSKILL),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "p0_small_entry_bookskill_attribution_v1",
        },
        {
            "scope": "P0/P1 active-entry calibration",
            "status": active_entry_calibration_status(),
            "valid_cards": row_count(P0_ACTIVE_ENTRY_CALIBRATION),
            "invalid_cards": 0,
            "tokens": 0,
            "source": "userpath_active_entry_calibration_v1",
        },
        {
            "scope": "P1 Pro operation confirmation",
            "status": "completed",
            "valid_cards": 14,
            "invalid_cards": 0,
            "tokens": usage_tokens(REPORT_DIR / "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1_usage_summary.csv"),
            "source": "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1",
        },
    ]
    hygiene = read_csv(CROSS_SECTOR_HYGIENE)
    if not hygiene.empty:
        for _, row in hygiene.iterrows():
            rows.append(
                {
                    "scope": f"Cross-sector stress {row.get('model')} seed={row.get('seed')}",
                    "status": "completed",
                    "valid_cards": int_or_none(row.get("merged_cards")),
                    "invalid_cards": int_or_none(row.get("final_missing_cards")),
                    "tokens": int_or_none(row.get("total_tokens")),
                    "source": "cross_sector_ridge_flash_pro_ablation_k03_v1",
                }
            )
    return pd.DataFrame(rows)


def render_report(
    *,
    p0_local: pd.DataFrame,
    p0_ds: pd.DataFrame,
    p0_user_path: pd.DataFrame,
    p0_operation: pd.DataFrame,
    p0_small_overlay: pd.DataFrame,
    p0_small_ml: pd.DataFrame,
    p0_small_transfer: pd.DataFrame,
    p0_small_transfer_channel: pd.DataFrame,
    p0_small_transfer_analog_rag: pd.DataFrame,
    p0_transfer_analog_rag_onoff: pd.DataFrame,
    p0_transfer_analog_rag_kline_tool_onoff: pd.DataFrame,
    p0_transfer_analog_rag_panel36_preflight: pd.DataFrame,
    p0_small_case_memory: pd.DataFrame,
    p0_small_bookskill: pd.DataFrame,
    p0_pps_q017_onoff: pd.DataFrame,
    p0_pps_q017_interactions: pd.DataFrame,
    p0_kline_threshold: pd.DataFrame,
    p0_active_entry_calibration: pd.DataFrame,
    p0_components: pd.DataFrame,
    p1_candidate: pd.DataFrame,
    p1_operation_confirm: pd.DataFrame,
    cross_stress: pd.DataFrame,
    hygiene: pd.DataFrame,
    outputs: dict[str, Path],
) -> str:
    lines = [
        "# Flash/Pro ICML-Style Capability Tables",
        "",
        "本报告把已经真实完成的 Flash/Pro 调用、P0 最新本地 24-panel 验证、P1 候选对比和关键消融放进统一实验表。数值只表示历史回测/离线评估表现，不承诺未来收益。",
        "",
        "## Executive Summary",
        "",
        "- **P0 单支盯盘**：最新本地周五 `opp_kline_confirm_no_raise` 分支在 H2026 24 个 100 股 panel 上 active_pos20 约 `0.53±0.03`、active_avg20 约 `+1.89pp`，比 `opp_only` 明显提升，但仍低于 0.60/0.65 验收线。",
        "- **P0 Flash/Pro Agent**：已完成 Flash 360 卡 ablation 与 Pro 36 卡确认。Pro 默认路径正收益率与 Flash 同为 `0.75`，但 H2026_1 仍只有 `0.50`，说明更强模型没有解决最新块泛化。",
        "- **P0 用户路径补充**：最新 H2026 单支盯盘 Flash 24-card 面板能稳定输出结构化建议，但 `exposure_cards=0`，仍偏防守；`quant_tool_summary_only` 的 cash_pos20 达 `0.6667`，更像定量工具提示有效，不等于 Agent 已会主动买入。",
        "- **P0 动作分叉**：本地用户动作回测显示，`branch_stack_v1` 的直接 `买入/加仓` 分叉仍弱，而 `小仓试探/持有` 在 H2026 周五 12-panel 达约 `0.6138±0.0603` 正收益率、`+4.6628±1.5208pp` 20日均值，是下一轮 DS 语义确认的主要候选。",
        "- **P0 K线阈值消融**：新增 top10 vs top05 阈值实验。收紧到 top 5% 没有形成 `narrow_threshold_candidate`，38/56 行应保持较宽 top10；top05 平均使 H2026 risk recall 下降约 0.05–0.06，因此只作高置信诊断，不作默认操作阈值。",
        "- **P0/P1 active-entry calibration**：96 个本地阈值配置 + 2 个 stateful replay 候选均未通过 green/yellow。P0 最好仍是 `base_v4_like/every_2_weeks`，H2026 active_pos20 约 `0.428`、active_avg20 约 `-0.305pp`；P1 最好阈值版 H2026 active_pos20 约 `0.449`、active_avg20 约 `+0.007pp`。结论是不要继续机械拧阈值，必须补正向新闻/公告/财报/同行确认或更强 tool。",
        "- **P0 非价格 overlay 消融**：现有新闻/财报/同行/筹码 overlay 没有稳定改善小仓分叉；`news_low_warning` 只能作为安全确认，财报缺失必须作为数据缺口而不是低风险证据。",
        "- **P0 小仓 ML confirmer**：本地 logistic confirmer 在 H2026 有若干高分切片，但 prior block/selected-row 支撑不足；收紧晋级门槛后 0 个 green/yellow，因此只能作为灰色上下文，不能硬过滤或升默认。",
        "- **P0 小仓 transfer confirmer**：用更大的历史候选池训练后，H2026 双周小仓切片可达约 `0.75-0.82` pos20、`+3.1pp` 以上 delta，但只有 0 个 green、9 个 yellow；它证明“ML tool 可提纯小仓分叉”，也证明必须补 fresh panel 和 DS Flash/Pro 语义确认，不能直接升默认。",
        "- **P0 transfer + 通道确认**：在 yellow transfer 候选上叠加新闻/财报/筹码等预注册 gate 后，出现 0 个 green、27 个 yellow；`news_financial_clean_no_hard` 与 `chip_support_no_overheat_no_hard` 在 H2026 panel 上有稳定增益，但 prior hit 只有半数级别，下一步应跑 DS Flash/Pro on/off，而不是直接上线。",
        "- **P0 transfer + analog/RAG**：把历史已验证小仓案例作为相似案例库后，早期 12-panel 曾出现 2 个 green 候选；但在 36-panel 严格复核中，所有候选的 prior evaluable blocks 只有 1 个，不满足时间泛化支撑。结论降级为“最新块亮、暂缓 Flash/Pro”，先补跨时间块证据。",
        "- **P0 transfer + analog/RAG on/off readiness**：已生成 24 个 H2026 单支盯盘样本 × 10 个 variant 的 dry-run evidence；row-level analog 只在 treatment/非 analog 消融中可见，`no_analogue/no_chip/no_financial/no_news/no_peer/no_bookskill` 隔离通过，future leak=0。基础 shard 的 `quant_tool_summaries=0`，所以又补跑了注入 `single_stock_kline_frequency_tool_v1` 的 dry-run：full/no_analogue 等 24/24 可见 quant checklist，`no_quant_tools` 0/24，`quant_tool_summary_only` 24/24。DS 当前 402，恢复后先跑 Flash，再用 Pro 确认通过项。",
        "- **P0 36-panel Flash 预检**：修正 prior block 统计后，旧的 2 个强 Flash 候选全部降级；124 行均为 `hold_before_flash_time_generalization_insufficient`。它们的 H2026 panel 指标仍亮，但 prior 可评估时间块不足，当前不应消耗 Flash/Pro token。",
        "- **P0 小仓 case-memory/RAG**：严格 `min_applicable_conditions=2` 完全不触发；宽松到 1 时只匹配 `bookskill_missing_or_weak` 并全量误杀小仓机会，因此当前 RAG 不能作为小仓 guard，只能说明 BookSkill/记忆标签还不够分支化。",
        "- **P0 小仓 BookSkill attribution**：`PPS-Q-017` 在周五/双周/周二小仓分叉均呈正向候选，但 false-veto 成本偏高，暂作 DS on/off 语义确认候选；`PPS-M-003` 只在周二强，`PPS-Q-023` 因 grounding 不足不能晋级；BookSkill 缺口不是硬 veto，也不是 alpha。",
        "- **PPS-Q-017 细化审计**：on/off 样本已通过 evidence 隔离；`no_pps_q017` 只隐藏该策略卡且 future leak=0。离线交互显示 `financial_no_recent_event`、`chip_support_visible` 值得进入 DS prompt 检查，但不能机械升权。",
        "- **关键组件**：Peer、Risk Review Queue、BookSkill、Branch/RAG Case 在 36 点 paired 里有正 delta；News、Quant Tools、Opportunity Tool 本轮带来错升/错杀成本，暂不升权。",
        "- **P1 候选对比**：Pro 操作确认面板 14 组、112 个候选操作建议全部可执行且无模糊阈值；同领域/跨领域 Top1/Top2 excess 为正，但样本仍小，需三次 fresh panel 复核。",
        "- **当前 DS 状态**：2026-06-29 smoke 复测仍返回 402 Payment Required，因此本报告没有新增大规模 Flash/Pro 调用；新调用恢复后应先跑小规模 aligned smoke，再扩三次采样。",
        "",
        "## Table 1. P0 Local H2026 24-Panel Stack",
        "",
        markdown_table(
            p0_local,
            [
                "config",
                "role",
                "panels",
                "active_rate",
                "active_pos20_mean",
                "active_pos20_std",
                "active_avg20_pp_mean",
                "active_avg20_pp_std",
                "strategy_avg20_pp_mean",
                "loss_gt5_rate_mean",
                "promotion_status",
            ],
        ),
        "",
        "## Table 2. P0 Flash/Pro Single-Stock Agent Ablation",
        "",
        markdown_table(
            p0_ds,
            [
                "model",
                "config",
                "role",
                "cards",
                "invalid",
                "cash_pos20",
                "cash_avg20_pp",
                "cash_std20_pp",
                "active_exposure",
                "exposure_cards",
                "data_missing_cards",
            ],
        ),
        "",
        "## Table 3. P0 Latest User-Path Flash Panel",
        "",
        markdown_table(
            p0_user_path,
            [
                "config",
                "role",
                "split",
                "cards",
                "invalid",
                "cash_pos20",
                "cash_avg20_pp",
                "cash_std20_pp",
                "active_exposure",
                "exposure_cards",
                "data_missing_cards",
                "verdict",
            ],
        ),
        "",
        "## Table 4. P0 Local User-Action Policy on H2026 Panels",
        "",
        markdown_table(
            p0_operation,
            [
                "frequency",
                "config",
                "role",
                "panels",
                "cash_pos20_mean_std",
                "cash_avg20_pp_mean_std",
                "buy_add_pos20_mean_std",
                "buy_add_avg20_pp_mean_std",
                "small_entry_pos20_mean_std",
                "small_entry_avg20_pp_mean_std",
                "reduce_correct_mean_std",
                "false_large_gain_mean_std",
                "main_read",
            ],
        ),
        "",
        "## Table 5. P0 K-Line/Peer Threshold Ablation",
        "",
        markdown_table(
            p0_kline_threshold,
            [
                "config",
                "verdict",
                "h2026_pos_top10",
                "h2026_pos_top05",
                "h2026_mean_top10_pp",
                "h2026_mean_top05_pp",
                "h2026_risk_recall_top10",
                "h2026_risk_recall_top05",
                "delta_pos_top05_minus_top10",
                "delta_risk_recall_top05_minus_top10",
                "main_read",
            ],
        ),
        "",
        "## Table 6. P0/P1 Active-Entry Threshold Calibration",
        "",
        markdown_table(
            p0_active_entry_calibration,
            [
                "task_mode",
                "config",
                "role",
                "decision_frequency",
                "prior_active_pos20_mean",
                "prior_active_avg20_pp_mean",
                "h2026_active_pos20",
                "h2026_active_avg20_pp",
                "h2026_active_rate",
                "h2026_strategy_avg20_pp",
                "h2026_hold_avg20_pp",
                "promotion_status",
                "main_read",
            ],
        ),
        "",
        "## Table 7. P0 Small-Entry Non-Price Overlay Ablation",
        "",
        markdown_table(
            p0_small_overlay,
            [
                "frequency",
                "config",
                "role",
                "prior_pos20_mean",
                "prior_avg20_pp_mean",
                "h2026_rows",
                "h2026_rate_vs_branch",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_loss_gt5",
                "h2026_delta_pos",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 8. P0 Small-Entry ML Confirmer Diagnostic",
        "",
        markdown_table(
            p0_small_ml,
            [
                "frequency",
                "config",
                "role",
                "prior_blocks",
                "prior_selected_rows_mean",
                "h2026_rows",
                "h2026_rate_vs_branch",
                "h2026_base_pos20",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_loss_gt5",
                "h2026_delta_pos",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 9. P0 Small-Entry Transfer Confirmer Diagnostic",
        "",
        markdown_table(
            p0_small_transfer,
            [
                "frequency",
                "config",
                "role",
                "cohort",
                "feature_set",
                "confirm_quantile",
                "prior_blocks",
                "prior_selected_rows_mean",
                "prior_delta_pos_mean",
                "prior_delta_avg_mean",
                "prior_delta_pos_hit",
                "h2026_base_rows",
                "h2026_rows",
                "h2026_rate_vs_branch",
                "h2026_base_pos20",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_delta_pos",
                "h2026_delta_avg_pp",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 10. P0 Small-Entry Transfer Channel Confirmation",
        "",
        markdown_table(
            p0_small_transfer_channel,
            [
                "frequency",
                "config",
                "gate",
                "role",
                "prior_blocks",
                "prior_selected_rows_mean",
                "prior_delta_pos_hit",
                "h2026_transfer_rows",
                "h2026_rows",
                "h2026_rate_vs_transfer",
                "h2026_transfer_pos20",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_delta_pos_vs_transfer",
                "h2026_delta_avg_pp_vs_transfer",
                "panel_pos20_mean_std",
                "panel_avg20_pp_mean_std",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 11. P0 Transfer Analog-Case/RAG Confirmation",
        "",
        markdown_table(
            p0_small_transfer_analog_rag,
            [
                "frequency",
                "config",
                "analog_id",
                "gate",
                "role",
                "prior_blocks",
                "prior_selected_rows_mean",
                "prior_delta_pos_hit",
                "prior_delta_avg_hit",
                "h2026_transfer_rows",
                "h2026_rows",
                "h2026_rate_vs_transfer",
                "h2026_transfer_pos20",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_loss_gt5",
                "h2026_delta_pos_vs_transfer",
                "h2026_delta_avg_pp_vs_transfer",
                "h2026_avg_analog_pos_rate",
                "panel_pos20_mean_std",
                "panel_avg20_pp_mean_std",
                "panel_loss_gt5_mean_std",
                "panel_delta_pos_mean_std",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 12. P0 Transfer Analog/RAG Flash-Pro On/Off Readiness",
        "",
        markdown_table(
            p0_transfer_analog_rag_onoff,
            [
                "variant",
                "role",
                "task_mode",
                "valid_block",
                "evidence_packs",
                "row_level_analog_packs",
                "analog_visible_packs",
                "chip_visible_packs",
                "financial_visible_packs",
                "news_visible_packs",
                "peer_visible_packs",
                "bookskill_visible_packs",
                "quant_tool_visible_packs",
                "future_key_leak_count",
                "decision_cards",
                "status",
                "next_action",
            ],
        ),
        "",
        "## Table 13. P0 Transfer Analog/RAG + K-Line Quant Tool On/Off Readiness",
        "",
        markdown_table(
            p0_transfer_analog_rag_kline_tool_onoff,
            [
                "variant",
                "role",
                "task_mode",
                "valid_block",
                "evidence_packs",
                "row_level_analog_packs",
                "analog_visible_packs",
                "chip_visible_packs",
                "financial_visible_packs",
                "news_visible_packs",
                "peer_visible_packs",
                "bookskill_visible_packs",
                "quant_tool_visible_packs",
                "future_key_leak_count",
                "decision_cards",
                "status",
                "next_action",
            ],
        ),
        "",
        "## Table 13b. P0 Transfer Analog/RAG 36-Panel Flash Preflight",
        "",
        markdown_table(
            p0_transfer_analog_rag_panel36_preflight,
            [
                "frequency",
                "config",
                "analog_id",
                "gate",
                "role",
                "local_status",
                "prior_evaluable_blocks",
                "prior_evaluable_rows_mean",
                "prior_delta_pos_hit",
                "prior_evaluable_delta_pos_hit",
                "prior_evaluable_delta_avg_hit",
                "panels",
                "panel_rows_mean",
                "panel_pos20_mean",
                "panel_pos20_p10",
                "panel_avg20_pp_mean",
                "panel_avg20_pp_p10",
                "panel_loss_gt5_mean",
                "panel_loss_gt5_p90",
                "panel_delta_pos_vs_transfer_mean",
                "panel_delta_pos_vs_transfer_p10",
                "panel_delta_avg_pp_vs_transfer_p10",
                "flash_preflight_status",
                "pro_status",
                "main_read",
            ],
        ),
        "",
        "## Table 14. P0 Small-Entry Case-Memory/RAG Ablation",
        "",
        markdown_table(
            p0_small_case_memory,
            [
                "source",
                "min_applicable_conditions",
                "frequency",
                "config",
                "role",
                "h2026_total_rows",
                "h2026_retained_rows",
                "h2026_retained_rate",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_false_veto_positive_rows",
                "h2026_captured_loss_gt5_rows",
                "h2026_delta_pos",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 15. P0 Small-Entry BookSkill Attribution",
        "",
        markdown_table(
            p0_small_bookskill,
            [
                "frequency",
                "config",
                "role",
                "prior_delta_pos_mean",
                "prior_delta_pos_hit",
                "h2026_selected_rows",
                "h2026_selected_rate",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_false_veto_positive_rows",
                "h2026_captured_loss_gt5_rows",
                "h2026_delta_pos",
                "promotion_status",
                "verdict",
            ],
        ),
        "",
        "## Table 16. P0 PPS-Q-017 On/Off Readiness",
        "",
        markdown_table(
            p0_pps_q017_onoff,
            [
                "config",
                "role",
                "evidence_packs",
                "pps_q017_visible_packs",
                "avg_bookskill_cards",
                "future_key_leak_count",
                "hidden_strategy_ids",
                "decision_cards",
                "status",
                "main_read",
            ],
        ),
        "",
        "## Table 17. P0 PPS-Q-017 Channel Interaction Audit",
        "",
        markdown_table(
            p0_pps_q017_interactions,
            [
                "config",
                "role",
                "rows",
                "selected_rate",
                "pos20",
                "avg20_pp",
                "loss_gt5",
                "delta_pos",
                "delta_avg_pp",
                "unique_stocks",
                "verdict",
                "main_read",
            ],
        ),
        "",
        "## Table 18. P0 Paired Component Ablation",
        "",
        markdown_table(
            p0_components,
            [
                "component",
                "paired_rows",
                "changed_rows",
                "raised_positive",
                "raised_negative",
                "lowered_positive",
                "lowered_negative",
                "delta_cash20_sum_pp",
                "useful_delta_pp",
                "harmful_delta_pp",
                "verdict",
            ],
        ),
        "",
        "## Table 19. P1 Candidate Comparison Flash/Pro Ablation",
        "",
        markdown_table(
            p1_candidate,
            [
                "model",
                "comparison_scenario",
                "variant",
                "role",
                "cards",
                "top1_excess_mean",
                "top2_excess_mean",
                "top1_positive_rate",
                "top2_positive_rate",
                "delta_top1_excess_vs_anchor",
                "delta_top2_excess_vs_anchor",
            ],
        ),
        "",
        "## Table 20. P1 Pro Operation Confirmation User Path",
        "",
        markdown_table(
            p1_operation_confirm,
            [
                "comparison_scenario",
                "cards",
                "actionable_rows",
                "vague_threshold_rows",
                "top1_excess_mean",
                "top2_excess_mean",
                "top1_positive_rate",
                "top2_positive_rate",
                "top1_worst_rate",
                "regret_mean",
                "avg_confidence",
                "verdict",
            ],
        ),
        "",
        "## Table 21. P1 Cross-Sector 3-Seed Stress Test",
        "",
        markdown_table(
            cross_stress,
            [
                "model",
                "variant",
                "role",
                "panels",
                "cards",
                "top1_excess_mean",
                "top1_excess_std",
                "top2_excess_mean",
                "top2_excess_std",
                "top1_positive_rate_mean",
                "top2_positive_rate_mean",
            ],
        ),
        "",
        "## Table 22. Run Hygiene and Token Accounting",
        "",
        markdown_table(hygiene, ["scope", "status", "valid_cards", "invalid_cards", "tokens", "source"]),
        "",
        "## Interpretation Discipline",
        "",
        "- `gray_baseline` 是参考基线，不是模型能力；报告中保留它们是为了防止把低暴露现金路径误读为 alpha。",
        "- P0 的 `exposure_cards=0` 表示 DS 卡片没有进入高主动暴露；这更像防守/审计器表现，不是完整买入引擎。",
        "- P0 用户路径里的旧字段 `research_only/not_investment_instruction` 是历史 schema 兼容字段；当前项目规则允许清晰操作建议，但必须带证据、阈值、反证、数据缺口和风险。",
        "- P0 本地周五融合分支是下一轮 DS Flash/Pro 候选，不是已上线默认策略；它需要恢复 DS 后补 `full / no_kline / no_risk / no_opportunity / quant_only` paired smoke。",
        "- P0 小仓分叉可作为“低仓位试探/继续持有”的候选动作；不得把它解释为高仓位买入，因为 `buy_add` 分叉在最新块没有稳定优势。",
        "- 新闻/财报/同行/筹码 overlay 目前主要作为解释和反证材料；除 `news_low_warning` 这类安全条件外，不能机械升权。财报缺失应明确写为数据缺口。",
        "- 小仓 ML confirmer 的 H2026 局部高收益不能越过 prior 支撑闸门；没有足够 prior block 和 selected-row 覆盖时，只能给 Agent 作灰色参考，不得把分数当作硬阈值。",
        "- 小仓 transfer confirmer 是 ML-to-Agent 的正向候选：它可以作为 DS 问卷里的“候选支持证据”，但 yellow 不是默认；使用时必须让 Agent 再检查新闻/公告/财报/同行/BookSkill 反证。",
        "- 小仓 transfer + channel confirmation 说明非价格通道更适合作为“升降置信的语义问卷”，而不是机械过滤器；`news_financial_clean_no_hard`、`chip_support_no_overheat_no_hard` 是下一轮 Flash/Pro on/off 的优先候选。",
        "- 小仓 transfer + analog/RAG 是当前最像 agent-based model 优势的本地证据：相似历史案例能给 DS 明确的可解释记忆，但 green 仍只是进入模型确认的门票，不是用户端默认买入规则。",
        "- 小仓 case-memory/RAG 当前不能作为自动 guard：`min=2` 没有适用案例，`min=1` 被 `bookskill_missing_or_weak` 全量触发并误杀大量正样本；下一步应补 BookSkill grounding 和动作分支案例，而不是放宽 RAG 阈值。",
        "- 小仓 BookSkill attribution 只能把具体策略 ID 推给 DS 做 on/off 语义确认；`bookskill_gap_any` 不是 hard veto，`positive_historical_skill_any` 只是源卡统计诊断，`PPS-Q-023` 这类弱 grounding 条目必须先补源文本再测试。",
        "- P1 候选对比是当前更接近产品化的能力；跨领域候选对比必须保留行业归一化和 latest-block gate，并继续三次 fresh panel 验证。",
        "",
        "## Artifacts",
        "",
    ]
    for path in outputs.values():
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def render_latex(
    *,
    p0_local: pd.DataFrame,
    p0_ds: pd.DataFrame,
    p0_user_path: pd.DataFrame,
    p0_operation: pd.DataFrame,
    p0_small_overlay: pd.DataFrame,
    p0_small_ml: pd.DataFrame,
    p0_small_transfer: pd.DataFrame,
    p0_small_transfer_channel: pd.DataFrame,
    p0_small_transfer_analog_rag: pd.DataFrame,
    p0_transfer_analog_rag_onoff: pd.DataFrame,
    p0_transfer_analog_rag_kline_tool_onoff: pd.DataFrame,
    p0_transfer_analog_rag_panel36_preflight: pd.DataFrame,
    p0_small_case_memory: pd.DataFrame,
    p0_small_bookskill: pd.DataFrame,
    p0_pps_q017_onoff: pd.DataFrame,
    p0_pps_q017_interactions: pd.DataFrame,
    p0_kline_threshold: pd.DataFrame,
    p0_active_entry_calibration: pd.DataFrame,
    p0_components: pd.DataFrame,
    p1_candidate: pd.DataFrame,
    p1_operation_confirm: pd.DataFrame,
    cross_stress: pd.DataFrame,
    hygiene: pd.DataFrame,
) -> str:
    tables = [
        ("P0 local H2026 24-panel stack", p0_local),
        ("P0 Flash/Pro single-stock agent ablation", p0_ds),
        ("P0 latest single-stock user path Flash panel", p0_user_path),
        ("P0 local user-action policy on H2026 panels", p0_operation),
        ("P0 K-line/peer threshold ablation", p0_kline_threshold),
        ("P0/P1 active-entry threshold calibration", p0_active_entry_calibration),
        ("P0 small-entry non-price overlay ablation", p0_small_overlay),
        ("P0 small-entry ML confirmer diagnostic", p0_small_ml),
        ("P0 small-entry transfer confirmer diagnostic", p0_small_transfer),
        ("P0 small-entry transfer channel confirmation", p0_small_transfer_channel),
        ("P0 transfer analog-case/RAG confirmation", p0_small_transfer_analog_rag),
        ("P0 transfer analog/RAG Flash-Pro on/off readiness", p0_transfer_analog_rag_onoff),
        ("P0 transfer analog/RAG + K-line quant tool on/off readiness", p0_transfer_analog_rag_kline_tool_onoff),
        ("P0 transfer analog/RAG 36-panel Flash preflight", p0_transfer_analog_rag_panel36_preflight),
        ("P0 small-entry case-memory/RAG ablation", p0_small_case_memory),
        ("P0 small-entry BookSkill attribution", p0_small_bookskill),
        ("P0 PPS-Q-017 on/off readiness", p0_pps_q017_onoff),
        ("P0 PPS-Q-017 channel interaction audit", p0_pps_q017_interactions),
        ("P0 paired component ablation", p0_components),
        ("P1 candidate comparison Flash/Pro ablation", p1_candidate),
        ("P1 Pro operation confirmation user path", p1_operation_confirm),
        ("P1 cross-sector 3-seed stress test", cross_stress),
        ("Run hygiene", hygiene),
    ]
    chunks = [
        "% Auto-generated by scripts/build_icml_style_model_capability_tables.py",
        "% Values are offline/backtest metrics, not future-return promises.",
        "",
    ]
    for caption, frame in tables:
        chunks.append(frame.to_latex(index=False, caption=caption, longtable=False, escape=True))
        chunks.append("")
    return "\n".join(chunks)


def p0_local_read(policy: str) -> str:
    if policy == "opp_kline_confirm_no_raise":
        return "best yellow P0 candidate; precision up, still below acceptance threshold"
    if policy == "branch_stack_v1":
        return "same active branch as opp+kline; keep as branch alias"
    if policy == "opp_only":
        return "opportunity-only baseline, lower precision"
    if policy == "kline_only_no_raise":
        return "diagnostic Kline confirmation, not standalone alpha"
    if "baseline" in policy:
        return "gray reference baseline"
    return "diagnostic"


def ds_p0_read(model: str, variant: str) -> str:
    if model == "DS V4 Pro":
        return "Pro default confirmation done; no H2026 fix"
    if variant == "full_agent_with_opportunity_tool":
        return "Flash full P0 agent with opportunity tool; defensive low exposure"
    if variant == "full_agent_without_opportunity_tool":
        return "Flash default-like P0 path used for Pro comparison"
    if variant == "quant_tool_summary_only":
        return "high cash pos from very defensive behavior, not alpha"
    return "Flash component ablation"


def component_control(component: str) -> str:
    mapping = {
        "BookSkill": "no_bookskill",
        "Branch/RAG Case": "no_branch_case_context",
        "Memory/RAG": "no_memory",
        "News": "no_news",
        "Opportunity Tool": "full_agent_without_opportunity_tool",
        "Peer Context": "no_peer",
        "Quant Tools": "no_quant_tools",
        "Risk Review Queue": "full_agent_without_risk_review_queue",
    }
    return mapping.get(component, "unknown_control")


def component_verdict(delta: float | None, harmful: float | None) -> str:
    if delta is None or math.isnan(delta):
        return "insufficient"
    if delta > 0 and (harmful is None or harmful >= -8):
        return "positive_but_needs_fresh_panel"
    if delta > 0:
        return "positive_with_large_error_cost"
    return "do_not_promote"


def p0_user_path_verdict(cash_pos: float | None, active: float | None) -> str:
    if cash_pos is None or math.isnan(cash_pos):
        return "insufficient"
    if active is not None and active > 0.05 and cash_pos >= 0.6:
        return "promising_but_defensive_exposure"
    if cash_pos >= 0.6:
        return "quant_hint_positive_not_agent_alpha"
    if cash_pos >= 0.5:
        return "usable_for_watchlist_safety_not_final_buy_engine"
    return "below_user_path_threshold"


def p0_operation_read(frequency: str, policy: str, row: pd.Series) -> str:
    if policy == "branch_stack_v1":
        if frequency == "weekly_friday":
            return "small-entry branch strongest; buy/add branch weak, use DS confirmation before raising exposure"
        if frequency == "every_2_weeks":
            return "small-entry branch remains positive; slower cadence is plausible"
        return "Tuesday cadence weaker; diagnostic only"
    if policy == "opp_kline_confirm_no_raise":
        return "confirmed opportunity baseline; useful but less user-action specific"
    if policy == "opp_only":
        return "opportunity-only ablation; lower precision than confirmed branch"
    if policy == "hold_all_baseline":
        return "gray market exposure baseline"
    if policy == "bank_all_baseline":
        return "gray cash baseline"
    return "diagnostic"


def p0_overlay_verdict(row: pd.Series) -> str:
    rule = str(row.get("rule_id"))
    selected_rows = num(row.get("h2026_selected_rows")) or 0
    delta_pos = num(row.get("h2026_delta_pos"))
    delta_avg = num(row.get("h2026_delta_avg"))
    status = str(row.get("promotion_status"))
    if rule == "small_entry_all":
        return "keep_as_branch_reference"
    if rule == "news_low_warning" and selected_rows >= 80 and (delta_pos or 0) >= 0:
        return "safety_condition_only_not_alpha_gate"
    if selected_rows < 30:
        return "too_sparse_do_not_promote"
    if "yellow" in status and (delta_pos or 0) >= 0 and (delta_avg or 0) >= 0:
        return "needs_ds_semantic_confirmation"
    if (delta_pos or 0) < 0 or (delta_avg or 0) < 0:
        return "do_not_use_as_mechanical_filter"
    return "diagnostic_only"


def p0_small_ml_verdict(row: pd.Series) -> str:
    variant = str(row.get("variant"))
    if variant == "small_entry_all":
        return "branch_reference"
    status = str(row.get("promotion_status"))
    prior_blocks = num(row.get("prior_blocks")) or 0
    prior_rows = num(row.get("prior_selected_rows_mean")) or 0
    h_pos = num(row.get("h2026_selected_pos20")) or 0
    if "green" in status or "yellow" in status:
        return "needs_ds_semantic_confirmation"
    if prior_blocks < 2 or prior_rows < 30:
        return "diagnostic_only_sparse_prior_support"
    if h_pos >= 0.65:
        return "diagnostic_only_prior_gate_passed_but_not_promoted"
    return "diagnostic_only"


def p0_small_transfer_verdict(row: pd.Series) -> str:
    variant = str(row.get("variant"))
    if variant == "small_entry_all":
        return "branch_reference"
    status = str(row.get("promotion_status"))
    prior_hit = num(row.get("prior_delta_pos_hit")) or 0.0
    prior_rows = num(row.get("prior_selected_rows_mean")) or 0.0
    h_pos = num(row.get("h2026_selected_pos20")) or 0.0
    h_delta = num(row.get("h2026_delta_pos")) or 0.0
    selected_rate = num(row.get("h2026_selected_rate")) or 0.0
    if "green" in status:
        return "candidate_for_ds_flash_pro_confirmation"
    if "yellow" in status:
        return "yellow_only_prior_support_half_hit_requires_fresh_panel"
    if h_pos >= 0.75 and h_delta >= 0.12 and (prior_hit < 0.75 or prior_rows < 50):
        return "h2026_bright_but_prior_support_too_weak"
    if selected_rate < 0.05:
        return "too_sparse_for_user_facing_rule"
    return "diagnostic_only_not_default"


def p0_small_transfer_channel_verdict(row: pd.Series) -> str:
    status = str(row.get("promotion_status"))
    gate = str(row.get("gate_id"))
    h_rows = int_or_none(row.get("h2026_selected_rows")) or 0
    prior_hit = num(row.get("prior_delta_pos_hit")) or 0.0
    delta_pos = num(row.get("h2026_delta_pos_vs_transfer")) or 0.0
    delta_avg = num(row.get("h2026_delta_avg_vs_transfer")) or 0.0
    if status == "transfer_reference":
        return "reference_for_channel_filter_delta"
    if "green" in status:
        return "ready_for_ds_flash_pro_onoff_confirmation"
    if "yellow" in status:
        if gate == "news_financial_clean_no_hard":
            return "yellow_news_financial_clean_gate_prior_half_hit"
        if gate == "chip_support_no_overheat_no_hard":
            return "yellow_chip_support_gate_needs_fresh_panel"
        return "yellow_channel_gate_needs_fresh_panel"
    if h_rows < 15:
        return "too_sparse_or_zero_trigger"
    if delta_pos < 0 or delta_avg < 0:
        return "false_filter_or_mean_return_cost"
    if prior_hit < 0.5:
        return "diagnostic_only_prior_hit_too_low"
    return "diagnostic_only_not_default"


def p0_small_transfer_analog_rag_verdict(row: pd.Series) -> str:
    status = str(row.get("promotion_status"))
    gate = str(row.get("gate_id"))
    prior_hit = num(row.get("prior_delta_pos_hit")) or 0.0
    delta_pos = num(row.get("h2026_delta_pos_vs_transfer")) or 0.0
    delta_avg = num(row.get("h2026_delta_avg_vs_transfer")) or 0.0
    h_rows = int_or_none(row.get("h2026_selected_rows")) or 0
    if status == "transfer_reference":
        return "reference_for_analog_delta"
    if "green" in status:
        if gate == "chip_support_plus_analog065":
            return "green_candidate_chip_plus_analog_requires_ds_flash_pro"
        return "green_candidate_requires_ds_flash_pro"
    if "yellow" in status:
        if "news_financial" in gate:
            return "yellow_news_financial_plus_analog_prior_half_hit"
        if "chip_support" in gate:
            return "yellow_chip_plus_analog_needs_fresh_panel"
        return "yellow_analog_gate_needs_fresh_panel"
    if h_rows < 15:
        return "too_sparse_for_user_facing_rule"
    if delta_pos < 0 or delta_avg < 0:
        return "analog_false_filter_or_mean_return_cost"
    if prior_hit < 0.5:
        return "diagnostic_only_prior_hit_too_low"
    return "diagnostic_only_not_default"


def panel36_role(status: str) -> str:
    if status == "flash_candidate_strong_not_pro_ready":
        return "flash_shortlist_not_pro_ready"
    if status == "flash_candidate_observe_needs_tighter_panel_or_prior":
        return "observe_backup"
    if status == "hold_before_flash_time_generalization_insufficient":
        return "hold_time_generalization_insufficient"
    if status == "reject_before_flash_panel_weak_or_sparse":
        return "reject_before_flash"
    return "diagnostic"


def transfer_analog_rag_onoff_role(variant: str) -> str:
    if variant == "full_agent":
        return "treatment"
    if variant == "no_analogue_case_context":
        return "targeted_analog_rag_ablation"
    if variant in {"no_chip_context", "no_financial_report", "no_news", "no_peer", "no_bookskill", "no_quant_tools"}:
        return "channel_ablation"
    if variant in {"python_only", "quant_tool_summary_only"}:
        return "gray_baseline"
    return "diagnostic_variant"


def transfer_analog_rag_onoff_status(row: pd.Series, *, model_stage: str = "") -> str:
    variant = str(row.get("variant"))
    packs = int_or_none(row.get("evidence_packs")) or 0
    leaks = int_or_none(row.get("future_key_leak_count_sum")) or 0
    if leaks:
        return "blocked_future_leak"
    if packs <= 0:
        return "blocked_no_evidence_packs"
    if variant == "full_agent" and (int_or_none(row.get("row_level_analogue_visible_sum")) or 0) == packs:
        return "ready_for_flash_then_pro_treatment"
    if variant == "no_analogue_case_context" and (int_or_none(row.get("analogue_visible_sum")) or 0) == 0:
        return "ready_for_flash_then_pro_targeted_ablation"
    if variant == "no_chip_context" and (int_or_none(row.get("chip_visible_sum")) or 0) == 0:
        return "ready_channel_ablation"
    if variant == "no_financial_report" and (int_or_none(row.get("financial_visible_sum")) or 0) == 0:
        return "ready_channel_ablation"
    if variant == "no_news" and (int_or_none(row.get("news_visible_sum")) or 0) == 0:
        return "ready_channel_ablation"
    if variant == "no_peer" and (int_or_none(row.get("peer_visible_sum")) or 0) == 0:
        return "ready_channel_ablation"
    if variant == "no_bookskill" and (int_or_none(row.get("bookskill_visible_sum")) or 0) == 0:
        return "ready_channel_ablation"
    if variant == "no_quant_tools" and (int_or_none(row.get("quant_tool_visible_sum")) or 0) == 0:
        if "kline_quant_tool" in model_stage or "quant_tool" in model_stage:
            return "ready_channel_ablation"
        return "ready_but_uninformative_no_quant_tool_visible"
    if variant in {"python_only", "quant_tool_summary_only"}:
        return "ready_gray_baseline"
    return "review_visibility_before_ds_call"


def transfer_analog_rag_onoff_next_action(row: pd.Series, *, model_stage: str = "") -> str:
    packs = int_or_none(row.get("evidence_packs")) or 0
    leaks = int_or_none(row.get("future_key_leak_count_sum")) or 0
    variant = str(row.get("variant"))
    if leaks or packs <= 0:
        return "fix dry-run evidence before any DS call"
    if variant == "no_quant_tools" and (int_or_none(row.get("quant_tool_visible_sum")) or 0) == 0:
        if "kline_quant_tool" in model_stage or "quant_tool" in model_stage:
            return "run Flash first; compare against full_agent to measure K-line quant checklist contribution"
        return "not an informative quant ablation in this shard; add single-stock quant/opportunity tool if needed"
    return "run Flash first; run Pro only if Flash paired on/off and fresh panel remain positive"


def p0_small_case_memory_verdict(row: pd.Series, min_conditions: int) -> str:
    guard = str(row.get("guard_policy"))
    if guard == "no_case_guard":
        return "branch_reference"
    retained_rate = num(row.get("h2026_retained_rate"))
    delta_pos = num(row.get("h2026_delta_pos"))
    false_veto = int_or_none(row.get("h2026_false_veto_positive_rows")) or 0
    captured_loss = int_or_none(row.get("h2026_captured_loss_gt5_rows")) or 0
    if min_conditions >= 2 and retained_rate == 1 and (delta_pos or 0) == 0:
        return "no_applicable_cases_no_signal"
    if retained_rate == 0 and false_veto > captured_loss:
        return "overbroad_guard_reject"
    if guard in {"applicable_any", "risk_condition_ge1"} and false_veto > captured_loss:
        return "false_veto_too_high"
    return "diagnostic_only_not_promoted"


def p0_small_bookskill_verdict(row: pd.Series) -> str:
    rule = str(row.get("rule_id"))
    frequency = str(row.get("frequency"))
    selected_rows = int_or_none(row.get("h2026_selected_rows")) or 0
    delta_pos = num(row.get("h2026_delta_pos")) or 0.0

    if rule == "small_entry_all":
        return "branch_reference"
    if rule == "positive_historical_skill_any":
        return "source_card_stat_diagnostic_only"
    if "needs_grounding" in rule:
        return "weak_until_grounded_no_promote"
    if rule == "bookskill_gap_any":
        return "gap_diagnostic_not_veto_or_alpha"
    if selected_rows < 20:
        return "too_sparse_do_not_promote"
    if rule == "skill_PPS_Q_017" and selected_rows >= 50 and delta_pos >= 0.02:
        return "candidate_for_ds_onoff_not_default"
    if rule == "skill_PPS_M_003" and frequency == "weekly_tuesday" and selected_rows >= 50 and delta_pos >= 0.10:
        return "tuesday_only_candidate_needs_frequency_gate"
    return "diagnostic_only_not_promoted"


def pps_q017_onoff_role(variant: str) -> str:
    if variant == "full_agent":
        return "treatment"
    if variant == "no_pps_q017":
        return "targeted_ablation"
    if variant == "no_bookskill":
        return "broad_bookskill_ablation"
    if variant in {"python_only", "quant_only"}:
        return "gray_baseline"
    return "channel_control"


def pps_q017_onoff_read(variant: str, row: pd.Series, cards: int | None) -> str:
    leaks = int_or_none(row.get("future_key_leak_count")) or 0
    visible = int_or_none(row.get("pps_q017_visible_packs")) or 0
    packs = int_or_none(row.get("evidence_packs")) or 0
    if leaks:
        return "blocked_until_future_leak_fixed"
    if cards:
        return "DS cards available; read paired result table before promotion"
    if variant == "no_pps_q017" and visible == 0 and packs > 0:
        return "targeted ablation is isolated; ready for Flash/Pro shard"
    if variant == "full_agent" and visible == packs and packs > 0:
        return "treatment evidence keeps PPS-Q-017 visible"
    if variant == "no_bookskill" and visible == 0:
        return "broad BookSkill removal control is isolated"
    return "dry-run evidence ready; no decision cards yet"


def pps_q017_interaction_role(verdict: str) -> str:
    if verdict == "candidate_condition_for_ds_prompt_check":
        return "prompt_candidate"
    if verdict == "weak_skill_gap_diagnostic_not_promotion":
        return "diagnostic_not_promotion"
    if verdict == "negative_or_false_filter_risk":
        return "risk_or_false_filter"
    return "diagnostic"


def pps_q017_interaction_read(rule_id: str, verdict: str) -> str:
    if rule_id == "financial_no_recent_event":
        return "strong offline slice; ask DS to distinguish benign no-event vs missing disclosure"
    if rule_id == "chip_support_visible":
        return "candidate support condition; require price/Kline confirmation before action"
    if rule_id == "financial_event_matched":
        return "event presence alone was harmful; use as quality/negative-event audit, not positive"
    if rule_id in {"peer_relative_positive", "chip_low_overhang"}:
        return "looked intuitive but underperformed; do not use mechanically"
    if rule_id == "weak_skill_present":
        return "large diagnostic lift but weak grounding; enrich sources before promotion"
    if verdict == "candidate_condition_for_ds_prompt_check":
        return "candidate for DS on/off prompt check only"
    if verdict == "negative_or_false_filter_risk":
        return "counterexample risk; keep as caution"
    return "diagnostic condition"


def p0_kline_threshold_read(row: pd.Series) -> str:
    verdict = str(row.get("threshold_verdict"))
    config = f"{row.get('decision_frequency')}::{row.get('feature_group')}"
    delta_recall = num(row.get("delta_h2026_risk_recall_top05_minus_top10"))
    if verdict == "keep_wider_threshold":
        return "keep top10 as default checklist threshold; top05 weakens precision or risk recall"
    if verdict == "narrow_threshold_candidate":
        return "candidate only if fresh panel and non-price confirmation also pass"
    if verdict == "narrow_threshold_mean_only_diagnostic":
        return "top05 improves mean but loses risk recall; high-conviction diagnostic, not default"
    if delta_recall is not None and delta_recall < -0.04:
        return "top05 materially reduces risk recall; do not use as risk guard"
    if "rev_chip_core_fixed" in config:
        return "rev+chip remains the preferred K-line checklist family"
    return "diagnostic threshold result"


def p0_active_entry_calibration_read(row: pd.Series) -> str:
    task_mode = str(row.get("task_mode"))
    variant = str(row.get("variant"))
    pos = num(row.get("h2026_active_pos20")) or 0.0
    avg = num(row.get("h2026_active_avg20")) or 0.0
    rate = num(row.get("h2026_active_rate")) or 0.0
    if pos < 0.5 and avg < 0:
        return "negative latest-block active-entry evidence; do not promote"
    if task_mode == "candidate_select_then_single_watch" and avg >= 0 and pos < 0.5:
        return "slightly better weak-market mean, but active win-rate too low"
    if variant == "base_v4_like":
        return "default anchor reference; not sufficient as buy/add engine"
    if rate < 0.15:
        return "too low exposure for user-facing capability claim"
    return "reference-only threshold result; needs cross-channel confirmation"


def active_entry_calibration_status() -> str:
    hygiene = read_csv(P0_ACTIVE_ENTRY_CALIBRATION_HYGIENE)
    if hygiene.empty:
        return "completed_no_deepseek_hygiene_missing"
    status_by_check = {
        str(row.get("check")): str(row.get("status"))
        for _, row in hygiene.iterrows()
    }
    future_status = status_by_check.get("preview_future_key_scan", "")
    variants_status = status_by_check.get("variants", "")
    if future_status == "ok" and variants_status == "ok":
        return "completed_no_deepseek_future_scan_ok"
    return "completed_no_deepseek_review_hygiene"


def p1_operation_verdict(row: pd.Series) -> str:
    top1 = float(row.get("top1_excess_mean", 0) or 0)
    top2 = float(row.get("top2_excess_mean", 0) or 0)
    pos1 = float(row.get("top1_positive_rate", 0) or 0)
    scenario = str(row.get("comparison_scenario", ""))
    if top1 > 0 and top2 > 0 and pos1 >= 0.55:
        return f"{scenario}_usable_small_panel_needs_3seed"
    if top1 > 0 and top2 > 0:
        return f"{scenario}_positive_excess_but_hit_rate_weak"
    return f"{scenario}_not_ready"


def p1_read(model: str, variant: str, scenario: str, row: Any) -> str:
    top1 = getattr(row, "top1_excess_mean", float("nan"))
    top2 = getattr(row, "top2_excess_mean", float("nan"))
    if scenario == "same_sector" and variant == "ranker_anchor_agent" and top1 > 0:
        return f"{model} same-sector anchor positive"
    if scenario == "cross_sector" and variant == "ranker_anchor_agent" and (top1 < 0 or top2 < 0):
        return "cross-sector anchor unstable"
    if variant != "ranker_anchor_agent":
        return "component ablation; compare sign with anchor"
    return "diagnostic"


def cross_stress_read(row: pd.Series) -> str:
    if row.get("variant") == "ranker_anchor_agent" and float(row.get("top2_excess_mean", 0) or 0) < 0:
        return "stress test says cross-sector top2 not stable"
    if row.get("variant") != "ranker_anchor_agent":
        return "ablation stress result; not a promotion gate"
    return "anchor stress result"


def usage_tokens(path: Path) -> int | None:
    frame = read_csv(path)
    if frame.empty or "total_tokens" not in frame:
        return None
    return int(pd.to_numeric(frame["total_tokens"], errors="coerce").fillna(0).sum())


def row_count(path: Path) -> int | None:
    frame = read_csv(path)
    if frame.empty:
        return 0 if path.exists() else None
    return int(len(frame))


def visibility_evidence_pack_count(path: Path) -> int | None:
    frame = read_csv(path)
    if frame.empty:
        return 0 if path.exists() else None
    if "evidence_packs" not in frame:
        return len(frame)
    return int(pd.to_numeric(frame["evidence_packs"], errors="coerce").fillna(0).sum())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def mean(frame: pd.DataFrame, col: str) -> float | None:
    if col not in frame:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def std(frame: pd.DataFrame, col: str) -> float | None:
    if col not in frame:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else None
    return float(values.std())


def first_value(frame: pd.DataFrame, col: str) -> Any:
    if frame.empty or col not in frame:
        return ""
    return frame.iloc[0].get(col, "")


def num(value: Any) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def int_or_none(value: Any) -> int | None:
    number = num(value)
    if number is None:
        return None
    return int(number)


def safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)]
    return "".join(chars).strip("_") or "flash_pro_icml_capability_tables_v1"


if __name__ == "__main__":
    main()
