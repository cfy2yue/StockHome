"""Audit K-line/peer tool overlays on existing P0 user-operation decisions.

This is a local, no-DeepSeek experiment. Future 20-day returns are used only
for offline evaluation. The optional agent preview output contains only
decision-time scores and threshold hints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_multiscale_kline_peer_tool_v1 import (  # noqa: E402
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    REPORT_DIR,
    TARGET_BLOCKS,
    apply_frequency,
    attach_scores,
    block_for_date,
    blocks_before,
    build_feature_map,
    choose_opportunity_threshold,
    choose_risk_threshold,
    fit_model,
    load_frame,
    opportunity_label,
    risk_label,
    rolling_split,
    stable_hash_int,
)
from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH  # noqa: E402


DEFAULT_PREFIX = "p0_kline_peer_overlay_user_ops_v1"
BANK_20D_RETURN_PCT = 0.03 * 20 / 252 * 100
FUTURE_OR_RESULT_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "target_cash20",
    "sim_cash20",
    "gt_status",
    "gt_pass",
    "raw_positive_20d",
}

DEFAULT_DECISION_FILES = [
    "p0_small_entry_pps_q017_userop72_fresh2_key_ablation_flash_v1_user_operation_decision_detail.csv",
    "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_user_operation_decision_detail.csv",
    "p0_small_entry_general_channel_fresh2_key4_flash_v1_user_operation_decision_detail.csv",
    "p0_small_entry_general_channel_fresh3_key4_flash_v1_user_operation_decision_detail.csv",
    "p0_action_label_tool_flash_preflight_v2_pair_flash_user_operation_decision_detail.csv",
]
DEFAULT_VARIANT_BY_FILE = {
    "p0_small_entry_pps_q017_userop72_fresh2_key_ablation_flash_v1_user_operation_decision_detail.csv": "full_agent",
    "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_user_operation_decision_detail.csv": "full_agent",
    "p0_small_entry_general_channel_fresh2_key4_flash_v1_user_operation_decision_detail.csv": "full_agent_with_quant_tools",
    "p0_small_entry_general_channel_fresh3_key4_flash_v1_user_operation_decision_detail.csv": "full_agent_with_quant_tools",
    "p0_action_label_tool_flash_preflight_v2_pair_flash_user_operation_decision_detail.csv": "full_agent",
}
DEFAULT_TOOL_CONFIGS = [
    ("weekly_friday", "kline_peer_chip_news_risk", "hgb"),
    ("weekly_friday", "kline_peer_chip", "hgb"),
    ("every_2_weeks", "kline_peer_chip", "hgb"),
    ("weekly_tuesday", "kline_peer_chip_news_risk", "hgb"),
]
POLICIES = [
    "baseline_ds",
    "risk_cap_10",
    "risk_cap_0",
    "opp_floor_20",
    "opp_floor_30",
    "opp_floor_20_nonrisk",
    "opp_floor_30_nonrisk",
    "opp_floor_20_active_only",
    "opp_floor_30_active_only",
    "opp_floor_20_nonrisk_active",
    "opp_floor_30_nonrisk_active",
    "guarded_floor20_cap10",
    "guarded_floor30_cap10",
    "guarded_floor20_cap10_nonrisk",
    "guarded_floor30_cap10_nonrisk",
    "risk_review_cap20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 K-line/peer overlays on user-operation cards.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--decision-files", default=",".join(DEFAULT_DECISION_FILES))
    parser.add_argument(
        "--tool-configs",
        default=",".join(":".join(item) for item in DEFAULT_TOOL_CONFIGS),
        help="Comma list of frequency:feature_group:model.",
    )
    parser.add_argument("--max-hgb-train-rows", type=int, default=60000)
    parser.add_argument("--panel-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decision_files = [item.strip() for item in str(args.decision_files).split(",") if item.strip()]
    tool_configs = parse_tool_configs(args.tool_configs)

    user_ops = load_user_operation_rows(decision_files)
    joined = load_frame(args.joined_cache)
    score_rows = build_tool_score_rows(joined, tool_configs, max_hgb_train_rows=args.max_hgb_train_rows)
    merged = merge_user_ops_with_scores(user_ops, score_rows)
    overlay_detail = evaluate_overlays(merged)
    summary = summarize_overlays(overlay_detail)
    by_block = summarize_by_block(overlay_detail)
    by_source = summarize_by_source(overlay_detail)
    panels = summarize_h2026_panels(overlay_detail, panel_size=args.panel_size)
    preview = build_agent_preview(merged)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "score_rows": REPORT_DIR / f"{prefix}_tool_scores.csv",
        "overlay_detail": REPORT_DIR / f"{prefix}_overlay_detail.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "by_block": REPORT_DIR / f"{prefix}_by_block.csv",
        "by_source": REPORT_DIR / f"{prefix}_by_source.csv",
        "panels": REPORT_DIR / f"{prefix}_h2026_panel_stability.csv",
        "agent_preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    score_rows.to_csv(paths["score_rows"], index=False, encoding="utf-8-sig")
    overlay_detail.to_csv(paths["overlay_detail"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    by_block.to_csv(paths["by_block"], index=False, encoding="utf-8-sig")
    by_source.to_csv(paths["by_source"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panels"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["agent_preview"], preview)
    paths["report"].write_text(render_report(args, paths, user_ops, score_rows, overlay_detail, summary, by_block, by_source, panels), encoding="utf-8")

    print("A股研究Agent")
    print(f"user_ops={len(user_ops)} scored_matches={merged['has_tool_score'].sum()} overlay_rows={len(overlay_detail)}")
    print(f"report={paths['report']}")


def parse_tool_configs(value: str) -> list[tuple[str, str, str]]:
    configs: list[tuple[str, str, str]] = []
    for item in str(value).split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid tool config: {item}")
        configs.append((parts[0], parts[1], parts[2]))
    return configs


def load_user_operation_rows(files: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for filename in files:
        path = REPORT_DIR / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
        wanted_variant = DEFAULT_VARIANT_BY_FILE.get(filename)
        if wanted_variant and "variant" in frame:
            frame = frame[frame["variant"].astype(str).eq(wanted_variant)].copy()
        if frame.empty:
            continue
        frame["source_file"] = filename
        frame["source_family"] = source_family(filename)
        frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
        if "valid_block" not in frame:
            frame["valid_block"] = frame["decision_date"].map(block_for_date)
        frame["base_target_position"] = pd.to_numeric(frame.get("target_position"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        frame["return_20d"] = pd.to_numeric(frame.get("return_20d"), errors="coerce")
        frame["base_cash20"] = (
            frame["base_target_position"] * frame["return_20d"]
            + (1.0 - frame["base_target_position"]) * BANK_20D_RETURN_PCT
        )
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def source_family(filename: str) -> str:
    if "pps_q017" in filename:
        return "pps_q017_small_entry"
    if "general_channel" in filename:
        return "general_channel_small_entry"
    if "action_label" in filename:
        return "action_label_tool_v2"
    return "other"


def build_tool_score_rows(joined: pd.DataFrame, configs: list[tuple[str, str, str]], *, max_hgb_train_rows: int) -> pd.DataFrame:
    feature_map = build_feature_map(joined)
    rows: list[pd.DataFrame] = []
    for frequency, feature_group, model_name in configs:
        freq_frame = apply_frequency(joined, frequency)
        features = feature_map.get(feature_group, [])
        if len(features) < 5:
            continue
        for target_block in TARGET_BLOCKS:
            train, valid, target = rolling_split(freq_frame, target_block)
            if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                continue
            opp_model = fit_model(
                train,
                features,
                label=opportunity_label(train),
                model_name=model_name,
                feature_group=feature_group,
                max_hgb_train_rows=max_hgb_train_rows,
            )
            risk_model = fit_model(
                train,
                features,
                label=risk_label(train),
                model_name=model_name,
                feature_group=feature_group,
                max_hgb_train_rows=max_hgb_train_rows,
            )
            if opp_model is None or risk_model is None:
                continue
            valid_scored = attach_scores(valid, opp_model, risk_model)
            target_scored = attach_scores(target, opp_model, risk_model)
            opp_threshold, opp_valid = choose_opportunity_threshold(valid_scored)
            risk_threshold, risk_valid = choose_risk_threshold(valid_scored)
            out = target_scored[["date", "code", "name", "time_block", "opp_score", "risk_score"]].copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
            out["code"] = out["code"].astype(str).str.zfill(6)
            out["frequency"] = frequency
            out["feature_group"] = feature_group
            out["model"] = model_name
            out["tool_config"] = tool_config_id(frequency, feature_group, model_name)
            out["opp_threshold"] = float(opp_threshold)
            out["risk_threshold"] = float(risk_threshold)
            out["validation_active_pos_delta"] = opp_valid.get("active_pos_delta")
            out["validation_active_avg_delta"] = opp_valid.get("active_avg_delta")
            out["validation_risk_loss_rate"] = risk_valid.get("active_loss_rate")
            out["tool_risk_hard"] = pd.to_numeric(out["risk_score"], errors="coerce") >= float(risk_threshold)
            out["tool_opp_active"] = (pd.to_numeric(out["opp_score"], errors="coerce") >= float(opp_threshold)) & ~out["tool_risk_hard"]
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def tool_config_id(frequency: str, feature_group: str, model_name: str) -> str:
    return f"{frequency}:{feature_group}:{model_name}"


def merge_user_ops_with_scores(user_ops: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if user_ops.empty or scores.empty:
        out = user_ops.copy()
        out["has_tool_score"] = False
        return out
    left = user_ops.copy()
    right = scores.copy()
    left["decision_date"] = pd.to_datetime(left["decision_date"], errors="coerce").dt.date.astype(str)
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date.astype(str)
    merged = left.merge(
        right,
        left_on=["decision_date", "code"],
        right_on=["date", "code"],
        how="inner",
        suffixes=("", "_tool"),
    )
    merged["has_tool_score"] = True
    return merged


def overlay_position(
    base_position: float,
    *,
    policy: str,
    opp_active: bool,
    risk_hard: bool,
    ds_risk_action: bool = False,
) -> float:
    pos = min(max(float(base_position), 0.0), 1.0)
    active_base = pos > 1e-9
    nonrisk = not bool(ds_risk_action)
    if policy == "baseline_ds":
        return pos
    if policy == "risk_cap_10":
        return min(pos, 0.10) if risk_hard else pos
    if policy == "risk_cap_0":
        return 0.0 if risk_hard else pos
    if policy == "risk_review_cap20":
        return min(pos, 0.20) if risk_hard else pos
    if policy == "opp_floor_20":
        return max(pos, 0.20) if opp_active and not risk_hard else pos
    if policy == "opp_floor_30":
        return max(pos, 0.30) if opp_active and not risk_hard else pos
    if policy == "opp_floor_20_nonrisk":
        return max(pos, 0.20) if opp_active and not risk_hard and nonrisk else pos
    if policy == "opp_floor_30_nonrisk":
        return max(pos, 0.30) if opp_active and not risk_hard and nonrisk else pos
    if policy == "opp_floor_20_active_only":
        return max(pos, 0.20) if opp_active and not risk_hard and active_base else pos
    if policy == "opp_floor_30_active_only":
        return max(pos, 0.30) if opp_active and not risk_hard and active_base else pos
    if policy == "opp_floor_20_nonrisk_active":
        return max(pos, 0.20) if opp_active and not risk_hard and nonrisk and active_base else pos
    if policy == "opp_floor_30_nonrisk_active":
        return max(pos, 0.30) if opp_active and not risk_hard and nonrisk and active_base else pos
    if policy == "guarded_floor20_cap10":
        if risk_hard:
            return min(pos, 0.10)
        return max(pos, 0.20) if opp_active else pos
    if policy == "guarded_floor30_cap10":
        if risk_hard:
            return min(pos, 0.10)
        return max(pos, 0.30) if opp_active else pos
    if policy == "guarded_floor20_cap10_nonrisk":
        if risk_hard:
            return min(pos, 0.10)
        return max(pos, 0.20) if opp_active and nonrisk else pos
    if policy == "guarded_floor30_cap10_nonrisk":
        if risk_hard:
            return min(pos, 0.10)
        return max(pos, 0.30) if opp_active and nonrisk else pos
    raise ValueError(f"unknown overlay policy: {policy}")


def evaluate_overlays(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        ret = safe_float(row.get("return_20d"))
        base_position = safe_float(row.get("base_target_position"))
        base_cash = safe_float(row.get("base_cash20"))
        opp_active = bool(row.get("tool_opp_active"))
        risk_hard = bool(row.get("tool_risk_hard"))
        ds_risk_action = to_bool(row.get("risk_action"))
        for policy in POLICIES:
            pos = overlay_position(
                base_position,
                policy=policy,
                opp_active=opp_active,
                risk_hard=risk_hard,
                ds_risk_action=ds_risk_action,
            )
            cash = pos * ret + (1.0 - pos) * BANK_20D_RETURN_PCT
            position_delta = pos - base_position
            rows.append(
                {
                    "source_file": row.get("source_file"),
                    "source_family": row.get("source_family"),
                    "panel_label": row.get("panel_label"),
                    "variant": row.get("variant"),
                    "valid_block": row.get("valid_block"),
                    "decision_date": row.get("decision_date"),
                    "code": str(row.get("code")).zfill(6),
                    "name": row.get("name"),
                    "tool_config": row.get("tool_config"),
                    "frequency": row.get("frequency"),
                    "feature_group": row.get("feature_group"),
                    "model": row.get("model"),
                    "policy": policy,
                    "opp_score": row.get("opp_score"),
                    "risk_score": row.get("risk_score"),
                    "opp_threshold": row.get("opp_threshold"),
                    "risk_threshold": row.get("risk_threshold"),
                    "tool_opp_active": opp_active,
                    "tool_risk_hard": risk_hard,
                    "ds_risk_action": ds_risk_action,
                    "base_target_position": base_position,
                    "overlay_target_position": round(float(pos), 6),
                    "position_delta": round(float(position_delta), 6),
                    "return_20d": ret,
                    "base_cash20": base_cash,
                    "overlay_cash20": round(float(cash), 6),
                    "delta_cash20": round(float(cash - base_cash), 6),
                    "raised_positive": bool(position_delta > 1e-9 and ret > 0),
                    "raised_negative": bool(position_delta > 1e-9 and ret <= 0),
                    "lowered_positive": bool(position_delta < -1e-9 and ret > 0),
                    "lowered_negative": bool(position_delta < -1e-9 and ret <= 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_overlays(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detail.groupby(["tool_config", "policy"], sort=True):
        rows.append(summary_row(group, {"tool_config": keys[0], "policy": keys[1]}))
    out = pd.DataFrame(rows)
    out["promotion_status"] = out.apply(overlay_status, axis=1)
    out["rank_score"] = out.apply(overlay_rank_score, axis=1)
    return out.sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def summarize_by_block(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detail.groupby(["tool_config", "policy", "valid_block"], sort=True):
        rows.append(summary_row(group, {"tool_config": keys[0], "policy": keys[1], "valid_block": keys[2]}))
    return pd.DataFrame(rows)


def summarize_by_source(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in detail.groupby(["tool_config", "policy", "source_family"], sort=True):
        rows.append(summary_row(group, {"tool_config": keys[0], "policy": keys[1], "source_family": keys[2]}))
    return pd.DataFrame(rows)


def summary_row(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    cash = pd.to_numeric(group["overlay_cash20"], errors="coerce")
    base_cash = pd.to_numeric(group["base_cash20"], errors="coerce")
    delta = pd.to_numeric(group["delta_cash20"], errors="coerce")
    pos = pd.to_numeric(group["overlay_target_position"], errors="coerce")
    base_pos = pd.to_numeric(group["base_target_position"], errors="coerce")
    ret = pd.to_numeric(group["return_20d"], errors="coerce")
    return {
        **keys,
        "rows": int(len(group)),
        "unique_stock_dates": int(group[["decision_date", "code"]].drop_duplicates().shape[0]),
        "source_families": int(group["source_family"].nunique()) if "source_family" in group else 0,
        "blocks": int(group["valid_block"].nunique()) if "valid_block" in group else 0,
        "base_cash_avg": round(float(base_cash.mean()), 6),
        "overlay_cash_avg": round(float(cash.mean()), 6),
        "delta_cash_avg": round(float(delta.mean()), 6),
        "delta_cash_sum": round(float(delta.sum()), 6),
        "overlay_cash_positive_rate": round(float((cash > 0).mean()), 6),
        "base_cash_positive_rate": round(float((base_cash > 0).mean()), 6),
        "delta_positive_rate": round(float((delta > 0).mean()), 6),
        "base_avg_position": round(float(base_pos.mean()), 6),
        "overlay_avg_position": round(float(pos.mean()), 6),
        "changed_rows": int(delta.abs().gt(1e-9).sum()),
        "raised_positive": int(group["raised_positive"].sum()),
        "raised_negative": int(group["raised_negative"].sum()),
        "lowered_positive": int(group["lowered_positive"].sum()),
        "lowered_negative": int(group["lowered_negative"].sum()),
        "tool_opp_active_rate": round(float(group["tool_opp_active"].mean()), 6),
        "tool_risk_hard_rate": round(float(group["tool_risk_hard"].mean()), 6),
        "ds_risk_action_rate": round(float(group["ds_risk_action"].mean()), 6) if "ds_risk_action" in group else np.nan,
        "raw_pos_rate": round(float((ret > 0).mean()), 6),
    }


def overlay_status(row: pd.Series) -> str:
    if str(row.get("policy")) == "baseline_ds":
        return "control"
    delta = safe_float(row.get("delta_cash_avg"))
    changed = safe_float(row.get("changed_rows"))
    raised_negative = safe_float(row.get("raised_negative"))
    lowered_positive = safe_float(row.get("lowered_positive"))
    blocks = safe_float(row.get("blocks"))
    families = safe_float(row.get("source_families"))
    if changed <= 0:
        return "no_effect_control"
    if delta > 0 and blocks >= 3 and families >= 2 and raised_negative <= lowered_positive:
        return "observe_candidate_needs_block_check"
    if delta > 0:
        return "diagnostic_positive_not_promoted"
    return "rejected_overlay"


def overlay_rank_score(row: pd.Series) -> float:
    return (
        safe_float(row.get("delta_cash_avg"))
        + 0.02 * safe_float(row.get("delta_cash_sum"))
        + 0.05 * safe_float(row.get("lowered_negative"))
        - 0.05 * safe_float(row.get("raised_negative"))
        - 0.03 * safe_float(row.get("lowered_positive"))
    )


def summarize_h2026_panels(detail: pd.DataFrame, *, panel_size: int) -> pd.DataFrame:
    h = detail[detail["valid_block"].astype(str).eq(FINAL_OOT)].copy()
    if h.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in h.groupby(["tool_config", "policy"], sort=True):
        codes = sorted(group["code"].astype(str).unique())
        for seed in range(3):
            selected_codes = set(sorted(codes, key=lambda code: stable_hash_int("p0_overlay_panel", seed, keys[0], keys[1], code))[: min(panel_size, len(codes))])
            panel = group[group["code"].astype(str).isin(selected_codes)].copy()
            rows.append(summary_row(panel, {"tool_config": keys[0], "policy": keys[1], "panel_seed": seed, "panel_codes": len(selected_codes)}))
    return pd.DataFrame(rows)


def build_agent_preview(merged: pd.DataFrame, max_rows: int = 2000) -> list[dict[str, Any]]:
    if merged.empty:
        return []
    sample = merged.sort_values(["tool_risk_hard", "tool_opp_active", "decision_date", "code"], ascending=[False, False, True, True]).head(max_rows)
    rows = []
    for _, row in sample.iterrows():
        if bool(row.get("tool_risk_hard")):
            action_hint = "kline_peer_risk_review_cap_position"
            position_hint = "cap_new_position_0_to_10pct_or_review_existing"
        elif bool(row.get("tool_opp_active")):
            action_hint = "kline_peer_supports_small_trial_only"
            position_hint = "can_support_10_to_20pct_trial_if_nonprice_no_hard_counter"
        else:
            action_hint = "kline_peer_neutral_checklist"
            position_hint = "do_not_raise_position_from_kline_alone"
        record = {
            "date": row.get("decision_date"),
            "code": str(row.get("code")).zfill(6),
            "name": row.get("name"),
            "tool_id": "p0_kline_peer_overlay_user_ops_v1",
            "tool_config": row.get("tool_config"),
            "opp_score": round(safe_float(row.get("opp_score")), 6),
            "risk_score": round(safe_float(row.get("risk_score")), 6),
            "opp_threshold": round(safe_float(row.get("opp_threshold")), 6),
            "risk_threshold": round(safe_float(row.get("risk_threshold")), 6),
            "action_hint": action_hint,
            "position_hint": position_hint,
            "agent_use_rule": "Use as checklist/position overlay only; do not buy/add from K-line peer signal without news/financial/BookSkill/peer non-price confirmation.",
            "source_ref_ids": "p0_kline_peer_overlay_user_ops_v1;p0_multiscale_kline_peer_tool_v1",
            "research_only": True,
            "not_investment_instruction": True,
        }
        assert_no_future_fields(record)
        rows.append(record)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            assert_no_future_fields(row)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_KEYS or key.startswith("return_") or key.endswith("_cash20"):
                raise ValueError(f"future/result field leaked into preview: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def render_report(
    args: argparse.Namespace,
    paths: dict[str, Path],
    user_ops: pd.DataFrame,
    score_rows: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    by_block: pd.DataFrame,
    by_source: pd.DataFrame,
    panels: pd.DataFrame,
) -> str:
    best = summary.head(16) if not summary.empty else summary
    non_baseline = summary[summary["policy"].astype(str).ne("baseline_ds")].copy() if not summary.empty else summary
    best_overlay = non_baseline.sort_values("rank_score", ascending=False).head(1) if not non_baseline.empty else non_baseline
    lines = [
        "# P0 K-Line/Peer Overlay on User Operations",
        "",
        "本实验是本地离线审计，不调用 DeepSeek，不读取或输出 API key/token。未来 20 日结果只用于评估 overlay 对既有 P0 用户操作建议的影响，不进入 agent preview。",
        "",
        "## Purpose",
        "",
        "检验多尺度历史 K 线、相关股票 K 线、Tushare 行业/地域 peer 和筹码工具，在真实 P0 用户操作卡上更适合作为：加仓确认、降仓/卖出复核，还是只作为检查清单。",
        "",
        "## Inputs",
        "",
        f"- decision_rows_after_default_variant_filter: `{len(user_ops)}`",
        f"- tool_score_rows: `{len(score_rows)}`",
        f"- overlay_eval_rows: `{len(detail)}`",
        f"- joined_cache: `{args.joined_cache}`",
        f"- tool_configs: `{args.tool_configs}`",
        f"- output_detail: `{paths['overlay_detail']}`",
        f"- output_agent_preview: `{paths['agent_preview']}`",
        "",
        "## Key Finding",
        "",
        *_key_findings(summary, by_block, panels),
        "",
        "## Overlay Policies",
        "",
        "- `baseline_ds`: 不改动 DeepSeek/Python 已输出的用户仓位建议。",
        "- `risk_cap_10 / risk_cap_0 / risk_review_cap20`: K线/相关股风险分数过阈值时，分别把仓位上限压到 10%、0% 或 20%。",
        "- `opp_floor_20 / opp_floor_30`: K线/相关股机会分数过阈值且风险未过阈值时，把小仓地板提高到 20% 或 30%。",
        "- `*_nonrisk`: 只有原 DS 用户建议不是减仓/卖出复核等风险动作时，才允许提高小仓地板。",
        "- `*_active_only`: 只有原 DS 已经给了非零仓位时，才允许提高小仓地板。",
        "- `guarded_floor20_cap10 / guarded_floor30_cap10`: 机会确认时提高小仓地板，风险确认时压仓。",
        "",
        "## Overall Summary",
        "",
        markdown_table(best),
        "",
        "## Best Overlay By Block",
        "",
        markdown_table(_best_by_group(by_block, ["valid_block"])),
        "",
        "## Best Overlay By Source Family",
        "",
        markdown_table(_best_by_group(by_source, ["source_family"])),
        "",
        "## H2026 3-Panel Stability",
        "",
        markdown_table(panels.head(80) if not panels.empty else panels),
        "",
        "## Interpretation",
        "",
        "- 只有同时跨来源、跨时间块、H2026 三次 panel 稳定为正，且没有明显 `raised_negative` 增加，才允许进入 DS 小样本。",
        "- 如果风险 cap 的 `delta_cash_avg` 为负，说明 K线/peer 风险分会错杀后验正样本，只能作为复查提示，不能自动降仓。",
        "- 如果机会 floor 的均值为正但 `raised_negative` 很高，说明它只适合提示“可考虑小仓”，不能自动加仓。",
        "- 本轮若没有通过项，默认继续沿用 P0 小仓主线：K线/相关股是解释、复查频率和阈值材料，不是独立买入公式。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def _key_findings(summary: pd.DataFrame, by_block: pd.DataFrame, panels: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- 没有形成 overlay 结果。"]
    non_baseline = summary[summary["policy"].astype(str).ne("baseline_ds")].sort_values("rank_score", ascending=False)
    if non_baseline.empty:
        return ["- 只有 baseline 行，未形成 overlay 对照。"]
    best = non_baseline.iloc[0]
    lines = [
        "- 最好 overlay 为 "
        f"`{best['tool_config']} / {best['policy']}`：平均 `delta_cash_avg={float(best['delta_cash_avg']):+.4f}pp`，"
        f"changed_rows=`{int(best['changed_rows'])}`，raised_negative=`{int(best['raised_negative'])}`，"
        f"lowered_positive=`{int(best['lowered_positive'])}`，状态 `{best['promotion_status']}`。"
    ]
    if str(best["promotion_status"]).startswith("observe"):
        lines.append("- 它只能进入观察候选：需要分块和 H2026 panel 继续确认，不直接改默认用户仓位。")
    elif float(best["delta_cash_avg"]) <= 0:
        lines.append("- 关键反证：overlay 平均没有改善，K线/相关股不能自动加仓或降仓。")
    if not by_block.empty:
        block_best = _best_by_group(by_block[by_block["policy"].astype(str).ne("baseline_ds")], ["valid_block"])
        if not block_best.empty:
            positives = int(pd.to_numeric(block_best["delta_cash_avg"], errors="coerce").gt(0).sum())
            lines.append(f"- 分块检查：各块最佳 overlay 中 `{positives}/{len(block_best)}` 个 delta 为正；不足全正则不能升默认。")
    if not panels.empty:
        h = panels[panels["policy"].astype(str).eq(str(best["policy"])) & panels["tool_config"].astype(str).eq(str(best["tool_config"]))]
        if not h.empty:
            values = pd.to_numeric(h["delta_cash_avg"], errors="coerce")
            lines.append(f"- H2026 三次 panel：delta_cash_avg `{values.mean():+.4f}±{values.std():.4f}pp`。")
    lines.append("- Agent 使用建议：把 K线/peer 分数写入证据包，作为小仓确认或风险复查提示；没有新闻/财报/BookSkill/非价格证据确认时，不允许仅凭它提高仓位。")
    return lines


def _best_by_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "rank_score" not in data:
        data["rank_score"] = data.apply(overlay_rank_score, axis=1)
    rows = []
    for _, group in data.groupby(group_cols, sort=True):
        rows.append(group.sort_values("rank_score", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()
