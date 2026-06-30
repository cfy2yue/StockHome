"""Audit a channel-augmented single-stock opportunity/risk scorer.

This is a local, time-safe experiment. Future labels are used only for offline
evaluation. Agent-facing preview rows are built from whitelisted score fields
and explicitly reject future/result columns.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_single_stock_review_quality import (  # noqa: E402
    FINAL_OOT,
    TARGET_BLOCKS,
    block_base_metrics,
    choose_opportunity_threshold,
    choose_risk_threshold,
    fit_risk_model,
    load_merged_frame,
    loss_exposure_after_exclude,
    risk_recall,
    score_risk,
    selection_hygiene,
    side_metrics,
)
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    fit_additive_bin_model,
    score_frame,
    _rolling_split,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
CHANNEL_SCORE_PATH = REPORT_DIR / "channel_rule_outcome_classifier_v1_scored_detail.csv"
REPORT_PATH = REPORT_DIR / "single_stock_channel_scorer_v1.md"
CSV_PATH = REPORT_DIR / "single_stock_channel_scorer_v1.csv"
AGENT_PREVIEW_PATH = REPORT_DIR / "single_stock_channel_scorer_v1_agent_tool_preview.jsonl"

SAFE_CHANNEL_SCORE_COLUMNS = {
    "logistic_channel_outcome__prob_hard_counter": "channel_hard_counter_prob",
    "logistic_channel_outcome__prob_soft_gap": "channel_soft_gap_prob",
    "logistic_channel_outcome__prob_positive_support": "channel_positive_support_prob",
    "logistic_channel_outcome__prob_neutral": "channel_neutral_prob",
}
FUTURE_OR_RESULT_FIELDS = {
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
CHANNEL_FEATURES = [
    "channel_hard_counter_prob",
    "channel_soft_gap_prob",
    "channel_positive_support_prob",
    "channel_neutral_prob",
    "channel_counter_gap_prob",
    "channel_soft_or_hard_prob",
    "channel_positive_gap_prob",
    "channel_hard_counter_yellow_flag",
    "channel_hard_counter_high_flag",
    "channel_soft_gap_dominant_flag",
    "channel_score_coverage",
]


def _norm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def _norm_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = _norm_columns(frame)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def load_safe_channel_scores(path: Path = CHANNEL_SCORE_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load only decision-time channel probabilities from the scored detail file."""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    raw = _norm_columns(raw)
    required = ["date", "code", *SAFE_CHANNEL_SCORE_COLUMNS.keys()]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"missing channel score columns: {missing}")
    out = raw[required].copy()
    out = _norm_keys(out)
    out = out.rename(columns=SAFE_CHANNEL_SCORE_COLUMNS)
    for col in SAFE_CHANNEL_SCORE_COLUMNS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    core = list(SAFE_CHANNEL_SCORE_COLUMNS.values())
    out["channel_score_coverage"] = out[core].notna().all(axis=1).astype(float)
    out["channel_counter_gap_prob"] = out["channel_hard_counter_prob"] - out["channel_positive_support_prob"]
    out["channel_soft_or_hard_prob"] = out["channel_hard_counter_prob"] + out["channel_soft_gap_prob"]
    out["channel_positive_gap_prob"] = out["channel_positive_support_prob"] - out["channel_hard_counter_prob"]
    out["channel_hard_counter_yellow_flag"] = (
        (out["channel_hard_counter_prob"] >= 0.80) & (out["channel_hard_counter_prob"] < 0.95)
    ).astype(float)
    out["channel_hard_counter_high_flag"] = (out["channel_hard_counter_prob"] >= 0.95).astype(float)
    out["channel_soft_gap_dominant_flag"] = (
        (out["channel_soft_gap_prob"] >= out["channel_hard_counter_prob"])
        & (out["channel_soft_gap_prob"] >= out["channel_positive_support_prob"])
    ).astype(float)
    out = out[["date", "code", *CHANNEL_FEATURES]].drop_duplicates(["date", "code"], keep="last")
    forbidden = sorted(set(out.columns) & FUTURE_OR_RESULT_FIELDS)
    if forbidden:
        raise ValueError(f"future/result fields escaped safe channel loader: {forbidden}")
    return out, CHANNEL_FEATURES.copy()


def attach_channel_features(frame: pd.DataFrame, channel_scores: pd.DataFrame) -> pd.DataFrame:
    introduced_forbidden = sorted(set(channel_scores.columns) & FUTURE_OR_RESULT_FIELDS)
    if introduced_forbidden:
        raise ValueError(f"channel score frame contains future/result fields: {introduced_forbidden}")
    out = _norm_keys(frame).merge(channel_scores, on=["date", "code"], how="left")
    return out


def channel_only_opportunity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hard = pd.to_numeric(out.get("channel_hard_counter_prob"), errors="coerce").fillna(0.0)
    soft = pd.to_numeric(out.get("channel_soft_gap_prob"), errors="coerce").fillna(0.0)
    pos = pd.to_numeric(out.get("channel_positive_support_prob"), errors="coerce").fillna(0.0)
    neutral = pd.to_numeric(out.get("channel_neutral_prob"), errors="coerce").fillna(0.0)
    out["ml_score"] = pos - hard - 0.25 * soft + 0.10 * neutral
    return out


def channel_only_risk(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hard = pd.to_numeric(out.get("channel_hard_counter_prob"), errors="coerce").fillna(0.0)
    soft = pd.to_numeric(out.get("channel_soft_gap_prob"), errors="coerce").fillna(0.0)
    pos = pd.to_numeric(out.get("channel_positive_support_prob"), errors="coerce").fillna(0.0)
    out["risk_score"] = hard + 0.35 * soft - 0.25 * pos
    return out


def _append_opportunity_row(
    rows: list[dict[str, Any]],
    *,
    target_block: str,
    side: str,
    model: str,
    selected: pd.DataFrame,
    target: pd.DataFrame,
    base: dict[str, Any],
    threshold: float | None = None,
    selected_features: list[str] | None = None,
) -> None:
    rows.append(
        {
            "target_block": target_block,
            "side": side,
            "model": model,
            "threshold": round(float(threshold), 6) if threshold is not None and pd.notna(threshold) else np.nan,
            "selected_features": ",".join(selected_features or []),
            **base,
            **side_metrics(selected, base),
            **selection_hygiene(selected, target),
        }
    )


def _append_risk_row(
    rows: list[dict[str, Any]],
    *,
    target_block: str,
    side: str,
    model: str,
    selected: pd.DataFrame,
    target: pd.DataFrame,
    base: dict[str, Any],
    threshold: float | None = None,
    selected_features: list[str] | None = None,
) -> None:
    rows.append(
        {
            "target_block": target_block,
            "side": side,
            "model": model,
            "threshold": round(float(threshold), 6) if threshold is not None and pd.notna(threshold) else np.nan,
            "selected_features": ",".join(selected_features or []),
            "risk_recall": risk_recall(target, selected),
            **base,
            **side_metrics(selected, base),
            **selection_hygiene(selected, target),
            **loss_exposure_after_exclude(target, selected, base),
        }
    )


def _research_grade(row: pd.Series, opp_thr: float, risk_thr: float) -> str:
    coverage = float(pd.to_numeric(row.get("channel_score_coverage"), errors="coerce") or 0.0)
    if coverage < 1.0:
        return "信息不足"
    risk = float(pd.to_numeric(row.get("risk_score"), errors="coerce") or 0.0)
    opp = float(pd.to_numeric(row.get("opportunity_score"), errors="coerce") or 0.0)
    hard = float(pd.to_numeric(row.get("channel_hard_counter_prob"), errors="coerce") or 0.0)
    soft = float(pd.to_numeric(row.get("channel_soft_gap_prob"), errors="coerce") or 0.0)
    if risk >= risk_thr and hard >= 0.95:
        return "暂时剔除"
    if opp >= opp_thr and hard < 0.80:
        return "继续深挖"
    if hard >= 0.80 or soft >= 0.50:
        return "放入观察"
    return "放入观察"


def _risk_tier(row: pd.Series) -> str:
    hard = float(pd.to_numeric(row.get("channel_hard_counter_prob"), errors="coerce") or 0.0)
    soft = float(pd.to_numeric(row.get("channel_soft_gap_prob"), errors="coerce") or 0.0)
    pos = float(pd.to_numeric(row.get("channel_positive_support_prob"), errors="coerce") or 0.0)
    if hard >= 0.95:
        return "hard_counter_high_risk_review_ge_0.95"
    if hard >= 0.80:
        return "hard_counter_yellow_review_0.80_0.95"
    if soft >= hard and soft >= pos:
        return "soft_gap_dominant_low_hard"
    return "low_hard_counter_probability"


def _reject_agent_preview_leak(frame: pd.DataFrame) -> None:
    escaped = sorted(set(frame.columns) & FUTURE_OR_RESULT_FIELDS)
    if escaped:
        raise ValueError(f"agent preview contains future/result fields: {escaped}")
    text = " ".join(str(v) for v in frame.head(200).to_dict("records"))
    forbidden_terms = ["买入", "卖出", "强烈推荐", "目标价必达", "buy", "sell"]
    found = [term for term in forbidden_terms if term in text]
    if found:
        raise ValueError(f"agent preview contains disallowed instruction terms: {found}")


def build_agent_preview(
    target: pd.DataFrame,
    opp_scored: pd.DataFrame,
    risk_scored: pd.DataFrame,
    *,
    opp_threshold: float,
    risk_threshold: float,
) -> pd.DataFrame:
    cols = ["date", "code", "time_block", *CHANNEL_FEATURES]
    preview = target[[c for c in cols if c in target.columns]].copy()
    preview["tool_id"] = "single_stock_channel_scorer_v1"
    preview["tool_version"] = "channel_augmented_additive_bin_v1"
    preview["task_mode"] = "single_stock_watch"
    preview["policy_profile"] = "rolling_time_safe_channel_augmented"
    preview["decision_frequency"] = "scheduled_twice_weekly_or_key_points"
    preview["opportunity_score"] = pd.to_numeric(opp_scored["ml_score"], errors="coerce").round(6).to_numpy()
    preview["risk_score"] = pd.to_numeric(risk_scored["risk_score"], errors="coerce").round(6).to_numpy()
    preview["opportunity_threshold"] = round(float(opp_threshold), 6)
    preview["risk_threshold"] = round(float(risk_threshold), 6)
    preview["risk_tier"] = preview.apply(_risk_tier, axis=1)
    preview["research_grade"] = preview.apply(_research_grade, axis=1, args=(opp_threshold, risk_threshold))
    preview["required_confirmation"] = preview["risk_tier"].map(
        {
            "hard_counter_high_risk_review_ge_0.95": (
                "news_quality,financial_event,peer_weakness,bookskill_failure,chip_overhang"
            ),
            "hard_counter_yellow_review_0.80_0.95": (
                "at_least_two_independent_counter_evidence_channels"
            ),
            "soft_gap_dominant_low_hard": "missing_or_conflicting_channel_review",
            "low_hard_counter_probability": "normal_review",
        }
    )
    preview["source_ref_ids"] = "channel_rule_outcome_classifier_v1, single_stock_channel_scorer_v1"
    preview["research_only"] = True
    preview["not_investment_instruction"] = True
    _reject_agent_preview_leak(preview)
    return preview


def _variant_light(rows: pd.DataFrame, side: str) -> str:
    sub = rows[rows["side"] == side]
    h = sub[sub["target_block"] == FINAL_OOT]
    if h.empty:
        return "red"
    r = h.iloc[0]
    if "risk" in side:
        d_loss = float(r.get("delta_loss_vs_base", np.nan))
        d_mean = float(r.get("delta_mean_vs_base", np.nan))
        recall = float(r.get("risk_recall", np.nan))
        exp_red = float(r.get("loss_exposure_reduction", np.nan))
        prior = sub[sub["target_block"] != FINAL_OOT]["delta_loss_vs_base"].dropna()
        prior_hit = float((prior > 0).mean()) if len(prior) else 0.0
        if d_loss >= 0.03 and d_mean <= 0 and recall >= 0.15 and exp_red >= 0.01 and prior_hit >= 0.75:
            return "green"
        if d_loss > 0 and d_mean <= 0:
            return "yellow"
        return "red"
    d_pos = float(r.get("delta_pos_vs_base", np.nan))
    d_mean = float(r.get("delta_mean_vs_base", np.nan))
    prior = sub[sub["target_block"] != FINAL_OOT]["delta_pos_vs_base"].dropna()
    prior_hit = float((prior > 0).mean()) if len(prior) else 0.0
    if d_pos >= 0.03 and d_mean > 0 and prior_hit >= 0.75:
        return "green"
    if d_pos > 0 and d_mean > 0:
        return "yellow"
    return "red"


def _render_report(rows: pd.DataFrame, notes: list[str], preview_rows: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Single-stock channel scorer audit v1",
        "",
        f"> Generated: {ts} | Final OOT: `{FINAL_OOT}` | Agent preview rows: {preview_rows}",
        "",
        "This is research-only. Labels and forward returns are offline evaluation only.",
        "The agent preview file contains only whitelisted score fields and four user-facing grades.",
        "",
        "## Method",
        "",
        "- Baseline ML: existing rolling additive_bin opportunity/risk models.",
        "- Channel augmented ML: same models plus safe channel classifier probabilities.",
        "- Channel-only rule: interpretable hard/soft/positive probability formulas.",
        "- Thresholds are selected on the previous validation block; H2026_1 is target only.",
        "",
        "## Coverage notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- none"])
    lines.extend(
        [
            "",
            "## Opportunity comparison",
            "",
            "| side | H2026 pos | H2026 dpos | H2026 mean | H2026 dmean | light |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for side in ("opportunity_base_ml", "opportunity_channel_augmented_ml", "opportunity_channel_only_rule"):
        h = rows[(rows["side"] == side) & (rows["target_block"] == FINAL_OOT)]
        if h.empty:
            continue
        r = h.iloc[0]
        lines.append(
            f"| {side} | {r['positive_20d_rate']:.4f} | {r['delta_pos_vs_base']:+.4f} | "
            f"{r['avg_return_20d']:.4f} | {r['delta_mean_vs_base']:+.4f} | {_variant_light(rows, side)} |"
        )
    lines.extend(
        [
            "",
            "## Risk comparison",
            "",
            "| side | H2026 loss | H2026 dloss | H2026 mean | H2026 dmean | recall | exposure cut | light |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for side in ("risk_base_ml", "risk_channel_augmented_ml", "risk_channel_only_rule"):
        h = rows[(rows["side"] == side) & (rows["target_block"] == FINAL_OOT)]
        if h.empty:
            continue
        r = h.iloc[0]
        lines.append(
            f"| {side} | {r['loss_gt5_rate']:.4f} | {r['delta_loss_vs_base']:+.4f} | "
            f"{r['avg_return_20d']:.4f} | {r['delta_mean_vs_base']:+.4f} | "
            f"{r.get('risk_recall', np.nan):.4f} | {r.get('loss_exposure_reduction', np.nan):+.4f} | "
            f"{_variant_light(rows, side)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Promote only if channel augmentation improves the target block without weakening prior-block stability.",
            "- If channel-only is weaker than baseline, keep classifier probabilities as risk evidence rather than alpha.",
            "- The JSONL preview is a tool contract for DS/Kimi/Codex agents; it is not a trading instruction.",
            "",
            "## Outputs",
            f"- `{CSV_PATH.relative_to(ROOT)}`",
            f"- `{AGENT_PREVIEW_PATH.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("A-share research agent")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    merged, raw_feats, base_notes = load_merged_frame()
    channel_scores, channel_features = load_safe_channel_scores()
    merged = attach_channel_features(merged, channel_scores)
    channel_coverage = float(merged["channel_score_coverage"].fillna(0).mean()) if "channel_score_coverage" in merged else 0.0
    notes = [
        *(base_notes or []),
        f"safe channel score rows={len(channel_scores)}; join coverage={channel_coverage:.4f}",
        f"channel features={len(channel_features)}",
    ]

    all_feats = sorted(set(raw_feats + channel_features))
    rows: list[dict[str, Any]] = []
    h2026_preview: pd.DataFrame | None = None

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(merged, target_block)
        if len(train_base) < 500 or len(validation) < 200 or len(target) < 200:
            notes.append(
                f"skip {target_block}: train={len(train_base)} valid={len(validation)} target={len(target)}"
            )
            continue
        base = block_base_metrics(target)

        opp_base_model = fit_additive_bin_model(train_base, raw_feats, feature_group="single_stock_opportunity_base")
        opp_base_val = score_frame(validation, opp_base_model)
        opp_base_tgt = score_frame(target, opp_base_model)
        opp_base_thr, _ = choose_opportunity_threshold(opp_base_val)
        _append_opportunity_row(
            rows,
            target_block=target_block,
            side="opportunity_base_ml",
            model="additive_bin_base",
            selected=opp_base_tgt[opp_base_tgt["ml_score"] >= opp_base_thr],
            target=target,
            base=base,
            threshold=opp_base_thr,
            selected_features=opp_base_model.selected_features,
        )

        opp_channel_model = fit_additive_bin_model(
            train_base, all_feats, feature_group="single_stock_opportunity_channel_augmented"
        )
        opp_channel_val = score_frame(validation, opp_channel_model)
        opp_channel_tgt = score_frame(target, opp_channel_model)
        opp_channel_thr, _ = choose_opportunity_threshold(opp_channel_val)
        _append_opportunity_row(
            rows,
            target_block=target_block,
            side="opportunity_channel_augmented_ml",
            model="additive_bin_channel_augmented",
            selected=opp_channel_tgt[opp_channel_tgt["ml_score"] >= opp_channel_thr],
            target=target,
            base=base,
            threshold=opp_channel_thr,
            selected_features=opp_channel_model.selected_features,
        )

        opp_rule_val = channel_only_opportunity(validation)
        opp_rule_tgt = channel_only_opportunity(target)
        opp_rule_thr, _ = choose_opportunity_threshold(opp_rule_val)
        _append_opportunity_row(
            rows,
            target_block=target_block,
            side="opportunity_channel_only_rule",
            model="channel_probability_formula",
            selected=opp_rule_tgt[opp_rule_tgt["ml_score"] >= opp_rule_thr],
            target=target,
            base=base,
            threshold=opp_rule_thr,
            selected_features=["positive_support-hard_counter-0.25*soft_gap+0.10*neutral"],
        )

        risk_base_model = fit_risk_model(train_base, raw_feats)
        risk_base_val = score_risk(validation, risk_base_model)
        risk_base_tgt = score_risk(target, risk_base_model)
        risk_base_thr, _ = choose_risk_threshold(risk_base_val)
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_base_ml",
            model="additive_bin_loss_base",
            selected=risk_base_tgt[risk_base_tgt["risk_score"] >= risk_base_thr],
            target=target,
            base=base,
            threshold=risk_base_thr,
            selected_features=risk_base_model.selected_features,
        )

        risk_channel_model = fit_risk_model(train_base, all_feats)
        risk_channel_val = score_risk(validation, risk_channel_model)
        risk_channel_tgt = score_risk(target, risk_channel_model)
        risk_channel_thr, _ = choose_risk_threshold(risk_channel_val)
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_channel_augmented_ml",
            model="additive_bin_loss_channel_augmented",
            selected=risk_channel_tgt[risk_channel_tgt["risk_score"] >= risk_channel_thr],
            target=target,
            base=base,
            threshold=risk_channel_thr,
            selected_features=risk_channel_model.selected_features,
        )

        risk_rule_val = channel_only_risk(validation)
        risk_rule_tgt = channel_only_risk(target)
        risk_rule_thr, _ = choose_risk_threshold(risk_rule_val)
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_channel_only_rule",
            model="channel_probability_formula",
            selected=risk_rule_tgt[risk_rule_tgt["risk_score"] >= risk_rule_thr],
            target=target,
            base=base,
            threshold=risk_rule_thr,
            selected_features=["hard_counter+0.35*soft_gap-0.25*positive_support"],
        )

        if target_block == FINAL_OOT:
            h2026_preview = build_agent_preview(
                target,
                opp_channel_tgt,
                risk_channel_tgt,
                opp_threshold=opp_channel_thr,
                risk_threshold=risk_channel_thr,
            )

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    preview_rows = 0
    if h2026_preview is not None:
        preview_rows = len(h2026_preview)
        _reject_agent_preview_leak(h2026_preview)
        with AGENT_PREVIEW_PATH.open("w", encoding="utf-8") as handle:
            for record in h2026_preview.to_dict("records"):
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    REPORT_PATH.write_text(_render_report(out, notes, preview_rows), encoding="utf-8")

    print(f"rows={len(out)} blocks={out['target_block'].nunique() if not out.empty else 0}")
    print(f"report: {REPORT_PATH}")
    print(f"csv: {CSV_PATH}")
    print(f"agent_preview: {AGENT_PREVIEW_PATH} rows={preview_rows}")
    for side in (
        "opportunity_base_ml",
        "opportunity_channel_augmented_ml",
        "opportunity_channel_only_rule",
        "risk_base_ml",
        "risk_channel_augmented_ml",
        "risk_channel_only_rule",
    ):
        h = out[(out["target_block"] == FINAL_OOT) & (out["side"] == side)]
        if h.empty:
            continue
        r = h.iloc[0]
        if side.startswith("risk"):
            print(
                f"{side}: loss={r['loss_gt5_rate']:.4f} dloss={r['delta_loss_vs_base']:+.4f} "
                f"mean={r['avg_return_20d']:.4f} recall={r.get('risk_recall', np.nan):.4f} "
                f"exposure_cut={r.get('loss_exposure_reduction', np.nan):+.4f} light={_variant_light(out, side)}"
            )
        else:
            print(
                f"{side}: pos={r['positive_20d_rate']:.4f} dpos={r['delta_pos_vs_base']:+.4f} "
                f"mean={r['avg_return_20d']:.4f} dmean={r['delta_mean_vs_base']:+.4f} "
                f"light={_variant_light(out, side)}"
            )


if __name__ == "__main__":
    main()
