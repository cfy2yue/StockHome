"""Calibrate a capped single-stock risk review queue.

The v1 channel scorer showed useful H2026 risk recall but weak stability and
too much exposure in channel-only rules. This script keeps labels offline,
selects risk-review exposure caps on the previous validation block, and writes
only a compact H2026 review queue for agent use.
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

from scripts.audit_single_stock_channel_scorer_v1 import (  # noqa: E402
    CHANNEL_FEATURES,
    FUTURE_OR_RESULT_FIELDS,
    attach_channel_features,
    load_safe_channel_scores,
)
from scripts.audit_single_stock_review_quality import (  # noqa: E402
    FINAL_OOT,
    TARGET_BLOCKS,
    block_base_metrics,
    fit_risk_model,
    load_merged_frame,
    loss_exposure_after_exclude,
    risk_recall,
    score_risk,
    selection_hygiene,
    side_metrics,
)
from scripts.run_lightweight_ml_channel_experiment import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
REPORT_PATH = REPORT_DIR / "single_stock_risk_calibration_v2.md"
CSV_PATH = REPORT_DIR / "single_stock_risk_calibration_v2.csv"
QUEUE_PATH = REPORT_DIR / "single_stock_risk_calibration_v2_review_queue.jsonl"
FIXED15_QUEUE_PATH = REPORT_DIR / "single_stock_risk_calibration_v2_fixed15_candidate_queue.jsonl"

CAP_GRID = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
FIXED_DIAGNOSTIC_CAPS = (0.10, 0.15, 0.20)
MIN_REVIEW_ROWS = 80
MAX_DEFAULT_REVIEW_EXPOSURE = 0.20
MIN_VALIDATION_RECALL_TARGET = 0.15
MIN_VALIDATION_LOSS_LIFT = 0.03


def select_top_pct_per_date(frame: pd.DataFrame, score_col: str, pct: float) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("date", sort=False):
        if group.empty:
            continue
        k = max(1, int(np.ceil(len(group) * pct)))
        pieces.append(group.sort_values(score_col, ascending=False).head(k))
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def add_review_priority_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    risk = pd.to_numeric(out.get("risk_score"), errors="coerce").fillna(0.0)
    hard = pd.to_numeric(out.get("channel_hard_counter_prob"), errors="coerce").fillna(0.0)
    soft = pd.to_numeric(out.get("channel_soft_gap_prob"), errors="coerce").fillna(0.0)
    pos = pd.to_numeric(out.get("channel_positive_support_prob"), errors="coerce").fillna(0.0)
    high = pd.to_numeric(out.get("channel_hard_counter_high_flag"), errors="coerce").fillna(0.0)
    yellow = pd.to_numeric(out.get("channel_hard_counter_yellow_flag"), errors="coerce").fillna(0.0)
    soft_dom = pd.to_numeric(out.get("channel_soft_gap_dominant_flag"), errors="coerce").fillna(0.0)
    counter_gap = hard - pos
    out["review_priority_score"] = risk + 0.22 * hard + 0.06 * soft + 0.08 * high + 0.04 * yellow
    out["review_priority_score"] = out["review_priority_score"] + 0.04 * counter_gap.clip(lower=0) - 0.08 * soft_dom
    return out


def _risk_objective(selected: pd.DataFrame, pool: pd.DataFrame, base: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    metrics = side_metrics(selected, base)
    hygiene = selection_hygiene(selected, pool)
    exposure = float(hygiene.get("active_exposure", 0.0) or 0.0)
    recall = risk_recall(pool, selected)
    recall_value = 0.0 if pd.isna(recall) else float(recall)
    exposure_cut = loss_exposure_after_exclude(pool, selected, base)
    cut_value = exposure_cut.get("loss_exposure_reduction", np.nan)
    cut_value = 0.0 if pd.isna(cut_value) else float(cut_value)
    loss_lift = metrics.get("delta_loss_vs_base", np.nan)
    loss_lift = -1.0 if pd.isna(loss_lift) else float(loss_lift)
    mean_delta = metrics.get("delta_mean_vs_base", np.nan)
    mean_penalty = max(float(mean_delta), 0.0) / 10.0 if pd.notna(mean_delta) else 0.0
    score = 7.0 * loss_lift + 8.0 * cut_value + 3.0 * recall_value - 0.7 * exposure - mean_penalty
    return score, {**metrics, **hygiene, "risk_recall": recall, **exposure_cut}


def choose_capped_policy(validation: pd.DataFrame, score_col: str, cap_grid: tuple[float, ...] = CAP_GRID) -> tuple[float, dict[str, Any]]:
    base = block_base_metrics(validation)
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for pct in cap_grid:
        selected = select_top_pct_per_date(validation, score_col, pct)
        if len(selected) < MIN_REVIEW_ROWS:
            continue
        score, metrics = _risk_objective(selected, validation, base)
        candidates.append((score, pct, metrics))
    if not candidates:
        pct = min(cap_grid)
        selected = select_top_pct_per_date(validation, score_col, pct)
        _, metrics = _risk_objective(selected, validation, base)
        return pct, metrics
    stable = [
        item for item in candidates
        if float(item[2].get("risk_recall", 0.0) or 0.0) >= MIN_VALIDATION_RECALL_TARGET
        and float(item[2].get("delta_loss_vs_base", -1.0) or -1.0) >= MIN_VALIDATION_LOSS_LIFT
        and float(item[2].get("active_exposure", 1.0) or 1.0) <= MAX_DEFAULT_REVIEW_EXPOSURE
    ]
    best = max(stable or candidates, key=lambda item: item[0])
    return best[1], best[2]


def _append_risk_row(
    rows: list[dict[str, Any]],
    *,
    target_block: str,
    side: str,
    model: str,
    selected: pd.DataFrame,
    target: pd.DataFrame,
    base: dict[str, Any],
    selected_features: list[str] | None = None,
    cap_pct: float | None = None,
    validation_metrics: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "target_block": target_block,
            "side": side,
            "model": model,
            "cap_pct": round(float(cap_pct), 6) if cap_pct is not None and pd.notna(cap_pct) else np.nan,
            "selected_features": ",".join(selected_features or []),
            "risk_recall": risk_recall(target, selected),
            "validation_loss_gt5_rate": (validation_metrics or {}).get("loss_gt5_rate", np.nan),
            "validation_delta_loss_vs_base": (validation_metrics or {}).get("delta_loss_vs_base", np.nan),
            "validation_risk_recall": (validation_metrics or {}).get("risk_recall", np.nan),
            "validation_active_exposure": (validation_metrics or {}).get("active_exposure", np.nan),
            **base,
            **side_metrics(selected, base),
            **selection_hygiene(selected, target),
            **loss_exposure_after_exclude(target, selected, base),
        }
    )


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


def build_review_queue(selected: pd.DataFrame, *, cap_pct: float, policy_status: str) -> pd.DataFrame:
    allowed = [
        "date",
        "code",
        "time_block",
        "risk_score",
        "review_priority_score",
        *CHANNEL_FEATURES,
    ]
    queue = selected[[c for c in allowed if c in selected.columns]].copy()
    queue["tool_id"] = "single_stock_risk_calibration_v2"
    queue["tool_version"] = "capped_review_queue_v2"
    queue["task_mode"] = "single_stock_watch"
    queue["policy_profile"] = "risk_review_only"
    queue["decision_frequency"] = "scheduled_twice_weekly_or_key_points"
    queue["cap_pct"] = round(float(cap_pct), 6)
    queue["risk_tier"] = queue.apply(_risk_tier, axis=1)
    queue["research_grade"] = np.where(
        queue["risk_tier"].eq("hard_counter_high_risk_review_ge_0.95"),
        "暂时剔除",
        "放入观察",
    )
    queue["review_queue_reason"] = np.where(
        queue["risk_tier"].eq("hard_counter_high_risk_review_ge_0.95"),
        "high_hard_counter_and_capped_risk_queue",
        "capped_risk_review_queue_requires_cross_channel_confirmation",
    )
    queue["policy_status"] = policy_status
    queue["source_ref_ids"] = "single_stock_risk_calibration_v2, channel_rule_outcome_classifier_v1"
    queue["research_only"] = True
    queue["not_investment_instruction"] = True
    _reject_queue_leak(queue)
    return queue


def _reject_queue_leak(frame: pd.DataFrame) -> None:
    escaped = sorted(set(frame.columns) & FUTURE_OR_RESULT_FIELDS)
    if escaped:
        raise ValueError(f"review queue contains future/result fields: {escaped}")
    grades = set(frame.get("research_grade", pd.Series(dtype=str)).dropna().astype(str))
    allowed_grades = {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
    if not grades <= allowed_grades:
        raise ValueError(f"unexpected research grades: {sorted(grades - allowed_grades)}")
    text = " ".join(str(v) for v in frame.head(200).to_dict("records"))
    forbidden_terms = ["买入", "卖出", "强烈推荐", "目标价必达", "buy", "sell"]
    found = [term for term in forbidden_terms if term in text]
    if found:
        raise ValueError(f"review queue contains disallowed instruction terms: {found}")


def _stability_status(rows: pd.DataFrame, side: str) -> str:
    sub = rows[rows["side"] == side].copy()
    h = sub[sub["target_block"] == FINAL_OOT]
    prior = sub[sub["target_block"] != FINAL_OOT]
    if h.empty or prior.empty:
        return "insufficient"
    hrow = h.iloc[0]
    prior_loss_hit = float((pd.to_numeric(prior["delta_loss_vs_base"], errors="coerce") > 0.03).mean())
    prior_cut_hit = float((pd.to_numeric(prior["loss_exposure_reduction"], errors="coerce") > 0.005).mean())
    h_loss = float(hrow.get("delta_loss_vs_base", np.nan))
    h_recall = float(hrow.get("risk_recall", np.nan))
    h_cut = float(hrow.get("loss_exposure_reduction", np.nan))
    h_exp = float(hrow.get("active_exposure", np.nan))
    if (
        prior_loss_hit >= 0.75
        and prior_cut_hit >= 0.75
        and h_loss >= 0.03
        and h_recall >= 0.15
        and h_cut >= 0.01
        and h_exp <= MAX_DEFAULT_REVIEW_EXPOSURE
    ):
        return "green_review_candidate"
    if h_loss > 0 and h_cut > 0:
        return "yellow_review_only"
    return "red_not_promoted"


def _render_report(
    rows: pd.DataFrame,
    notes: list[str],
    queue_rows: int,
    queue_grade_counts: dict[str, int],
    fixed15_queue_rows: int = 0,
    fixed15_grade_counts: dict[str, int] | None = None,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Single-stock risk calibration v2",
        "",
        (
            f"> Generated: {ts} | Final OOT: `{FINAL_OOT}` | "
            f"review_queue_rows={queue_rows} | fixed15_candidate_rows={fixed15_queue_rows}"
        ),
        "",
        "Research-only. Future returns and labels are offline evaluation only; review queue JSONL is field-whitelisted.",
        "",
        "## Method",
        "",
        "- Keep v1 opportunity scorer unchanged; this file only calibrates risk-review queues.",
        "- Train base and channel-augmented additive_bin loss scorers with rolling prior blocks.",
        "- Select capped review percentage on the previous validation block, then apply it to the target block.",
        "- Compare model score top-percent and channel-aware review priority score.",
        "",
        "## Coverage notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- none"])
    lines.extend(
        [
            "",
            "## H2026 risk comparison",
            "",
            "| side | cap | loss | dloss | mean | dmean | recall | exposure cut | active | status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    h2026_sides = rows[rows["target_block"] == FINAL_OOT]["side"].drop_duplicates().tolist()
    for side in h2026_sides:
        h = rows[(rows["side"] == side) & (rows["target_block"] == FINAL_OOT)]
        if h.empty:
            continue
        r = h.iloc[0]
        lines.append(
            f"| {side} | {r.get('cap_pct', np.nan):.3f} | {r['loss_gt5_rate']:.4f} | {r['delta_loss_vs_base']:+.4f} | "
            f"{r['avg_return_20d']:.4f} | {r['delta_mean_vs_base']:+.4f} | "
            f"{r.get('risk_recall', np.nan):.4f} | {r.get('loss_exposure_reduction', np.nan):+.4f} | "
            f"{r.get('active_exposure', np.nan):.4f} | {_stability_status(rows, side)} |"
        )
    lines.extend(["", "## Review queue preview", ""])
    for grade, count in queue_grade_counts.items():
        lines.append(f"- {grade}: {count}")
    if fixed15_grade_counts:
        lines.extend(["", "## Fixed15 candidate queue preview", ""])
        for grade, count in fixed15_grade_counts.items():
            lines.append(f"- {grade}: {count}")
    fixed15 = rows[(rows["target_block"] == FINAL_OOT) & (rows["side"] == "risk_channel_priority_fixed_15pct")]
    if not fixed15.empty:
        r = fixed15.iloc[0]
        lines.extend(
            [
                "",
                "## Candidate Note",
                "",
                (
                    "- `risk_channel_priority_fixed_15pct` is the best current Pareto candidate "
                    f"(H2026 recall={r.get('risk_recall', np.nan):.4f}, "
                    f"exposure cut={r.get('loss_exposure_reduction', np.nan):+.4f}, "
                    f"active={r.get('active_exposure', np.nan):.4f}), but it is identified after this audit "
                    "and must be treated as a next-OOT candidate, not retroactive H2026 default proof."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A capped queue can be useful if it preserves H2026 recall/loss exposure improvement without reviewing most of the universe.",
            "- Green requires prior-block stability plus H2026 risk lift; otherwise the queue stays review-only.",
            "- The queue is not a trading instruction and does not override Agent cross-channel review.",
            "",
            "## Outputs",
            f"- `{CSV_PATH.relative_to(ROOT)}`",
            f"- `{QUEUE_PATH.relative_to(ROOT)}`",
            f"- `{FIXED15_QUEUE_PATH.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines)


def _write_queue_jsonl(queue: pd.DataFrame, path: Path) -> tuple[int, dict[str, int]]:
    _reject_queue_leak(queue)
    grade_counts = {str(k): int(v) for k, v in queue["research_grade"].value_counts().to_dict().items()}
    with path.open("w", encoding="utf-8") as handle:
        for record in queue.to_dict("records"):
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return len(queue), grade_counts


def main() -> None:
    print("A-share research agent")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    merged, raw_feats, base_notes = load_merged_frame()
    channel_scores, channel_features = load_safe_channel_scores()
    merged = attach_channel_features(merged, channel_scores)
    all_feats = sorted(set(raw_feats + channel_features))
    channel_coverage = float(merged["channel_score_coverage"].fillna(0).mean()) if "channel_score_coverage" in merged else 0.0
    notes = [
        *(base_notes or []),
        f"safe channel score rows={len(channel_scores)}; join coverage={channel_coverage:.4f}",
        f"cap_grid={CAP_GRID}; validation recall target={MIN_VALIDATION_RECALL_TARGET}; fixed diagnostic caps={FIXED_DIAGNOSTIC_CAPS}",
    ]
    rows: list[dict[str, Any]] = []
    final_queue: pd.DataFrame | None = None
    fixed15_queue: pd.DataFrame | None = None
    final_cap = float("nan")

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(merged, target_block)
        if len(train_base) < 500 or len(validation) < 200 or len(target) < 200:
            notes.append(f"skip {target_block}: train={len(train_base)} valid={len(validation)} target={len(target)}")
            continue
        base = block_base_metrics(target)

        base_model = fit_risk_model(train_base, raw_feats)
        val_base = score_risk(validation, base_model)
        tgt_base = score_risk(target, base_model)
        base_pct, base_val_metrics = choose_capped_policy(val_base, "risk_score")
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_base_uncapped_top_pct",
            model="base_risk_score_validation_pct",
            selected=select_top_pct_per_date(tgt_base, "risk_score", base_pct),
            target=target,
            base=base,
            selected_features=base_model.selected_features,
            cap_pct=base_pct,
            validation_metrics=base_val_metrics,
        )

        channel_model = fit_risk_model(train_base, all_feats)
        val_channel = score_risk(validation, channel_model)
        tgt_channel = score_risk(target, channel_model)
        channel_pct, channel_val_metrics = choose_capped_policy(val_channel, "risk_score")
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_channel_uncapped_top_pct",
            model="channel_risk_score_validation_pct",
            selected=select_top_pct_per_date(tgt_channel, "risk_score", channel_pct),
            target=target,
            base=base,
            selected_features=channel_model.selected_features,
            cap_pct=channel_pct,
            validation_metrics=channel_val_metrics,
        )

        capped_pct = min(channel_pct, MAX_DEFAULT_REVIEW_EXPOSURE)
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_channel_capped_top_pct",
            model="channel_risk_score_capped_max20pct",
            selected=select_top_pct_per_date(tgt_channel, "risk_score", capped_pct),
            target=target,
            base=base,
            selected_features=channel_model.selected_features,
            cap_pct=capped_pct,
            validation_metrics=channel_val_metrics,
        )

        val_priority = add_review_priority_score(val_channel)
        tgt_priority = add_review_priority_score(tgt_channel)
        priority_pct, priority_val_metrics = choose_capped_policy(val_priority, "review_priority_score")
        priority_pct = min(priority_pct, MAX_DEFAULT_REVIEW_EXPOSURE)
        priority_sel = select_top_pct_per_date(tgt_priority, "review_priority_score", priority_pct)
        _append_risk_row(
            rows,
            target_block=target_block,
            side="risk_channel_priority_capped",
            model="channel_review_priority_capped_max20pct",
            selected=priority_sel,
            target=target,
            base=base,
            selected_features=[
                "risk_score",
                "hard_counter_prob",
                "soft_gap_prob",
                "positive_support_prob",
                "hard_counter_flags",
            ],
            cap_pct=priority_pct,
            validation_metrics=priority_val_metrics,
        )
        for fixed_cap in FIXED_DIAGNOSTIC_CAPS:
            fixed_cap = min(fixed_cap, MAX_DEFAULT_REVIEW_EXPOSURE)
            fixed_sel = select_top_pct_per_date(tgt_priority, "review_priority_score", fixed_cap)
            fixed_val_sel = select_top_pct_per_date(val_priority, "review_priority_score", fixed_cap)
            _, fixed_val_metrics = _risk_objective(fixed_val_sel, validation, block_base_metrics(validation))
            _append_risk_row(
                rows,
                target_block=target_block,
                side=f"risk_channel_priority_fixed_{int(round(fixed_cap * 100)):02d}pct",
                model="channel_review_priority_fixed_cap",
                selected=fixed_sel,
                target=target,
                base=base,
                selected_features=[
                    "risk_score",
                    "hard_counter_prob",
                    "soft_gap_prob",
                    "positive_support_prob",
                    "hard_counter_flags",
                ],
                cap_pct=fixed_cap,
                validation_metrics=fixed_val_metrics,
            )
            if target_block == FINAL_OOT and abs(fixed_cap - 0.15) < 1e-9:
                fixed15_queue = build_review_queue(
                    fixed_sel,
                    cap_pct=fixed_cap,
                    policy_status="fixed15_next_oot_candidate_not_retroactive_default",
                )

        if target_block == FINAL_OOT:
            final_queue = build_review_queue(priority_sel, cap_pct=priority_pct, policy_status="validation_selected_review_only")
            final_cap = priority_pct

    out = pd.DataFrame(rows)
    out.to_csv(CSV_PATH, index=False)
    queue_grade_counts: dict[str, int] = {}
    queue_rows = 0
    if final_queue is not None:
        queue_rows, queue_grade_counts = _write_queue_jsonl(final_queue, QUEUE_PATH)
    fixed15_grade_counts: dict[str, int] = {}
    fixed15_queue_rows = 0
    if fixed15_queue is not None:
        fixed15_queue_rows, fixed15_grade_counts = _write_queue_jsonl(fixed15_queue, FIXED15_QUEUE_PATH)
    REPORT_PATH.write_text(
        _render_report(out, notes, queue_rows, queue_grade_counts, fixed15_queue_rows, fixed15_grade_counts),
        encoding="utf-8",
    )

    print(f"rows={len(out)} blocks={out['target_block'].nunique() if not out.empty else 0}")
    print(f"report: {REPORT_PATH}")
    print(f"csv: {CSV_PATH}")
    print(f"review_queue: {QUEUE_PATH} rows={queue_rows} cap={final_cap}")
    print(f"fixed15_candidate_queue: {FIXED15_QUEUE_PATH} rows={fixed15_queue_rows} cap=0.15")
    for side in (
        "risk_base_uncapped_top_pct",
        "risk_channel_uncapped_top_pct",
        "risk_channel_capped_top_pct",
        "risk_channel_priority_capped",
        "risk_channel_priority_fixed_10pct",
        "risk_channel_priority_fixed_15pct",
        "risk_channel_priority_fixed_20pct",
    ):
        h = out[(out["target_block"] == FINAL_OOT) & (out["side"] == side)]
        if h.empty:
            continue
        r = h.iloc[0]
        print(
            f"{side}: loss={r['loss_gt5_rate']:.4f} dloss={r['delta_loss_vs_base']:+.4f} "
            f"mean={r['avg_return_20d']:.4f} recall={r.get('risk_recall', np.nan):.4f} "
            f"exposure_cut={r.get('loss_exposure_reduction', np.nan):+.4f} "
            f"active={r.get('active_exposure', np.nan):.4f} status={_stability_status(out, side)}"
        )


if __name__ == "__main__":
    main()
