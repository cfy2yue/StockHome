"""Audit whether financial safety hygiene should be a hard gate or review context.

Future returns are used only for this offline evaluation report. The output is
not an Agent evidence source and must not be used as a live decision feature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_DETAIL_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000" / "financial_asof_window_expansion_detail_cache.csv.gz"
DEFAULT_OUTPUT_PREFIX = "financial_safety_hygiene_gate_audit_v1"
DEFAULT_WINDOWS = [30, 60, 90, 180, 365]
DEFAULT_SCOPE = "high_ranker_q0.80"
MAX_CONCENTRATION = 0.35
MIN_TOTAL_ROWS = 30
MIN_H2026_ROWS = 8

VALID_BLOCKS = list(TIME_BLOCKS.keys())[1:]


GATE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "gate_id": "baseline",
        "gate_type": "baseline",
        "description": "all rows in the selected scope",
        "mask": lambda frame: pd.Series(True, index=frame.index),
    },
    {
        "gate_id": "exclude_high_risk",
        "gate_type": "filter",
        "description": "drop rows with financial_high_risk_guard",
        "mask": lambda frame: ~flag(frame, "financial_high_risk_guard"),
    },
    {
        "gate_id": "require_quality_low_risk",
        "gate_type": "positive_filter",
        "description": "keep rows with financial_quality_low_risk",
        "mask": lambda frame: flag(frame, "financial_quality_low_risk"),
    },
    {
        "gate_id": "require_positive_surprise",
        "gate_type": "positive_filter",
        "description": "keep rows with financial_positive_surprise_low_risk",
        "mask": lambda frame: flag(frame, "financial_positive_surprise_low_risk"),
    },
    {
        "gate_id": "quality_and_no_high_risk",
        "gate_type": "positive_filter",
        "description": "keep low-risk quality rows and exclude financial_high_risk_guard",
        "mask": lambda frame: flag(frame, "financial_quality_low_risk") & ~flag(frame, "financial_high_risk_guard"),
    },
    {
        "gate_id": "high_risk_subset",
        "gate_type": "review_subset",
        "description": "diagnostic subset with financial_high_risk_guard",
        "mask": lambda frame: flag(frame, "financial_high_risk_guard"),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit financial safety hygiene gates from cached as-of detail.")
    parser.add_argument("--detail-cache", type=Path, default=DEFAULT_DETAIL_CACHE)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--windows", default=",".join(str(item) for item in DEFAULT_WINDOWS))
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    windows = [int(item.strip()) for item in str(args.windows).split(",") if item.strip()]
    detail = load_detail_cache(args.detail_cache, windows=windows, scope=args.scope)
    step_metrics = evaluate_gate_variants(detail)
    aggregate = aggregate_gate_metrics(step_metrics)
    paths = write_outputs(args.output_prefix, step_metrics, aggregate, args.detail_cache, args.scope, windows)

    print("A股研究Agent")
    print(f"detail_rows={len(detail)}")
    print(f"step_rows={len(step_metrics)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"report={paths['report']}")


def load_detail_cache(path: Path, *, windows: list[int], scope: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing detail cache: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["window_days"] = pd.to_numeric(frame["window_days"], errors="coerce").astype("Int64")
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame[frame["window_days"].isin(windows) & frame["scope"].astype(str).eq(scope)].copy()
    frame = frame[frame["time_block"].isin(VALID_BLOCKS)].dropna(subset=["date", "code", "return_20d"]).copy()
    if frame.empty:
        raise ValueError("detail cache has no rows for requested windows/scope")
    return frame.reset_index(drop=True)


def flag(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.get(col, pd.Series(False, index=frame.index)).fillna(False).astype(bool)


def evaluate_gate_variants(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (window, scope), frame in detail.groupby(["window_days", "scope"], sort=True):
        frame = frame.copy()
        frame["date_pool_return_20d"] = frame.groupby("date")["return_20d"].transform("mean")
        baselines = baseline_by_block(frame)
        for gate in GATE_DEFINITIONS:
            selected = frame[gate["mask"](frame)].copy()
            for block in VALID_BLOCKS:
                block_selected = selected[selected["time_block"].eq(block)].copy()
                rows.append(evaluate_selection(int(window), str(scope), gate, str(block), block_selected, baselines.get(block, {})))
    return pd.DataFrame(rows)


def baseline_by_block(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for block, group in frame.groupby("time_block", sort=True):
        returns = pd.to_numeric(group["return_20d"], errors="coerce")
        out[str(block)] = {
            "baseline_rows": float(len(group)),
            "baseline_positive_20d_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            "baseline_loss_gt5_rate": float((returns <= -5).mean()) if len(returns) else np.nan,
            "baseline_avg_return_20d": float(returns.mean()) if len(returns) else np.nan,
        }
    return out


def evaluate_selection(window: int, scope: str, gate: dict[str, Any], block: str, selected: pd.DataFrame, baseline: dict[str, float]) -> dict[str, Any]:
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    excess = returns - pd.to_numeric(selected.get("date_pool_return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    selected_pos = float((returns > 0).mean()) if len(returns) else np.nan
    selected_loss = float((returns <= -5).mean()) if len(returns) else np.nan
    selected_avg = float(returns.mean()) if len(returns) else np.nan
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if len(selected) else np.nan
    base_pos = baseline.get("baseline_positive_20d_rate", np.nan)
    base_loss = baseline.get("baseline_loss_gt5_rate", np.nan)
    base_avg = baseline.get("baseline_avg_return_20d", np.nan)
    return {
        "window_days": int(window),
        "scope": scope,
        "gate_id": str(gate["gate_id"]),
        "gate_type": str(gate["gate_type"]),
        "description": str(gate["description"]),
        "valid_block": block,
        "selected_rows": int(len(selected)),
        "unique_stocks": int(selected["code"].nunique()) if len(selected) else 0,
        "coverage_dates": int(selected["date"].nunique()) if len(selected) else 0,
        "top_stock_concentration": _round(float(concentration)) if not pd.isna(concentration) else np.nan,
        "selected_positive_20d_rate": _round(selected_pos),
        "selected_avg_return_20d": _round(selected_avg),
        "selected_loss_gt5_rate": _round(selected_loss),
        "selected_pool_excess_20d": _round(float(excess.mean())) if len(excess) else np.nan,
        "baseline_positive_20d_rate": _round(base_pos),
        "baseline_avg_return_20d": _round(base_avg),
        "baseline_loss_gt5_rate": _round(base_loss),
        "positive_rate_lift_vs_baseline": _round(selected_pos - base_pos) if not pd.isna(selected_pos) and not pd.isna(base_pos) else np.nan,
        "avg_return_lift_vs_baseline": _round(selected_avg - base_avg) if not pd.isna(selected_avg) and not pd.isna(base_avg) else np.nan,
        "loss_gt5_lift_vs_baseline": _round(selected_loss - base_loss) if not pd.isna(selected_loss) and not pd.isna(base_loss) else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def aggregate_gate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (window, scope, gate_id), group in metrics.groupby(["window_days", "scope", "gate_id"], sort=True):
        prior = group[~group["valid_block"].eq("H2026_1")]
        h2026 = group[group["valid_block"].eq("H2026_1")]
        row = {
            "window_days": int(window),
            "scope": scope,
            "gate_id": gate_id,
            "gate_type": str(group["gate_type"].iloc[0]),
            "description": str(group["description"].iloc[0]),
            "total_selected_rows": int(group["selected_rows"].sum()),
            "h2026_selected_rows": int(h2026["selected_rows"].sum()) if not h2026.empty else 0,
            "prior_avg_return_lift": _mean(prior, "avg_return_lift_vs_baseline"),
            "h2026_avg_return_lift": _mean(h2026, "avg_return_lift_vs_baseline"),
            "prior_positive_rate_lift": _mean(prior, "positive_rate_lift_vs_baseline"),
            "h2026_positive_rate_lift": _mean(h2026, "positive_rate_lift_vs_baseline"),
            "prior_loss_gt5_lift": _mean(prior, "loss_gt5_lift_vs_baseline"),
            "h2026_loss_gt5_lift": _mean(h2026, "loss_gt5_lift_vs_baseline"),
            "prior_pool_excess_20d": _mean(prior, "selected_pool_excess_20d"),
            "h2026_pool_excess_20d": _mean(h2026, "selected_pool_excess_20d"),
            "max_top_stock_concentration": _max(group, "top_stock_concentration"),
            "policy_status": "",
            "research_only": True,
            "not_investment_instruction": True,
        }
        row["policy_status"] = policy_status(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["policy_status", "window_days", "gate_id"], ascending=[True, True, True])


def policy_status(row: dict[str, Any]) -> str:
    gate_type = str(row.get("gate_type") or "")
    if str(row.get("gate_id")) == "baseline":
        return "baseline_reference"
    if int(row.get("total_selected_rows") or 0) < MIN_TOTAL_ROWS or int(row.get("h2026_selected_rows") or 0) < MIN_H2026_ROWS:
        return "reject_too_few_samples"
    concentration = _num(row.get("max_top_stock_concentration"))
    if not pd.isna(concentration) and concentration > MAX_CONCENTRATION:
        return "reject_concentrated"
    prior_avg = _num(row.get("prior_avg_return_lift"))
    h_avg = _num(row.get("h2026_avg_return_lift"))
    prior_pos = _num(row.get("prior_positive_rate_lift"))
    h_pos = _num(row.get("h2026_positive_rate_lift"))
    prior_loss = _num(row.get("prior_loss_gt5_lift"))
    h_loss = _num(row.get("h2026_loss_gt5_lift"))
    if gate_type == "filter" and prior_avg >= 0.10 and h_avg >= 0.10 and prior_pos >= 0 and h_pos >= 0 and prior_loss <= 0 and h_loss <= 0:
        return "accepted_filter_candidate_needs_ds_panel"
    if gate_type == "review_subset" and prior_avg <= -0.25 and h_avg <= -0.25 and prior_loss >= 0.02 and h_loss >= 0.02:
        return "accepted_review_no_raise_candidate"
    if gate_type == "review_subset" and (prior_avg < 0 or h_avg < 0):
        return "observe_review_only"
    if gate_type in {"filter", "positive_filter"} and prior_avg > 0 and h_avg > 0:
        return "observe_filter_candidate"
    return "rejected_or_diagnostic_only"


def write_outputs(prefix: str, step_metrics: pd.DataFrame, aggregate: pd.DataFrame, detail_cache: Path, scope: str, windows: list[int]) -> dict[str, Path]:
    step_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    step_metrics.to_csv(step_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(aggregate, step_metrics, detail_cache, scope, windows), encoding="utf-8")
    return {"step_metrics": step_path, "aggregate": aggregate_path, "report": report_path}


def render_report(aggregate: pd.DataFrame, step_metrics: pd.DataFrame, detail_cache: Path, scope: str, windows: list[int]) -> str:
    return "\n".join(
        [
            "# Financial Safety Hygiene Gate Audit v1",
            "",
            "本报告只用于 A 股研究辅助的离线评估，不构成投资建议，不自动交易，不接券商接口。",
            "",
            "## Setup",
            "",
            f"- detail_cache: `{detail_cache.relative_to(ROOT) if detail_cache.is_absolute() and ROOT in detail_cache.parents else detail_cache}`",
            f"- scope: `{scope}`",
            f"- windows: `{','.join(str(item) for item in windows)}`",
            "- future 20d returns are used only in this offline report; this report is not an Agent evidence source.",
            "",
            "## Aggregate",
            "",
            table(aggregate),
            "",
            "## Step Metrics",
            "",
            table(step_metrics),
            "",
            "## Interpretation",
            "",
            "- `accepted_filter_candidate_needs_ds_panel` 才表示过滤 gate 值得进入小 DS 面板；它仍不是默认规则。",
            "- `accepted_review_no_raise_candidate` 只能作为财报复核/不升权 checklist，不是硬过滤或硬降级。",
            "- 若过滤 gate 只带来很小改善或块间方向不一致，应保持为 review context，避免把 safety hygiene 误当 alpha。",
            "",
        ]
    )


def _mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return _round(float(values.mean())) if values.notna().any() else np.nan


def _max(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return _round(float(values.max())) if values.notna().any() else np.nan


def _round(value: float) -> float:
    return round(float(value), 6) if value is not None and not pd.isna(value) else np.nan


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
