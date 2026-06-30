"""Audit financial/announcement as-of window choices without DS calls.

Future returns are used only in offline reports. Agent-facing previews contain
rule conditions and policy status only, never realized returns or GT labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    _portfolio_ranker_details,
    load_ground_truth,
)
from src.world_model.financial_report_channel import merge_financial_report_features_asof  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_FINANCIAL_FEATURES = ROOT / "data" / "date_generalization_cache" / "market_5000" / "financial_report_features.csv"
DEFAULT_DETAIL_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000" / "financial_asof_window_expansion_detail_cache.csv.gz"
DEFAULT_DETAIL_CACHE_META = ROOT / "data" / "date_generalization_cache" / "market_5000" / "financial_asof_window_expansion_detail_cache.meta.json"
DEFAULT_OUTPUT_PREFIX = "financial_asof_window_expansion_v1"
DEFAULT_WINDOWS = [30, 60, 90, 180, 365]
HIGH_RANKER_QUANTILE = 0.80
MIN_SELECTED_ROWS = 30
MIN_H2026_ROWS = 8
MAX_TOP_STOCK_CONCENTRATION = 0.35
DETAIL_CACHE_VERSION = "financial_asof_window_detail_cache_v1"
DEFAULT_GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]

FUTURE_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "gt_status",
    "gt_pass",
    "rule_outcome_label",
    "avg_return_20d",
    "positive_20d_rate",
}


RULES: list[dict[str, Any]] = [
    {
        "rule_id": "financial_any_event_asof_v1",
        "direction": "coverage",
        "description": "any time-safe financial/announcement event is visible in the as-of window",
        "required_flags": ["financial_event_matched"],
    },
    {
        "rule_id": "financial_quality_low_risk_v1",
        "direction": "positive",
        "description": "material financial/announcement event with high disclosure quality and low quality risk",
        "required_flags": ["financial_quality_low_risk"],
    },
    {
        "rule_id": "financial_positive_surprise_low_risk_v1",
        "direction": "positive",
        "description": "positive surprise financial event with low quality risk and good disclosure quality",
        "required_flags": ["financial_positive_surprise_low_risk"],
    },
    {
        "rule_id": "financial_multi_event_review_v1",
        "direction": "review",
        "description": "multiple recent financial/announcement events require explicit review",
        "required_flags": ["financial_multi_event_review"],
    },
    {
        "rule_id": "financial_high_risk_guard_v1",
        "direction": "risk",
        "description": "material financial event with high quality risk, negative surprise, correction, inquiry, or audit-risk wording",
        "required_flags": ["financial_high_risk_guard"],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit financial as-of window expansion.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument(
        "--ground-truth-sources",
        type=Path,
        nargs="*",
        default=DEFAULT_GT_SOURCES,
        help="Ground-truth source CSVs used to rebuild the default joined cache when needed.",
    )
    parser.add_argument("--financial-features", type=Path, default=DEFAULT_FINANCIAL_FEATURES)
    parser.add_argument("--detail-cache", type=Path, default=DEFAULT_DETAIL_CACHE, help="Internal offline detail cache. Use --no-detail-cache to disable.")
    parser.add_argument("--no-detail-cache", action="store_true", help="Disable reading/writing the internal offline detail cache.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--windows", default=",".join(str(item) for item in DEFAULT_WINDOWS))
    parser.add_argument("--high-ranker-quantile", type=float, default=HIGH_RANKER_QUANTILE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    windows = [int(item.strip()) for item in args.windows.split(",") if item.strip()]
    base = load_base_frame(args.joined_cache, ground_truth_sources=args.ground_truth_sources)
    features = load_financial_features(args.financial_features)
    detail, cache_status = load_or_build_window_detail(
        base,
        features,
        windows=windows,
        high_ranker_quantile=args.high_ranker_quantile,
        joined_cache_path=args.joined_cache,
        financial_features_path=args.financial_features,
        detail_cache_path=None if args.no_detail_cache else args.detail_cache,
    )
    metrics = evaluate_rules(detail)
    aggregate = aggregate_metrics(metrics)
    coverage = coverage_summary(detail)
    previews = build_agent_previews(
        aggregate,
        report_ref=f"reports/date_generalization/{args.output_prefix}.md",
    )
    paths = write_outputs(
        prefix=args.output_prefix,
        detail=detail,
        metrics=metrics,
        aggregate=aggregate,
        coverage=coverage,
        previews=previews,
    )

    print("A股研究Agent")
    print(f"base_rows={len(base)}")
    print(f"detail_rows={len(detail)}")
    print(f"metric_rows={len(metrics)}")
    print(f"detail_cache_status={cache_status}")
    print(f"report={paths['report']}")
    print(f"agent_preview={paths['agent_preview']}")


def load_base_frame(path: Path, *, ground_truth_sources: list[Path] | tuple[Path, ...] | None = None) -> pd.DataFrame:
    if Path(path).resolve() == DEFAULT_JOINED_GT_CACHE_PATH.resolve():
        frame = load_ground_truth(
            ground_truth_sources or DEFAULT_GT_SOURCES,
            kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
            corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
            tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
            chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
        )
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame.dropna(subset=["date", "code", "return_20d"]).copy()
    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="financial_asof_window_expansion",
        decision_frequency="every_2_weeks",
    )
    frame["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].copy()
    keep_cols = [
        "date",
        "code",
        "name",
        "time_block",
        "return_20d",
        "rev_chip_score_quantile",
        "news_missing_rate",
        "official_confirmation_score",
        "announcement_materiality_score",
        "news_warning_score",
        "tushare_industry",
        "tushare_area",
    ]
    return frame[[col for col in keep_cols if col in frame]].reset_index(drop=True)


def load_financial_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    if "code" in frame:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def build_window_detail(
    base: pd.DataFrame,
    features: pd.DataFrame,
    *,
    windows: list[int],
    high_ranker_quantile: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    decisions = base[["date", "code"]].copy()
    for window in windows:
        merged = merge_financial_report_features_asof(decisions, features, window_days=window)
        merged = merged.drop(columns=[col for col in ["date", "code"] if col in merged], errors="ignore")
        data = pd.concat([base.reset_index(drop=True), merged.reset_index(drop=True)], axis=1)
        data["window_days"] = int(window)
        data["scope"] = "all_pool"
        data["high_ranker_threshold"] = float(high_ranker_quantile)
        rows.append(add_financial_flags(data))
        high = data[data["rev_chip_score_quantile"].ge(high_ranker_quantile)].copy()
        high["scope"] = f"high_ranker_q{high_ranker_quantile:.2f}"
        rows.append(add_financial_flags(high))
    return pd.concat(rows, ignore_index=True)


def load_or_build_window_detail(
    base: pd.DataFrame,
    features: pd.DataFrame,
    *,
    windows: list[int],
    high_ranker_quantile: float,
    joined_cache_path: Path,
    financial_features_path: Path,
    detail_cache_path: Path | None,
) -> tuple[pd.DataFrame, str]:
    if detail_cache_path is None:
        return build_window_detail(base, features, windows=windows, high_ranker_quantile=high_ranker_quantile), "disabled"
    meta_path = _detail_cache_meta_path(detail_cache_path)
    expected_meta = _detail_cache_metadata(
        joined_cache_path=joined_cache_path,
        financial_features_path=financial_features_path,
        windows=windows,
        high_ranker_quantile=high_ranker_quantile,
        base_rows=len(base),
        feature_rows=len(features),
    )
    cached = _read_detail_cache(detail_cache_path, meta_path, expected_meta)
    if cached is not None:
        return cached, "hit"
    detail = build_window_detail(base, features, windows=windows, high_ranker_quantile=high_ranker_quantile)
    _write_detail_cache(detail, detail_cache_path, meta_path, expected_meta)
    return detail, "miss_rebuilt"


def _read_detail_cache(path: Path, meta_path: Path, expected_meta: dict[str, Any]) -> pd.DataFrame | None:
    if not path.exists() or not meta_path.exists():
        return None
    try:
        current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if current_meta != expected_meta:
        return None
    try:
        detail = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    except Exception:
        return None
    detail.columns = [col.lstrip("\ufeff") for col in detail.columns]
    if "code" in detail:
        detail["code"] = detail["code"].astype(str).str.zfill(6)
    if "date" in detail:
        detail["date"] = pd.to_datetime(detail["date"], errors="coerce").dt.date.astype(str)
    return detail


def _write_detail_cache(detail: pd.DataFrame, path: Path, meta_path: Path, metadata: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        detail.to_csv(tmp_path, index=False, encoding="utf-8-sig", compression="gzip")
        tmp_path.replace(path)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _detail_cache_meta_path(path: Path) -> Path:
    if path == DEFAULT_DETAIL_CACHE:
        return DEFAULT_DETAIL_CACHE_META
    return path.with_suffix(path.suffix + ".meta.json")


def _detail_cache_metadata(
    *,
    joined_cache_path: Path,
    financial_features_path: Path,
    windows: list[int],
    high_ranker_quantile: float,
    base_rows: int,
    feature_rows: int,
) -> dict[str, Any]:
    return {
        "cache_version": DETAIL_CACHE_VERSION,
        "offline_eval_only": True,
        "contains_future_outcomes": True,
        "not_agent_facing": True,
        "joined_cache": _optional_file_fingerprint(joined_cache_path),
        "financial_features": _optional_file_fingerprint(financial_features_path),
        "windows": [int(item) for item in windows],
        "high_ranker_quantile": float(high_ranker_quantile),
        "base_rows": int(base_rows),
        "feature_rows": int(feature_rows),
    }


def _optional_file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def add_financial_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    status = out.get("financial_report_join_status", pd.Series("", index=out.index)).fillna("").astype(str)
    event_types = out.get("financial_report_event_types", pd.Series("", index=out.index)).fillna("").astype(str)
    event_count = num(out, "financial_report_event_count", 0.0)
    materiality = num(out, "financial_report_materiality_score", 0.0)
    risk = num(out, "financial_quality_risk_score", 0.0)
    surprise = num(out, "financial_surprise_score", 0.0)
    disclosure = num(out, "financial_disclosure_quality_score", 0.0)
    missing = num(out, "financial_report_missing_rate", 1.0)
    event_matched = status.eq("event_window_matched") & event_count.gt(0)
    risk_type = event_types.str.contains("correction|inquiry|audit|financial_correction|financial_inquiry", case=False, regex=True)
    out["financial_event_matched"] = event_matched
    out["financial_quality_low_risk"] = event_matched & materiality.ge(0.65) & disclosure.ge(0.80) & risk.le(0.25) & missing.le(0.20)
    out["financial_positive_surprise_low_risk"] = event_matched & surprise.gt(0.15) & risk.le(0.25) & disclosure.ge(0.80) & missing.le(0.20)
    out["financial_multi_event_review"] = event_matched & event_count.ge(2) & missing.le(0.35)
    out["financial_high_risk_guard"] = event_matched & materiality.ge(0.60) & (
        risk.ge(0.45) | surprise.le(-0.35) | risk_type
    )
    return out


def num(frame: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def evaluate_rules(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (window, scope), scope_frame in detail.groupby(["window_days", "scope"], sort=True):
        scope_frame = scope_frame.copy()
        scope_frame["date_pool_return_20d"] = scope_frame.groupby("date")["return_20d"].transform("mean")
        baseline = baseline_by_block(scope_frame)
        for rule in RULES:
            selected = select_rule_rows(scope_frame, rule)
            for block in VALID_BLOCKS:
                block_frame = scope_frame[scope_frame["time_block"].eq(block)].copy()
                block_selected = selected[selected["time_block"].eq(block)].copy()
                rows.append(evaluate_selection(int(window), str(scope), rule, block_frame, block_selected, baseline.get(block, {})))
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


def select_rule_rows(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for flag in rule.get("required_flags", []):
        mask &= frame.get(flag, pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return frame[mask].copy()


def evaluate_selection(
    window: int,
    scope: str,
    rule: dict[str, Any],
    block_frame: pd.DataFrame,
    selected: pd.DataFrame,
    baseline: dict[str, float],
) -> dict[str, Any]:
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    excess = returns - pd.to_numeric(selected.get("date_pool_return_20d"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    concentration = selected["code"].astype(str).value_counts(normalize=True).max() if not selected.empty else np.nan
    selected_pos = float((returns > 0).mean()) if len(returns) else np.nan
    selected_loss = float((returns <= -5).mean()) if len(returns) else np.nan
    base_pos = baseline.get("baseline_positive_20d_rate", np.nan)
    base_loss = baseline.get("baseline_loss_gt5_rate", np.nan)
    base_avg = baseline.get("baseline_avg_return_20d", np.nan)
    selected_avg = float(returns.mean()) if len(returns) else np.nan
    return {
        "window_days": int(window),
        "scope": scope,
        "rule_id": rule["rule_id"],
        "direction": rule["direction"],
        "description": rule["description"],
        "valid_block": str(block_frame["time_block"].iloc[0]) if not block_frame.empty else "",
        "candidate_rows": int(len(block_frame)),
        "selected_rows": int(len(selected)),
        "selected_rate": round(float(len(selected) / max(1, len(block_frame))), 6),
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "coverage_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "top_stock_concentration": round(float(concentration), 6) if not pd.isna(concentration) else np.nan,
        "baseline_positive_20d_rate": _round(base_pos),
        "selected_positive_20d_rate": _round(selected_pos),
        "positive_rate_lift": _round(selected_pos - base_pos) if not pd.isna(selected_pos) and not pd.isna(base_pos) else np.nan,
        "baseline_loss_gt5_rate": _round(base_loss),
        "selected_loss_gt5_rate": _round(selected_loss),
        "loss_gt5_lift": _round(selected_loss - base_loss) if not pd.isna(selected_loss) and not pd.isna(base_loss) else np.nan,
        "baseline_avg_return_20d": _round(base_avg),
        "selected_avg_return_20d": _round(selected_avg),
        "selected_pool_excess_20d": _round(float(excess.mean())) if len(excess) else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["window_days", "scope", "rule_id"], sort=True):
        window, scope, rule_id = keys
        direction = str(group["direction"].iloc[0])
        prior = group[~group["valid_block"].eq("H2026_1")]
        h2026 = group[group["valid_block"].eq("H2026_1")]
        row = {
            "window_days": int(window),
            "scope": scope,
            "rule_id": rule_id,
            "direction": direction,
            "description": str(group["description"].iloc[0]),
            "total_selected_rows": int(group["selected_rows"].sum()),
            "h2026_selected_rows": int(h2026["selected_rows"].sum()) if not h2026.empty else 0,
            "prior_positive_rate_lift": _mean(prior, "positive_rate_lift"),
            "h2026_positive_rate_lift": _mean(h2026, "positive_rate_lift"),
            "prior_loss_gt5_lift": _mean(prior, "loss_gt5_lift"),
            "h2026_loss_gt5_lift": _mean(h2026, "loss_gt5_lift"),
            "prior_pool_excess_20d": _mean(prior, "selected_pool_excess_20d"),
            "h2026_pool_excess_20d": _mean(h2026, "selected_pool_excess_20d"),
            "prior_avg_return_20d": _mean(prior, "selected_avg_return_20d"),
            "h2026_avg_return_20d": _mean(h2026, "selected_avg_return_20d"),
            "max_top_stock_concentration": _max(group, "top_stock_concentration"),
            "min_coverage_dates": _min(group[group["selected_rows"].gt(0)], "coverage_dates"),
            "policy_status": "",
            "research_only": True,
            "not_investment_instruction": True,
        }
        row["policy_status"] = policy_status(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["policy_status", "scope", "window_days", "rule_id"], ascending=[True, True, True, True]
    )


def policy_status(row: dict[str, Any]) -> str:
    if int(row.get("total_selected_rows") or 0) < MIN_SELECTED_ROWS or int(row.get("h2026_selected_rows") or 0) < MIN_H2026_ROWS:
        return "reject_too_few_samples"
    concentration = row.get("max_top_stock_concentration")
    if concentration is not None and not pd.isna(concentration) and float(concentration) > MAX_TOP_STOCK_CONCENTRATION:
        return "reject_concentrated"
    direction = str(row.get("direction") or "")
    prior_excess = _num(row.get("prior_pool_excess_20d"))
    h_excess = _num(row.get("h2026_pool_excess_20d"))
    prior_pos = _num(row.get("prior_positive_rate_lift"))
    h_pos = _num(row.get("h2026_positive_rate_lift"))
    prior_loss = _num(row.get("prior_loss_gt5_lift"))
    h_loss = _num(row.get("h2026_loss_gt5_lift"))
    if direction == "positive" and prior_excess > 0 and h_excess > 0 and prior_pos >= 0.03 and h_pos >= 0.03 and prior_loss <= 0 and h_loss <= 0:
        return "accepted_positive_candidate_needs_ds_panel"
    if direction == "risk" and prior_excess < 0 and h_excess < 0 and prior_loss > 0 and h_loss > 0:
        return "accepted_risk_review_candidate"
    if direction == "positive" and prior_excess > 0 and h_excess > 0:
        return "observe_positive_thin_or_rate_weak"
    if direction == "risk" and (prior_excess < 0 or h_excess < 0):
        return "observe_risk_review_candidate"
    return "rejected_or_diagnostic_only"


def coverage_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    flags = [
        "financial_event_matched",
        "financial_quality_low_risk",
        "financial_positive_surprise_low_risk",
        "financial_multi_event_review",
        "financial_high_risk_guard",
    ]
    for (window, scope, block), group in detail.groupby(["window_days", "scope", "time_block"], sort=True):
        for flag in flags:
            values = group.get(flag, pd.Series(False, index=group.index)).fillna(False).astype(bool)
            rows.append(
                {
                    "window_days": int(window),
                    "scope": scope,
                    "time_block": block,
                    "signal": flag,
                    "coverage_rate": round(float(values.mean()), 6) if len(values) else 0.0,
                    "rows": int(values.sum()),
                    "unique_stocks": int(group.loc[values, "code"].nunique()) if values.any() else 0,
                    "coverage_dates": int(group.loc[values, "date"].nunique()) if values.any() else 0,
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return pd.DataFrame(rows)


def build_agent_previews(aggregate: pd.DataFrame, report_ref: str | None = None) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    if aggregate.empty:
        return previews
    report_ref = report_ref or "reports/date_generalization/financial_asof_window_expansion_v1.md"
    rules = {rule["rule_id"]: rule for rule in RULES}
    for _, row in aggregate.iterrows():
        rule_id = str(row["rule_id"])
        rule = rules[rule_id]
        preview = {
            "tool_id": f"financial_asof_window:{row['scope']}:{int(row['window_days'])}d:{rule_id}",
            "tool_version": "financial_asof_window_expansion_v1",
            "source_type": "paid_standardized",
            "source_name": "tushare_pro_local_cache",
            "scope": str(row["scope"]),
            "window_days": int(row["window_days"]),
            "rule_id": rule_id,
            "direction": rule["direction"],
            "policy_status": str(row["policy_status"]),
            "usable_in_agent_default": str(row["policy_status"]) in {
                "accepted_positive_candidate_needs_ds_panel",
                "accepted_risk_review_candidate",
            },
            "rule_conditions": rule.get("required_flags", []),
            "required_safety": [
                "available_at_must_be_on_or_before_decision_time",
                "report_period_and_disclosure_date_required",
                "not_standalone_investment_instruction",
                "must_be_cross_checked_with_news_peer_bookskill_kline_and_chip",
            ],
            "action_hint": action_hint(str(row["policy_status"]), rule["direction"]),
            "description": rule["description"],
            "source_ref_ids": [
                "data/date_generalization_cache/market_5000/financial_report_features.csv",
                "data/date_generalization_cache/market_5000/financial_report_events.csv",
                report_ref,
            ],
            "research_only": True,
            "not_investment_instruction": True,
        }
        assert_no_future_fields(preview)
        previews.append(preview)
    return previews


def action_hint(status: str, direction: str) -> str:
    if status == "accepted_positive_candidate_needs_ds_panel":
        return "may_enter_small_ds_panel_as_one_positive_confirmation_not_standalone"
    if status == "accepted_risk_review_candidate":
        return "may_enter_agent_as_risk_review_or_downweight_checklist"
    if status.startswith("observe") and direction == "risk":
        return "observe_only_risk_review_candidate_needs_more_panel_evidence"
    if status.startswith("observe"):
        return "observe_only_not_default_promotion"
    return "do_not_use_for_upgrade"


def write_outputs(
    *,
    prefix: str,
    detail: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    coverage: pd.DataFrame,
    previews: list[dict[str, Any]],
) -> dict[str, Path]:
    aggregate_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    metrics_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    coverage_path = REPORT_DIR / f"{prefix}_coverage.csv"
    preview_path = REPORT_DIR / f"{prefix}_agent_preview.jsonl"
    detail_path = REPORT_DIR / f"{prefix}_safe_detail.csv.gz"
    report_path = REPORT_DIR / f"{prefix}.md"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    safe_cols = [
        "window_days",
        "scope",
        "date",
        "code",
        "name",
        "time_block",
        "rev_chip_score_quantile",
        "financial_report_join_status",
        "financial_report_event_count",
        "financial_report_materiality_score",
        "financial_quality_risk_score",
        "financial_surprise_score",
        "financial_disclosure_quality_score",
        "financial_report_missing_rate",
        "financial_report_latest_period",
        "financial_report_event_types",
        "financial_report_available_at",
        "financial_event_matched",
        "financial_quality_low_risk",
        "financial_positive_surprise_low_risk",
        "financial_multi_event_review",
        "financial_high_risk_guard",
    ]
    detail[[col for col in safe_cols if col in detail]].to_csv(detail_path, index=False, encoding="utf-8-sig", compression="gzip")
    with preview_path.open("w", encoding="utf-8") as handle:
        for item in previews:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(report_path, aggregate, coverage, metrics, detail_path, preview_path)
    return {
        "aggregate": aggregate_path,
        "metrics": metrics_path,
        "coverage": coverage_path,
        "safe_detail": detail_path,
        "agent_preview": preview_path,
        "report": report_path,
    }


def write_report(path: Path, aggregate: pd.DataFrame, coverage: pd.DataFrame, metrics: pd.DataFrame, detail_path: Path, preview_path: Path) -> None:
    lines = [
        "# Financial As-Of Window Expansion Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "用现有本地财报/公告特征测试 30/60/90/180/365 天 as-of 窗口是否能缓解覆盖稀疏与股票集中问题。未来 20 日收益只用于本离线报告；Agent preview 不含未来收益、GT 或同池超额字段。",
        "",
        "## Outputs",
        "",
        f"- safe_detail: `{detail_path.relative_to(ROOT)}`",
        f"- agent_preview: `{preview_path.relative_to(ROOT)}`",
        "",
        "## Aggregate",
        "",
        table(aggregate),
        "",
        "## Coverage",
        "",
        table(coverage.head(120)),
        "",
        "## Step Metrics",
        "",
        table(metrics.head(160)),
        "",
        "## Interpretation",
        "",
        "- `accepted_positive_candidate_needs_ds_panel` 只表示可进入小 DS 面板作为一条正向确认，不是默认 alpha。",
        "- `accepted_risk_review_candidate` 只能作为风险复核/降权检查清单，不是投资指令。",
        "- 若长窗口只是增加旧财报覆盖但 prior/H2026 不一致，说明必须补真实公告/财报数据源，而不是放宽窗口。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_FIELDS:
                raise ValueError(f"future/result field leaked to preview: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def _round(value: float) -> float:
    return round(float(value), 6) if value is not None and not pd.isna(value) else np.nan


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


def _min(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce")
    return _round(float(values.min())) if values.notna().any() else np.nan


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
