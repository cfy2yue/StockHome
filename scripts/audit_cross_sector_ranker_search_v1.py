"""Local cross-sector ranker search for P1 candidate comparison.

This script is deliberately zero-DeepSeek-token. It uses future 20d returns
only for offline evaluation, never for agent evidence. The goal is to decide
whether cross-sector candidate comparison has a better deterministic or
walk-forward ML anchor than the current rank_avg_rev_watch baseline.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_comparison_workflow_v1 import SAFE_AGENT_FEATURES, SCORE_COLUMNS  # noqa: E402

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_INPUT = REPORT_DIR / "candidate_comparison_stability_v1_candidate_rows_eval.csv"
DEFAULT_PREFIX = "cross_sector_ranker_search_v1"
TIME_BLOCK_ORDER = ["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]
BASELINE_SCORE = "rank_avg_rev_watch"

BASE_SCORE_COLUMNS = [
    "rank_avg_rev_watch",
    "rev_chip_core",
    "single_watch_proxy",
    "candidate_context_blend_v1",
    "original_total_score",
    "p1_default_selector_v1",
]

SEARCH_SCORE_COLUMNS = [
    *BASE_SCORE_COLUMNS,
    "cross_low_risk_anchor_v1",
    "cross_peer_confirmed_anchor_v1",
    "cross_news_financial_confirmed_v1",
    "cross_balanced_quality_anchor_v1",
    "cross_h2026_defensive_anchor_v1",
    "ml_ridge_walkforward_v1",
    "ml_hgbr_walkforward_v1",
    "cross_ml_ridge_rankavg_ensemble_v1",
    "cross_ml_hgbr_rankavg_ensemble_v1",
    "cross_ml_dual_rankavg_ensemble_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-sector P1 ranker candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default="every_2_weeks,weekly_friday,weekly_tuesday")
    parser.add_argument("--min-train-rows", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.input)
    frequencies = {item.strip() for item in args.frequencies.split(",") if item.strip()}
    if frequencies:
        frame = frame[frame["decision_frequency"].astype(str).isin(frequencies)].copy()
    cross = frame[frame["comparison_scenario"].astype(str).eq("cross_sector")].copy()
    cross = add_formula_scores(cross)
    cross = add_walkforward_ml_scores(cross, min_train_rows=args.min_train_rows)
    cross = add_ml_ensemble_scores(cross)

    detail = evaluate_scores(cross, SEARCH_SCORE_COLUMNS)
    aggregate = aggregate_metrics(detail)
    paired = paired_vs_baseline(detail, baseline=BASELINE_SCORE)
    gate = build_gate_table(aggregate, paired)
    panel = panel_metrics(detail)
    preview = build_agent_preview(cross, gate)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "detail": REPORT_DIR / f"{prefix}_detail.csv",
        "aggregate": REPORT_DIR / f"{prefix}_aggregate.csv",
        "paired": REPORT_DIR / f"{prefix}_paired_vs_baseline.csv",
        "gate": REPORT_DIR / f"{prefix}_gate.csv",
        "panel": REPORT_DIR / f"{prefix}_panel_metrics.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.csv",
        "candidate_rows_no_gt": REPORT_DIR / f"{prefix}_candidate_rows_no_gt_with_scores.csv",
        "candidate_rows_no_gt_every2": REPORT_DIR / f"{prefix}_candidate_rows_no_gt_every_2_weeks_with_scores.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    aggregate.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    paired.to_csv(paths["paired"], index=False, encoding="utf-8-sig")
    gate.to_csv(paths["gate"], index=False, encoding="utf-8-sig")
    panel.to_csv(paths["panel"], index=False, encoding="utf-8-sig")
    preview.to_csv(paths["preview"], index=False, encoding="utf-8-sig")
    candidate_no_gt = build_candidate_rows_no_gt(cross)
    candidate_no_gt.to_csv(paths["candidate_rows_no_gt"], index=False, encoding="utf-8-sig")
    candidate_no_gt[candidate_no_gt["decision_frequency"].astype(str).eq("every_2_weeks")].to_csv(
        paths["candidate_rows_no_gt_every2"], index=False, encoding="utf-8-sig"
    )
    paths["report"].write_text(render_report(cross, aggregate, paired, gate, panel, paths, report_name=prefix), encoding="utf-8")
    print(f"rows={len(cross)} groups={cross['comparison_group_id'].nunique()} scores={len(SEARCH_SCORE_COLUMNS)}")
    print(f"wrote: {paths['report']}")


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    required = {"comparison_group_id", "comparison_scenario", "decision_frequency", "time_block", "date", "code", "return_20d"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = frame.copy()
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame = frame[frame["time_block"].astype(str).isin(TIME_BLOCK_ORDER) & frame["return_20d"].notna()].copy()
    return frame


def add_formula_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Ensure rank_avg exists even when an upstream artifact predates it.
    if "rank_avg_rev_watch" not in out.columns or not pd.to_numeric(out.get("rank_avg_rev_watch"), errors="coerce").notna().any():
        out["rank_avg_rev_watch"] = group_z(out, "rev_chip_core") + group_z(out, "single_watch_proxy")

    risk = (
        0.22 * group_z(out, "news_warning_score").clip(lower=0)
        + 0.20 * group_z(out, "financial_quality_risk_score").clip(lower=0)
        + 0.08 * group_z(out, "news_missing_rate").clip(lower=0)
        + 0.08 * group_z(out, "financial_report_missing_rate").clip(lower=0)
        + 0.06 * group_z(out, "kline_volatility_ratio_3_20").clip(lower=0)
    )
    peer = (
        0.22 * group_z(out, "corr_peer_relative_return_20d")
        + 0.16 * group_z(out, "tushare_industry_relative_return_20d")
        + 0.10 * group_z(out, "tushare_industry_positive_breadth_20d")
        + 0.06 * group_z(out, "tushare_area_relative_return_20d")
    )
    event_quality = (
        0.16 * group_z(out, "news_opportunity_score")
        + 0.10 * group_z(out, "policy_background_score")
        + 0.10 * group_z(out, "official_confirmation_score")
        + 0.08 * group_z(out, "announcement_materiality_score")
        + 0.10 * group_z(out, "financial_surprise_score")
        + 0.08 * group_z(out, "financial_disclosure_quality_score")
    )
    price = 0.55 * group_z(out, "rev_chip_core") + 0.35 * group_z(out, "single_watch_proxy")

    out["cross_low_risk_anchor_v1"] = out["rank_avg_rev_watch"] - risk
    out["cross_peer_confirmed_anchor_v1"] = 0.75 * out["rank_avg_rev_watch"] + peer - 0.35 * risk
    out["cross_news_financial_confirmed_v1"] = 0.72 * out["rank_avg_rev_watch"] + event_quality - 0.40 * risk
    out["cross_balanced_quality_anchor_v1"] = 0.62 * out["rank_avg_rev_watch"] + 0.45 * peer + 0.35 * event_quality - 0.55 * risk
    out["cross_h2026_defensive_anchor_v1"] = price + 0.35 * peer - 0.75 * risk
    return out


def add_walkforward_ml_scores(frame: pd.DataFrame, *, min_train_rows: int) -> pd.DataFrame:
    out = frame.copy()
    for col in ["ml_ridge_walkforward_v1", "ml_hgbr_walkforward_v1"]:
        out[col] = np.nan
    features = ml_feature_columns(out)
    if not features:
        return out
    out["_group_excess_label"] = out["return_20d"] - out.groupby("comparison_group_id")["return_20d"].transform("mean")
    for frequency, freq_frame in out.groupby("decision_frequency", sort=True):
        freq_index = freq_frame.index
        for block in TIME_BLOCK_ORDER:
            train_blocks = TIME_BLOCK_ORDER[: TIME_BLOCK_ORDER.index(block)]
            if not train_blocks:
                continue
            train_idx = freq_frame.index[freq_frame["time_block"].astype(str).isin(train_blocks)]
            valid_idx = freq_frame.index[freq_frame["time_block"].astype(str).eq(block)]
            train = out.loc[train_idx]
            valid = out.loc[valid_idx]
            train = train[train["_group_excess_label"].notna()].copy()
            if len(train) < min_train_rows or valid.empty:
                continue
            x_train = train[features]
            y_train = train["_group_excess_label"].clip(lower=-30, upper=30)
            x_valid = valid[features]

            ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=8.0))
            ridge.fit(x_train, y_train)
            out.loc[valid_idx, "ml_ridge_walkforward_v1"] = ridge.predict(x_valid)

            hgbr = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=90,
                    learning_rate=0.045,
                    max_leaf_nodes=8,
                    l2_regularization=1.0,
                    min_samples_leaf=60,
                    random_state=17,
                ),
            )
            hgbr.fit(x_train, y_train)
            out.loc[valid_idx, "ml_hgbr_walkforward_v1"] = hgbr.predict(x_valid)
        # Suppress an unused-group linter complaint while keeping the grouping explicit.
        _ = frequency, freq_index
    return out.drop(columns=["_group_excess_label"], errors="ignore")


def add_ml_ensemble_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rank = group_z(out, "rank_avg_rev_watch")
    ridge = group_z(out, "ml_ridge_walkforward_v1")
    hgbr = group_z(out, "ml_hgbr_walkforward_v1")
    out["cross_ml_ridge_rankavg_ensemble_v1"] = 0.55 * ridge + 0.45 * rank
    out["cross_ml_hgbr_rankavg_ensemble_v1"] = 0.55 * hgbr + 0.45 * rank
    out["cross_ml_dual_rankavg_ensemble_v1"] = 0.38 * ridge + 0.34 * hgbr + 0.28 * rank
    return out


def ml_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [col for col in [*BASE_SCORE_COLUMNS, *SAFE_AGENT_FEATURES] if col in frame.columns]
    numeric = []
    for col in candidates:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() >= 100 and values.nunique(dropna=True) > 3:
            frame[col] = values
            numeric.append(col)
    return sorted(set(numeric))


def group_z(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_numeric(frame.get(field, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if math.isnan(std) or std <= 0 or len(group) < 2:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["comparison_group_id"].astype(str), sort=False).transform(_z).fillna(0.0)


def evaluate_scores(frame: pd.DataFrame, score_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_name in score_names:
        if score_name not in frame.columns:
            continue
        for group_id, group in frame.groupby("comparison_group_id", sort=True):
            scores = pd.to_numeric(group[score_name], errors="coerce")
            returns = pd.to_numeric(group["return_20d"], errors="coerce")
            eval_frame = group.assign(_score=scores, _ret=returns).dropna(subset=["_score", "_ret"])
            if len(eval_frame) < 3:
                continue
            ranked = eval_frame.sort_values(["_score", "code"], ascending=[False, True])
            top1 = ranked.iloc[0]
            top2 = ranked.head(min(2, len(ranked)))
            bottom2 = ranked.tail(min(2, len(ranked)))
            group_mean = float(eval_frame["_ret"].mean())
            top1_ret = float(top1["_ret"])
            top2_ret = float(top2["_ret"].mean())
            rank_ic = ranked["_score"].rank().corr(ranked["_ret"].rank(), method="pearson") if ranked["_score"].nunique() > 1 else np.nan
            rows.append(
                {
                    "comparison_group_id": group_id,
                    "decision_frequency": str(group["decision_frequency"].iloc[0]),
                    "time_block": str(group["time_block"].iloc[0]),
                    "sample_panel_id": str(group.get("sample_panel_id", pd.Series(["NA"])).iloc[0]),
                    "date": str(group["date"].iloc[0]),
                    "score_name": score_name,
                    "candidate_count": int(len(eval_frame)),
                    "rank_ic": float(rank_ic) if not pd.isna(rank_ic) else np.nan,
                    "top1_code": str(top1["code"]).zfill(6),
                    "top1_return_20d": top1_ret,
                    "top2_mean_return_20d": top2_ret,
                    "group_mean_return_20d": group_mean,
                    "top1_excess_20d": top1_ret - group_mean,
                    "top2_excess_20d": top2_ret - group_mean,
                    "top1_positive": bool(top1_ret > 0),
                    "top2_positive_rate": float((top2["_ret"] > 0).mean()),
                    "top1_is_best": bool(top1_ret >= float(eval_frame["_ret"].max())),
                    "top1_is_worst": bool(top1_ret <= float(eval_frame["_ret"].min())),
                    "top1_regret_vs_best": float(eval_frame["_ret"].max() - top1_ret),
                    "bottom2_loss_gt5_rate": float((bottom2["_ret"] <= -5).mean()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    rows: list[dict[str, Any]] = []
    for keys, group in detail.groupby(["decision_frequency", "score_name", "time_block"], sort=True):
        rows.append(aggregate_row(keys, group))
    for keys, group in detail.groupby(["decision_frequency", "score_name"], sort=True):
        rows.append(aggregate_row((keys[0], keys[1], "ALL"), group))
    return pd.DataFrame(rows).sort_values(["decision_frequency", "time_block", "score_name"]).reset_index(drop=True)


def aggregate_row(keys: tuple[Any, ...], group: pd.DataFrame) -> dict[str, Any]:
    frequency, score_name, block = keys
    rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce")
    return {
        "decision_frequency": frequency,
        "score_name": score_name,
        "time_block": block,
        "n_groups": int(group["comparison_group_id"].nunique()),
        "mean_rank_ic": round(float(rank_ic.mean()), 6) if rank_ic.notna().any() else np.nan,
        "rank_ic_positive_rate": round(float((rank_ic.dropna() > 0).mean()), 6) if rank_ic.notna().any() else np.nan,
        "top1_excess_mean": round(float(pd.to_numeric(group["top1_excess_20d"], errors="coerce").mean()), 6),
        "top2_excess_mean": round(float(pd.to_numeric(group["top2_excess_20d"], errors="coerce").mean()), 6),
        "top1_positive_rate": round(float(group["top1_positive"].astype(bool).mean()), 6),
        "top2_positive_rate": round(float(pd.to_numeric(group["top2_positive_rate"], errors="coerce").mean()), 6),
        "top1_best_rate": round(float(group["top1_is_best"].astype(bool).mean()), 6),
        "top1_worst_rate": round(float(group["top1_is_worst"].astype(bool).mean()), 6),
        "regret_mean": round(float(pd.to_numeric(group["top1_regret_vs_best"], errors="coerce").mean()), 6),
        "bottom2_loss_gt5_rate": round(float(pd.to_numeric(group["bottom2_loss_gt5_rate"], errors="coerce").mean()), 6),
    }


def paired_vs_baseline(detail: pd.DataFrame, *, baseline: str) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    base = detail[detail["score_name"].eq(baseline)].copy()
    base = base[
        [
            "comparison_group_id",
            "decision_frequency",
            "time_block",
            "top1_excess_20d",
            "top2_excess_20d",
            "top1_positive",
            "top1_is_worst",
            "top1_regret_vs_best",
        ]
    ].rename(
        columns={
            "top1_excess_20d": "base_top1_excess",
            "top2_excess_20d": "base_top2_excess",
            "top1_positive": "base_top1_positive",
            "top1_is_worst": "base_top1_is_worst",
            "top1_regret_vs_best": "base_regret",
        }
    )
    rows: list[dict[str, Any]] = []
    for score_name, subset in detail[~detail["score_name"].eq(baseline)].groupby("score_name", sort=True):
        merged = subset.merge(base, on=["comparison_group_id", "decision_frequency", "time_block"], how="inner")
        if merged.empty:
            continue
        for keys, group in merged.groupby(["decision_frequency", "time_block"], sort=True):
            rows.append(paired_row(keys, score_name, group))
        for frequency, group in merged.groupby("decision_frequency", sort=True):
            rows.append(paired_row((frequency, "ALL"), score_name, group))
    return pd.DataFrame(rows).sort_values(["decision_frequency", "time_block", "score_name"]).reset_index(drop=True)


def paired_row(keys: tuple[Any, ...], score_name: str, group: pd.DataFrame) -> dict[str, Any]:
    frequency, block = keys
    return {
        "decision_frequency": frequency,
        "score_name": score_name,
        "time_block": block,
        "n_groups": int(group["comparison_group_id"].nunique()),
        "delta_top1_excess_mean": round(float((group["top1_excess_20d"] - group["base_top1_excess"]).mean()), 6),
        "delta_top2_excess_mean": round(float((group["top2_excess_20d"] - group["base_top2_excess"]).mean()), 6),
        "delta_top1_positive_rate": round(float(group["top1_positive"].astype(float).mean() - group["base_top1_positive"].astype(float).mean()), 6),
        "delta_top1_worst_rate": round(float(group["top1_is_worst"].astype(float).mean() - group["base_top1_is_worst"].astype(float).mean()), 6),
        "delta_regret_mean": round(float((group["top1_regret_vs_best"] - group["base_regret"]).mean()), 6),
        "beats_baseline_top1_rate": round(float((group["top1_excess_20d"] > group["base_top1_excess"]).mean()), 6),
        "beats_baseline_top2_rate": round(float((group["top2_excess_20d"] > group["base_top2_excess"]).mean()), 6),
    }


def panel_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    rows = []
    for keys, group in detail.groupby(["decision_frequency", "score_name", "sample_panel_id"], sort=True):
        rows.append(aggregate_row((keys[0], keys[1], keys[2]), group))
    return pd.DataFrame(rows).rename(columns={"time_block": "sample_panel_id"}).reset_index(drop=True)


def build_gate_table(aggregate: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    all_rows = aggregate[aggregate["time_block"].eq("ALL")].copy()
    h2026 = aggregate[aggregate["time_block"].eq("H2026_1")].copy()
    h2026 = h2026.add_prefix("h2026_")
    out = all_rows.merge(
        h2026,
        left_on=["decision_frequency", "score_name"],
        right_on=["h2026_decision_frequency", "h2026_score_name"],
        how="left",
    )
    paired_all = paired[paired["time_block"].eq("ALL")][
        ["decision_frequency", "score_name", "delta_top1_excess_mean", "delta_top2_excess_mean", "beats_baseline_top2_rate"]
    ]
    out = out.merge(paired_all, on=["decision_frequency", "score_name"], how="left")
    out["candidate_for_cross_sector_anchor"] = (
        (pd.to_numeric(out["mean_rank_ic"], errors="coerce") >= 0.055)
        & (pd.to_numeric(out["rank_ic_positive_rate"], errors="coerce") >= 0.54)
        & (pd.to_numeric(out["top2_excess_mean"], errors="coerce") >= 0.45)
        & (pd.to_numeric(out["h2026_top2_excess_mean"], errors="coerce") >= 0.0)
        & (pd.to_numeric(out["top1_worst_rate"], errors="coerce") <= 0.13)
    )
    out["candidate_note"] = out.apply(gate_note, axis=1)
    return out.sort_values(["decision_frequency", "candidate_for_cross_sector_anchor", "top2_excess_mean"], ascending=[True, False, False]).reset_index(drop=True)


def gate_note(row: pd.Series) -> str:
    notes = []
    checks = [
        ("rank_ic_weak", row.get("mean_rank_ic"), 0.055, "lt"),
        ("rank_ic_hit_weak", row.get("rank_ic_positive_rate"), 0.54, "lt"),
        ("top2_excess_weak", row.get("top2_excess_mean"), 0.45, "lt"),
        ("h2026_top2_negative", row.get("h2026_top2_excess_mean"), 0.0, "lt"),
        ("top1_worst_high", row.get("top1_worst_rate"), 0.13, "gt"),
    ]
    for label, value, threshold, mode in checks:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            notes.append(label)
        elif mode == "lt" and value < threshold:
            notes.append(label)
        elif mode == "gt" and value > threshold:
            notes.append(label)
    return "pass_candidate" if not notes else ";".join(notes)


def build_agent_preview(frame: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    accepted = gate[gate["candidate_for_cross_sector_anchor"].astype(bool)].copy()
    if accepted.empty:
        accepted = gate.sort_values(["decision_frequency", "top2_excess_mean"], ascending=[True, False]).groupby("decision_frequency").head(1)
    score_names = accepted["score_name"].dropna().astype(str).unique().tolist()[:4]
    if not score_names:
        score_names = [BASELINE_SCORE]
    keep_cols = [
        "comparison_group_id",
        "decision_frequency",
        "time_block",
        "date",
        "code",
        "name",
        "tushare_industry",
        "tushare_area",
        *score_names,
        "news_warning_score",
        "financial_quality_risk_score",
        "corr_peer_relative_return_20d",
        "tushare_industry_relative_return_20d",
        "triggered_skills",
        "data_gaps",
    ]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    preview = frame[frame["time_block"].isin(["H2025_2", "H2026_1"])][keep_cols].head(120).copy()
    forbidden = {"return_20d", "future_return_20d", "fwd_ret_20d", "gt_status", "gt_pass", "pool_excess_20d"}
    return preview.drop(columns=[col for col in forbidden if col in preview.columns], errors="ignore")


def build_candidate_rows_no_gt(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = {
        "return_20d",
        "future_return_20d",
        "fwd_ret_20d",
        "pool_excess_20d",
        "rank_pct_in_date",
        "rank_pct_in_industry_date",
        "gt_status",
        "gt_pass",
        "rule_outcome_label",
    }
    out = frame.drop(columns=[col for col in forbidden if col in frame.columns], errors="ignore").copy()
    out["source_comparison_group_id"] = out["comparison_group_id"].astype(str)
    out["comparison_group_id"] = out["decision_frequency"].astype(str) + "__" + out["source_comparison_group_id"].astype(str)
    # Keep only columns the candidate-comparison runner can safely serialize.
    cols = [
        "comparison_group_id",
        "source_comparison_group_id",
        "comparison_scenario",
        "repeat_seed",
        "time_block",
        "date",
        "candidate_count",
        "candidate_codes",
        "candidate_names",
        "industry_context",
        "code",
        "name",
        "tushare_industry",
        "tushare_area",
        *sorted(set([*SCORE_COLUMNS, *SEARCH_SCORE_COLUMNS])),
        *[col for col in SAFE_AGENT_FEATURES if col in out.columns],
        "decision_frequency",
        "sample_panel_id",
    ]
    cols = [col for col in cols if col in out.columns]
    return out[cols]


def render_report(
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    gate: pd.DataFrame,
    panel: pd.DataFrame,
    paths: dict[str, Path],
    report_name: str,
) -> str:
    primary_gate = gate[gate["decision_frequency"].eq("every_2_weeks")].head(12)
    h2026 = aggregate[(aggregate["decision_frequency"].eq("every_2_weeks")) & (aggregate["time_block"].eq("H2026_1"))]
    h2026 = h2026.sort_values("top2_excess_mean", ascending=False).head(12)
    paired_all = paired[(paired["decision_frequency"].eq("every_2_weeks")) & (paired["time_block"].eq("ALL"))]
    paired_all = paired_all.sort_values("delta_top2_excess_mean", ascending=False).head(12)
    panel_every2 = panel[panel["decision_frequency"].eq("every_2_weeks")].copy()
    panel_every2 = panel_every2[
        panel_every2["score_name"].isin(
            [
                BASELINE_SCORE,
                "ml_hgbr_walkforward_v1",
                "ml_ridge_walkforward_v1",
                "cross_ml_ridge_rankavg_ensemble_v1",
                "cross_ml_hgbr_rankavg_ensemble_v1",
                "cross_ml_dual_rankavg_ensemble_v1",
                "cross_h2026_defensive_anchor_v1",
            ]
        )
    ]
    lines = [
        f"# Cross-Sector Ranker Search ({report_name})",
        "",
        "本审计只做本地离线评估，不调用 DeepSeek。未来 20 日收益只用于指标复盘，不进入 Agent evidence。",
        "",
        "## Setup",
        "",
        f"- rows: `{len(frame)}`",
        f"- groups: `{frame['comparison_group_id'].nunique()}`",
        f"- frequencies: `{', '.join(sorted(frame['decision_frequency'].astype(str).unique()))}`",
        "- label: group-relative `return_20d` only for offline walk-forward training/evaluation",
        "- baseline: `rank_avg_rev_watch`",
        "",
        "## Gate Summary: every_2_weeks",
        "",
        markdown_table(primary_gate),
        "",
        "## H2026_1 Top Scores: every_2_weeks",
        "",
        markdown_table(h2026),
        "",
        "## Paired Delta vs rank_avg_rev_watch: every_2_weeks ALL",
        "",
        markdown_table(paired_all),
        "",
        "## Panel Stability Snapshot",
        "",
        markdown_table(panel_every2.head(40)),
        "",
        "## Interpretation",
        "",
        "- `candidate_for_cross_sector_anchor=True` 只表示值得进入下一轮小面板/DS 验证，不表示可直接升默认。",
        "- 若 ML 分数只在 ALL 好、H2026_1 崩，视为时间过拟合，不采纳。",
        "- 若确定性公式优于 baseline 但 panel std 大，先作为 Agent 的辅助排序解释，不直接替换默认锚点。",
        "- 若所有候选都未通过 gate，跨领域任务继续降级为辅助比较，并优先改成“先行业归一化/同领域内筛选，再跨领域风险确认”。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


def safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()
