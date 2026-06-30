"""Audit non-price confirmation overlays for the P0 small-entry branch.

The previous operation-policy audit found a narrow yellow branch:
`branch_stack_v1` rows with `small_buy_hold`, i.e. confirmed by opportunity and
K-line tools but not the strongest/chase-like opportunity quantile. This script
tests whether existing news, financial, peer, chip, and K-line risk fields can
make that branch cleaner before spending DeepSeek tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    FUTURE_OR_RESULT_FIELDS,
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    apply_policy,
    build_scored_target,
    load_stack_frame,
    safe_float,
    safe_prefix,
)
from scripts.audit_p0_operation_policy_v1 import with_operation_actions  # noqa: E402
from scripts.audit_single_stock_review_quality import _rolling_split  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_nonprice_overlay_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
DEFAULT_KLINE_GROUP = "kline_peer_chip_news_risk"
MAX_HGB_TRAIN_ROWS = 60000

OVERLAY_COLUMNS = [
    "news_missing_rate",
    "news_warning_score",
    "news_opportunity_score",
    "news_evidence_quality",
    "official_confirmation_score",
    "announcement_materiality_score",
    "financial_report_missing_rate",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_materiality_score",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "lower_support",
    "upper_overhang",
    "winner_rate_pct",
    "cost_band_width",
    "kline_rsi14",
    "kline_return_20d",
    "kline_return_60d",
    "kline_volatility_ratio_20_60",
    "channel_positive_support_prob",
    "channel_hard_counter_prob",
    "channel_soft_gap_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit non-price overlays on P0 small-entry branch.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--kline-feature-group", default=DEFAULT_KLINE_GROUP)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--preview-max-rows", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    frame, feature_groups, notes = load_stack_frame()
    available_overlays = [col for col in OVERLAY_COLUMNS if col in frame.columns]

    metric_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    hygiene_rows: list[dict[str, Any]] = []

    for frequency in frequencies:
        freq_frame = apply_frequency(frame, frequency)
        for target_block in TARGET_BLOCKS:
            train, validation, target = _rolling_split(freq_frame, target_block)
            if len(train) < MIN_TRAIN_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "target_block": target_block,
                        "status": "skip_insufficient_rows",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            scored = build_scored_target(
                train,
                validation,
                target,
                feature_groups,
                kline_feature_group=args.kline_feature_group,
                max_hgb_train_rows=args.max_hgb_train_rows,
            )
            if scored.empty:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "target_block": target_block,
                        "status": "skip_model_unavailable",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "target_rows": len(target),
                    }
                )
                continue
            enriched = enrich_with_overlay_columns(scored, target, available_overlays)
            branch = with_operation_actions(apply_policy(enriched, "branch_stack_v1"))
            small = branch[branch["operation_action"].astype(str).eq("small_buy_hold")].copy()
            if small.empty:
                continue
            for rule in overlay_rules():
                subset = small[rule.mask(small)].copy()
                metric_rows.append(evaluate_rule(small, subset, frequency, target_block, rule))
            if target_block == FINAL_OOT:
                preview_rows.extend(build_safe_preview(small, frequency, args.preview_max_rows))

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(args, notes, available_overlays, summary, metrics, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def enrich_with_overlay_columns(scored: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = ["date", "code", *columns]
    overlay = target[[col for col in keep if col in target.columns]].copy()
    overlay["code"] = overlay["code"].astype(str).str.zfill(6)
    out = scored.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.merge(overlay.drop_duplicates(["date", "code"]), on=["date", "code"], how="left")


class OverlayRule:
    def __init__(self, rule_id: str, description: str, mask: Callable[[pd.DataFrame], pd.Series]):
        self.rule_id = rule_id
        self.description = description
        self.mask = mask


def overlay_rules() -> list[OverlayRule]:
    return [
        OverlayRule("small_entry_all", "all branch_stack small-entry rows", lambda df: pd.Series(True, index=df.index)),
        OverlayRule("news_low_warning", "news warning is low or absent", lambda df: num(df, "news_warning_score").fillna(0) <= 0.2),
        OverlayRule(
            "news_available_low_warning",
            "news coverage exists and warning is low",
            lambda df: (num(df, "news_missing_rate").fillna(1) < 0.8) & (num(df, "news_warning_score").fillna(0) <= 0.2),
        ),
        OverlayRule(
            "news_opportunity_beats_warning",
            "news opportunity score exceeds warning score",
            lambda df: num(df, "news_opportunity_score").fillna(0) > num(df, "news_warning_score").fillna(0),
        ),
        OverlayRule(
            "financial_available_low_risk",
            "financial channel available and risk score low",
            lambda df: (num(df, "financial_report_missing_rate").fillna(1) < 1)
            & (num(df, "financial_quality_risk_score").fillna(0.5) <= 0.5),
        ),
        OverlayRule(
            "peer_relative_positive",
            "correlated peer relative return is positive",
            lambda df: num(df, "corr_peer_relative_return_20d").fillna(-999) > 0,
        ),
        OverlayRule(
            "industry_breadth_positive",
            "industry positive breadth is at least half",
            lambda df: num(df, "tushare_industry_positive_breadth_20d").fillna(0) >= 0.5,
        ),
        OverlayRule(
            "chip_support_v2",
            "lower support is present and upper overhang is limited",
            lambda df: (num(df, "lower_support").fillna(0) >= 0.18) & (num(df, "upper_overhang").fillna(1) <= 0.12),
        ),
        OverlayRule(
            "not_overheated_rsi",
            "RSI avoids extreme overheat",
            lambda df: num(df, "kline_rsi14").fillna(50).between(35, 75),
        ),
        OverlayRule(
            "channel_positive_support",
            "channel classifier has moderate positive support",
            lambda df: num(df, "channel_positive_support_prob").fillna(0) >= 0.25,
        ),
        OverlayRule(
            "nonprice_confirm_min2",
            "at least two non-price confirmations",
            lambda df: confirmation_count(df) >= 2,
        ),
        OverlayRule(
            "small_entry_clean_confirmed",
            "clean small-entry: non-price min2 plus RSI and no hard channel risk",
            lambda df: (confirmation_count(df) >= 2)
            & num(df, "kline_rsi14").fillna(50).between(35, 75)
            & (num(df, "channel_hard_counter_prob").fillna(0) < 0.8),
        ),
        OverlayRule(
            "small_entry_news_fin_peer",
            "news low warning plus financial low risk plus peer positive",
            lambda df: (num(df, "news_warning_score").fillna(0) <= 0.2)
            & (num(df, "financial_quality_risk_score").fillna(0.5) <= 0.5)
            & (num(df, "corr_peer_relative_return_20d").fillna(-999) > 0),
        ),
    ]


def confirmation_count(df: pd.DataFrame) -> pd.Series:
    confirmations = [
        num(df, "news_warning_score").fillna(0) <= 0.2,
        (num(df, "financial_report_missing_rate").fillna(1) < 1) & (num(df, "financial_quality_risk_score").fillna(0.5) <= 0.5),
        num(df, "corr_peer_relative_return_20d").fillna(-999) > 0,
        num(df, "tushare_industry_positive_breadth_20d").fillna(0) >= 0.5,
        (num(df, "lower_support").fillna(0) >= 0.18) & (num(df, "upper_overhang").fillna(1) <= 0.12),
        num(df, "official_confirmation_score").fillna(0) > 0,
    ]
    out = pd.Series(0, index=df.index)
    for item in confirmations:
        out = out + item.astype(int)
    return out


def evaluate_rule(base: pd.DataFrame, subset: pd.DataFrame, frequency: str, target_block: str, rule: OverlayRule) -> dict[str, Any]:
    base_ret = pd.to_numeric(base["return_20d"], errors="coerce")
    ret = pd.to_numeric(subset["return_20d"], errors="coerce")
    return {
        "frequency": frequency,
        "target_block": target_block,
        "rule_id": rule.rule_id,
        "description": rule.description,
        "base_rows": int(len(base)),
        "selected_rows": int(len(subset)),
        "selected_rate_vs_branch": round(float(len(subset) / max(1, len(base))), 6),
        "base_pos20": positive_rate(base_ret),
        "base_avg20": mean_value(base_ret),
        "base_loss_gt5": rate_le(base_ret, -5),
        "selected_pos20": positive_rate(ret),
        "selected_avg20": mean_value(ret),
        "selected_loss_gt5": rate_le(ret, -5),
        "selected_gain_gt5": rate_ge(ret, 5),
        "delta_pos20_vs_branch": round(positive_rate(ret) - positive_rate(base_ret), 6) if len(ret) else np.nan,
        "delta_avg20_vs_branch": round(mean_value(ret) - mean_value(base_ret), 6) if len(ret) else np.nan,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "rule_id"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        desc = str(group["description"].iloc[0])
        row = {
            "frequency": keys[0],
            "rule_id": keys[1],
            "description": desc,
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_selected_rows_mean": mean_col(prior, "selected_rows"),
            "prior_selected_pos20_mean": mean_col(prior, "selected_pos20"),
            "prior_selected_avg20_mean": mean_col(prior, "selected_avg20"),
            "prior_delta_pos_mean": mean_col(prior, "delta_pos20_vs_branch"),
            "h2026_base_rows": get_val(hrow, "base_rows"),
            "h2026_selected_rows": get_val(hrow, "selected_rows"),
            "h2026_selected_rate": get_val(hrow, "selected_rate_vs_branch"),
            "h2026_selected_pos20": get_val(hrow, "selected_pos20"),
            "h2026_selected_avg20": get_val(hrow, "selected_avg20"),
            "h2026_selected_loss_gt5": get_val(hrow, "selected_loss_gt5"),
            "h2026_delta_pos": get_val(hrow, "delta_pos20_vs_branch"),
            "h2026_delta_avg": get_val(hrow, "delta_avg20_vs_branch"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    if str(row.get("rule_id")) == "small_entry_all":
        return "baseline_branch_reference"
    h_rows = safe_float(row.get("h2026_selected_rows"))
    h_pos = safe_float(row.get("h2026_selected_pos20"))
    h_avg = safe_float(row.get("h2026_selected_avg20"))
    h_loss = safe_float(row.get("h2026_selected_loss_gt5"))
    prior_pos = safe_float(row.get("prior_selected_pos20_mean"))
    prior_delta = safe_float(row.get("prior_delta_pos_mean"))
    if h_rows >= 80 and h_pos >= 0.65 and h_avg > 0 and h_loss <= 0.20 and prior_pos >= 0.65 and prior_delta >= 0:
        return "green_candidate_for_small_ds_confirmation"
    if h_rows >= 50 and h_pos >= 0.60 and h_avg > 0 and h_loss <= 0.25 and prior_pos >= 0.60:
        return "yellow_candidate_needs_fresh_confirmation"
    if h_rows >= 30 and h_avg > 0 and h_pos >= 0.55:
        return "observe_overlay_diagnostic_only"
    return "reject_or_reference_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        25 * safe_float(row.get("h2026_selected_pos20"))
        + safe_float(row.get("h2026_selected_avg20"))
        - 5 * safe_float(row.get("h2026_selected_loss_gt5"))
        + 5 * safe_float(row.get("prior_delta_pos_mean"))
    )


def build_safe_preview(frame: pd.DataFrame, frequency: str, max_rows: int) -> list[dict[str, Any]]:
    sample = frame.sort_values(["target_position", "opp_quantile_in_date", "kline_opp_score"], ascending=[False, False, False]).head(max_rows)
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "date": str(row.get("date")),
                "code": str(row.get("code")).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": str(row.get("time_block")),
                "tool_id": "p0_small_entry_nonprice_overlay_v1",
                "frequency": frequency,
                "base_branch": "branch_stack_v1.small_buy_hold",
                "operation_action_cn": "小仓试探/持有",
                "target_position": round(safe_float(row.get("target_position")), 4),
                "opp_score": round(safe_float(row.get("opp_score")), 6),
                "opp_quantile_in_date": round(safe_float(row.get("opp_quantile_in_date")), 6),
                "kline_opp_score": round(safe_float(row.get("kline_opp_score")), 6),
                "kline_risk_score": round(safe_float(row.get("kline_risk_score")), 6),
                "news_warning_score": preview_num(row.get("news_warning_score")),
                "news_missing_rate": preview_num(row.get("news_missing_rate")),
                "financial_quality_risk_score": preview_num(row.get("financial_quality_risk_score")),
                "financial_report_missing_rate": preview_num(row.get("financial_report_missing_rate")),
                "corr_peer_relative_return_20d": preview_num(row.get("corr_peer_relative_return_20d")),
                "lower_support": preview_num(row.get("lower_support")),
                "upper_overhang": preview_num(row.get("upper_overhang")),
                "kline_rsi14": preview_num(row.get("kline_rsi14")),
                "confirmation_count": int(confirmation_count(pd.DataFrame([row])).iloc[0]),
                "auto_trade": False,
                "agent_instruction": "use this as a small-entry candidate; require semantic confirmation before stronger exposure",
            }
        )
    return rows


def render_report(
    args: argparse.Namespace,
    notes: list[str],
    available_overlays: list[str],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    lines = [
        "# P0 Small-Entry Non-Price Overlay v1",
        "",
        "本报告只审计 `branch_stack_v1.small_buy_hold` 这条小仓试探分叉，目标是判断新闻、财报、同行、筹码和风险字段能否进一步确认或过滤该分叉。实验完全本地运行，不调用 DeepSeek。",
        "",
        "## Setup",
        "",
        f"- frequencies: `{args.frequencies}`",
        f"- kline_feature_group: `{args.kline_feature_group}`",
        "- split: walk-forward prior/validation/target; H2026_1 is final OOT.",
        "- rule metrics contain GT for offline evaluation; agent preview has no GT/future fields.",
        "",
        "## Coverage Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes[-8:]])
    lines.append(f"- overlay_columns_available: `{len(available_overlays)}`")
    lines.extend(
        [
            "",
            "## Main Summary",
            "",
            markdown_table(summary.head(36)),
            "",
            "## H2026 Rule Detail",
            "",
            markdown_table(
                h2026[
                    [
                        "frequency",
                        "rule_id",
                        "base_rows",
                        "selected_rows",
                        "selected_rate_vs_branch",
                        "base_pos20",
                        "selected_pos20",
                        "selected_avg20",
                        "selected_loss_gt5",
                        "delta_pos20_vs_branch",
                        "delta_avg20_vs_branch",
                    ]
                ].sort_values(["selected_pos20", "selected_avg20"], ascending=[False, False])
                if not h2026.empty
                else pd.DataFrame()
            ),
            "",
            "## Hygiene",
            "",
            markdown_table(hygiene) if not hygiene.empty else "_empty_",
            "",
            "## Decision",
            "",
            "- 只有 `green_candidate_for_small_ds_confirmation` 或稳定的 yellow 才进入 DS Flash 小样本；否则继续本地调试。",
            "- 若 overlay 只提高均值但样本行数太少，不能作为用户默认策略，只能作为解释/复核条件。",
            "- 新闻/财报/同行等非价格条件若不能改善该分叉，DS prompt 不应把它们作为机械升权条件。",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def preview_num(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return round(float(parsed), 6)


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict("records"):
            assert_no_future_fields(record)
            safe_record = json_safe(record)
            handle.write(json.dumps(safe_record, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return value


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lower = str(key).lower()
            if key in FUTURE_OR_RESULT_FIELDS or lower.startswith("return_") or "future" in lower or "gt_" in lower:
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def positive_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > 0).mean()), 6) if len(values) else np.nan


def mean_value(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def rate_le(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values <= threshold).mean()), 6) if len(values) else np.nan


def rate_ge(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values >= threshold).mean()), 6) if len(values) else np.nan


def mean_col(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def get_val(row: pd.Series, column: str) -> float:
    if row.empty:
        return np.nan
    try:
        out = float(row.get(column))
    except (TypeError, ValueError):
        return np.nan
    return round(out, 6) if not math.isnan(out) else np.nan


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
