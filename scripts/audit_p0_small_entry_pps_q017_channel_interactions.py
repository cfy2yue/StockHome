"""Audit channel interactions for PPS-Q-017 on P0 small-entry rows.

This local audit uses future returns only for offline evaluation. The output is
meant to guide the next DS on/off shard and should not be inserted into agent
evidence packs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_DETAIL = REPORT_DIR / "p0_small_entry_bookskill_attribution_v1_decision_detail.csv"
DEFAULT_JOINED = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_PREFIX = "p0_small_entry_pps_q017_channel_interactions_v1"
FOCUS_ID = "PPS-Q-017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local channel interactions for PPS-Q-017 small-entry rows.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = load_detail(args.detail)
    joined = load_joined(args.joined)
    data = attach_channels(detail, joined)
    metrics = build_interaction_metrics(data)
    summary = build_rule_summary(metrics)
    paths = write_outputs(args.output_prefix, data, metrics, summary)
    print("A股研究Agent")
    print(f"rows={len(data)} metrics={len(metrics)}")
    print(f"report={paths['report']}")


def load_detail(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    required = {"date", "code", "return_20d", "triggered_skill_ids", "frequency", "target_block"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing detail columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["positive_20d"] = frame["return_20d"].gt(0)
    frame["loss_gt5"] = frame["return_20d"].le(-5)
    return frame.dropna(subset=["date", "code", "return_20d"]).copy()


def load_joined(path: Path) -> pd.DataFrame:
    cols = {
        "date",
        "code",
        "news_count_30d",
        "news_warning_score",
        "news_opportunity_score",
        "news_missing_rate",
        "official_confirmation_score",
        "announcement_materiality_score",
        "financial_report_event_count",
        "financial_report_missing_rate",
        "financial_report_join_status",
        "peer_group_positive_breadth_20d",
        "peer_relative_to_group_20d",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "tushare_area_positive_breadth_20d",
        "prior_return_20d",
        "rsi14",
        "drawdown60",
        "close_above_ma200",
        "ma200_slope20",
        "lower_support",
        "upper_overhang",
        "winner_rate_pct",
    }
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in cols, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame.drop_duplicates(["date", "code"], keep="first")


def attach_channels(detail: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    data = detail.merge(joined, on=["date", "code"], how="left", suffixes=("", "_joined"))
    data["has_pps_q017"] = data["triggered_skill_ids"].map(lambda value: has_strategy(value, FOCUS_ID))
    data["news_available"] = (num(data.get("news_count_30d")) > 0) & (num(data.get("news_missing_rate")) < 1)
    data["news_low_warning"] = data["news_available"] & (num(data.get("news_warning_score")) <= 0.4)
    data["news_positive_or_official"] = data["news_available"] & (
        (num(data.get("news_opportunity_score")) >= 0.4)
        | (num(data.get("official_confirmation_score")) > 0)
        | (num(data.get("announcement_materiality_score")) > 0)
    )
    status = data.get("financial_report_join_status", pd.Series("", index=data.index)).astype(str)
    data["financial_event_matched"] = status.eq("event_window_matched") | (num(data.get("financial_report_event_count")) > 0)
    data["financial_no_recent_event"] = status.eq("no_event_in_window")
    data["financial_missing"] = num(data.get("financial_report_missing_rate")).ge(1) & ~data["financial_no_recent_event"]
    data["peer_breadth_ok"] = (
        num(data.get("peer_group_positive_breadth_20d")).ge(0.5)
        | num(data.get("tushare_industry_positive_breadth_20d")).ge(0.5)
        | num(data.get("tushare_area_positive_breadth_20d")).ge(0.5)
    )
    data["peer_relative_positive"] = (
        num(data.get("peer_relative_to_group_20d")).gt(0)
        | num(data.get("tushare_industry_relative_return_20d")).gt(0)
    )
    data["kline_deep_pullback"] = (num(data.get("prior_return_20d")) <= -5) | (num(data.get("drawdown60")) <= -10)
    data["kline_not_overheated"] = num(data.get("rsi14")).le(70)
    data["kline_above_ma200"] = truthy(data.get("close_above_ma200"))
    data["chip_low_overhang"] = num(data.get("upper_overhang")).le(0.25)
    data["chip_support_visible"] = num(data.get("lower_support")).ge(0.1)
    data["all_triggered_grounded_bool"] = truthy(data.get("all_triggered_grounded"))
    data["weak_skill_present"] = num(data.get("weak_skill_count")).gt(0)
    data["news_or_financial_available"] = data["news_available"] | data["financial_event_matched"]
    data["peer_and_kline_confirm"] = data["peer_relative_positive"] & data["kline_not_overheated"]
    data["soft_gap_bundle"] = data["financial_missing"] & ~data["news_available"]
    return data


def build_interaction_metrics(data: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "news_available",
        "news_low_warning",
        "news_positive_or_official",
        "financial_event_matched",
        "financial_no_recent_event",
        "financial_missing",
        "peer_breadth_ok",
        "peer_relative_positive",
        "kline_deep_pullback",
        "kline_not_overheated",
        "kline_above_ma200",
        "chip_low_overhang",
        "chip_support_visible",
        "all_triggered_grounded_bool",
        "weak_skill_present",
        "news_or_financial_available",
        "peer_and_kline_confirm",
        "soft_gap_bundle",
    ]
    rows: list[dict[str, Any]] = []
    pps = data[data["has_pps_q017"]].copy()
    not_pps = data[~data["has_pps_q017"]].copy()
    rows.append(metric_row("all_small_entry", "ALL", data, data))
    rows.append(metric_row("pps_q017_all", "ALL", pps, data))
    rows.append(metric_row("not_pps_q017_all", "ALL", not_pps, data))
    for flag in flags:
        mask = data[flag].fillna(False).astype(bool)
        pps_mask = pps[flag].fillna(False).astype(bool) if not pps.empty else pd.Series(dtype=bool)
        rows.append(metric_row("pps_q017_flag_true", flag, pps[pps_mask], pps))
        rows.append(metric_row("pps_q017_flag_false", f"not_{flag}", pps[~pps_mask], pps))
        rows.append(metric_row("all_small_entry_flag_true", flag, data[mask], data))
    return pd.DataFrame(rows).round(6)


def metric_row(scope: str, rule_id: str, selected: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    base_returns = pd.to_numeric(baseline.get("return_20d"), errors="coerce").dropna()
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce").dropna()
    base_pos = float(base_returns.gt(0).mean()) if len(base_returns) else None
    base_avg = float(base_returns.mean()) if len(base_returns) else None
    pos = float(returns.gt(0).mean()) if len(returns) else None
    avg = float(returns.mean()) if len(returns) else None
    verdict = local_verdict(len(returns), pos, avg, base_pos, base_avg)
    if rule_id == "weak_skill_present":
        verdict = "weak_skill_gap_diagnostic_not_promotion"
    return {
        "scope": scope,
        "rule_id": rule_id,
        "rows": int(len(returns)),
        "selected_rate": float(len(returns) / len(base_returns)) if len(base_returns) else None,
        "pos20": pos,
        "avg20_pp": avg,
        "loss_gt5": float(returns.le(-5).mean()) if len(returns) else None,
        "base_rows": int(len(base_returns)),
        "base_pos20": base_pos,
        "base_avg20_pp": base_avg,
        "delta_pos": (pos - base_pos) if pos is not None and base_pos is not None else None,
        "delta_avg_pp": (avg - base_avg) if avg is not None and base_avg is not None else None,
        "unique_stocks": int(selected["code"].nunique()) if "code" in selected else 0,
        "verdict": verdict,
    }


def build_rule_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    subset = metrics[metrics["scope"].astype(str).eq("pps_q017_flag_true")].copy()
    if subset.empty:
        return subset
    subset["rank_score"] = (
        pd.to_numeric(subset["delta_pos"], errors="coerce").fillna(-1) * 100
        + pd.to_numeric(subset["delta_avg_pp"], errors="coerce").fillna(-10)
        + pd.to_numeric(subset["selected_rate"], errors="coerce").fillna(0) * 5
    )
    return subset.sort_values(["verdict", "rank_score"], ascending=[True, False]).reset_index(drop=True)


def local_verdict(rows: int, pos: float | None, avg: float | None, base_pos: float | None, base_avg: float | None) -> str:
    if rows < 20:
        return "too_sparse_do_not_promote"
    delta_pos = (pos - base_pos) if pos is not None and base_pos is not None else 0
    delta_avg = (avg - base_avg) if avg is not None and base_avg is not None else 0
    if rows >= 50 and delta_pos >= 0.03 and delta_avg >= 0:
        return "candidate_condition_for_ds_prompt_check"
    if delta_pos < -0.03 or delta_avg < 0:
        return "negative_or_false_filter_risk"
    return "diagnostic_only"


def write_outputs(prefix: str, data: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "detail": REPORT_DIR / f"{safe}_detail.csv",
        "metrics": REPORT_DIR / f"{safe}_metrics.csv",
        "summary": REPORT_DIR / f"{safe}_summary.csv",
        "report": REPORT_DIR / f"{safe}.md",
    }
    safe_detail_cols = [
        "date",
        "code",
        "name",
        "frequency",
        "target_block",
        "has_pps_q017",
        "return_20d",
        "positive_20d",
        "loss_gt5",
        "news_available",
        "news_low_warning",
        "news_positive_or_official",
        "financial_event_matched",
        "financial_no_recent_event",
        "financial_missing",
        "peer_relative_positive",
        "kline_deep_pullback",
        "kline_not_overheated",
        "chip_low_overhang",
        "chip_support_visible",
        "weak_skill_present",
    ]
    data[[col for col in safe_detail_cols if col in data]].to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(metrics, summary, paths), encoding="utf-8")
    return paths


def render_report(metrics: pd.DataFrame, summary: pd.DataFrame, paths: dict[str, Path]) -> str:
    lines = [
        "# P0 Small-Entry PPS-Q-017 Channel Interaction Audit",
        "",
        "本报告是本地离线审计。`return_20d` 只用于评估，不进入 Agent evidence。",
        "",
        "## Main Baselines",
        "",
        markdown_table(metrics[metrics["rule_id"].eq("ALL")], ["scope", "rows", "pos20", "avg20_pp", "loss_gt5", "delta_pos", "delta_avg_pp", "unique_stocks", "verdict"]),
        "",
        "## PPS-Q-017 Condition Summary",
        "",
        markdown_table(summary.head(24), ["rule_id", "rows", "selected_rate", "pos20", "avg20_pp", "loss_gt5", "delta_pos", "delta_avg_pp", "unique_stocks", "verdict"]),
        "",
        "## Interpretation",
        "",
        "- `candidate_condition_for_ds_prompt_check` 只表示该条件值得在 DS prompt/on-off 中检查，不是机械买入或升权规则。",
        "- `negative_or_false_filter_risk` 表示这个条件可能会误导或错杀，后续 Agent 应把它作为反证边界。",
        "- 样本小于 20 的切片默认不晋级。",
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def has_strategy(value: Any, strategy_id: str) -> bool:
    items = [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]
    return str(strategy_id).strip() in items


def num(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").fillna(0.0)
    return pd.to_numeric(pd.Series(value), errors="coerce").fillna(0.0)


def truthy(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        if value.dtype == bool:
            return value.fillna(False)
        text = value.astype(str).str.lower()
        return text.isin({"true", "1", "yes", "y", "t"})
    return pd.Series([bool(value)])


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


if __name__ == "__main__":
    main()
