"""Scout broader P0 small-entry channel rules beyond PPS-Q-017.

This is an offline audit. It joins realized 20-day returns only for evaluation
and writes a separate no-GT preview for later Agent evidence design. The goal is
to find bounded, time-plausible candidates before spending DeepSeek tokens.
"""
from __future__ import annotations

import argparse
import json
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
DEFAULT_PREFIX = "p0_small_entry_general_channel_scout_v1"
PRIOR_BLOCKS = ["H2024_1", "H2025_1"]
FINAL_BLOCK = "H2026_1"
FUTURE_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
    "target_label",
}


JOIN_COLS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit broader small-entry channel rules before DS spending.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--preview-max-rows", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = load_detail(args.detail)
    joined = load_joined(args.joined)
    data = attach_flags(detail, joined)
    rules = build_rulebook()
    metrics, block_metrics = evaluate_rules(data, rules)
    summary = enrich_summary(metrics, block_metrics)
    preview = build_agent_preview(summary, data, rules, max_rows=args.preview_max_rows)
    paths = write_outputs(args.output_prefix, data, metrics, block_metrics, summary, preview)
    print("A股研究Agent")
    print(f"rows={len(data)} rules={len(rules)} summary={len(summary)}")
    print(f"green={int((summary['promotion_status'] == 'green_candidate_for_ds_sample').sum()) if not summary.empty else 0}")
    print(f"yellow={int((summary['promotion_status'] == 'yellow_candidate_for_more_panel').sum()) if not summary.empty else 0}")
    print(f"report={paths['report']}")


def load_detail(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    required = {"date", "code", "name", "return_20d", "target_block", "frequency", "triggered_skill_ids", "operation_action"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing detail columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["positive_20d"] = frame["return_20d"].gt(0)
    frame["loss_gt5"] = frame["return_20d"].le(-5)
    frame = frame[frame["operation_action"].astype(str).eq("small_buy_hold")].copy()
    return frame.dropna(subset=["date", "code", "return_20d"]).copy()


def load_joined(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in JOIN_COLS, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame.drop_duplicates(["date", "code"], keep="first")


def attach_flags(detail: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    joined = joined.drop_duplicates(["date", "code"], keep="first").copy()
    data = detail.merge(joined, on=["date", "code"], how="left", suffixes=("", "_joined"))
    skill_ids = data.get("triggered_skill_ids", pd.Series("", index=data.index)).fillna("").astype(str)
    for skill in ["PPS-Q-017", "PPS-M-003", "PPS-Q-009", "PPS-Q-019", "PPS-Q-023", "DOW-B-004"]:
        data[f"has_{safe_rule_id(skill).lower()}"] = skill_ids.map(lambda value, sid=skill: has_strategy(value, sid))
    data["not_pps_q017"] = ~data["has_pps_q_017"]
    data["freq_weekly_tuesday"] = data["frequency"].astype(str).eq("weekly_tuesday")
    data["freq_weekly_friday"] = data["frequency"].astype(str).eq("weekly_friday")
    data["freq_every_2_weeks"] = data["frequency"].astype(str).eq("every_2_weeks")
    data["news_available"] = (num(data.get("news_count_30d")) > 0) & (num(data.get("news_missing_rate")) < 1)
    data["news_low_warning"] = data["news_available"] & (num(data.get("news_warning_score")) <= 0.4)
    data["news_positive_or_official"] = data["news_available"] & (
        (num(data.get("news_opportunity_score")) >= 0.4)
        | (num(data.get("official_confirmation_score")) > 0)
        | (num(data.get("announcement_materiality_score")) > 0)
    )
    data["news_high_warning"] = data["news_available"] & (num(data.get("news_warning_score")) >= 0.7)
    status = data.get("financial_report_join_status", pd.Series("", index=data.index)).fillna("").astype(str)
    data["financial_event_matched"] = status.eq("event_window_matched") | (num(data.get("financial_report_event_count")) > 0)
    data["financial_no_recent_event"] = status.eq("no_event_in_window")
    data["financial_missing"] = num(data.get("financial_report_missing_rate")).ge(1) & ~data["financial_no_recent_event"]
    data["no_financial_event_matched"] = ~data["financial_event_matched"]
    data["peer_breadth_ok"] = (
        num(data.get("peer_group_positive_breadth_20d")).ge(0.5)
        | num(data.get("tushare_industry_positive_breadth_20d")).ge(0.5)
        | num(data.get("tushare_area_positive_breadth_20d")).ge(0.5)
    )
    data["peer_relative_positive"] = (
        num(data.get("peer_relative_to_group_20d")).gt(0)
        | num(data.get("tushare_industry_relative_return_20d")).gt(0)
    )
    data["peer_weak"] = ~data["peer_relative_positive"] & ~data["peer_breadth_ok"]
    data["kline_deep_pullback"] = (num(data.get("prior_return_20d")) <= -5) | (num(data.get("drawdown60")) <= -10)
    data["kline_oversold"] = num(data.get("rsi14")).le(35)
    data["kline_not_overheated"] = num(data.get("rsi14")).le(70)
    data["kline_above_ma200"] = truthy(data.get("close_above_ma200"))
    data["chip_support_visible"] = num(data.get("lower_support")).ge(0.1)
    data["chip_overhang_ok"] = num(data.get("upper_overhang")).le(0.4)
    data["chip_low_overhang"] = num(data.get("upper_overhang")).le(0.25)
    data["all_triggered_grounded_bool"] = truthy(data.get("all_triggered_grounded"))
    data["weak_skill_present"] = num(data.get("weak_skill_count")).gt(0)
    data["news_financial_clean"] = data["news_low_warning"] & data["no_financial_event_matched"]
    data["clean_chip_setup"] = data["news_financial_clean"] & data["chip_support_visible"] & data["kline_not_overheated"]
    data["clean_chip_pullback"] = data["clean_chip_setup"] & data["kline_deep_pullback"]
    data["clean_chip_peer_not_required"] = data["clean_chip_pullback"] & ~data["news_high_warning"]
    return data


def build_rulebook() -> list[dict[str, Any]]:
    specs: list[tuple[str, str, list[str]]] = [
        ("all_small_entry", "baseline: all branch_stack_v1.small_buy_hold rows", []),
        ("not_pps_q017", "non-PPS-Q-017 small-entry rows", ["not_pps_q017"]),
        ("not_pps_q017_news_low_warning", "non-PPS with low-warning news", ["not_pps_q017", "news_low_warning"]),
        ("not_pps_q017_chip_support", "non-PPS with visible chip support", ["not_pps_q017", "chip_support_visible"]),
        ("not_pps_q017_clean_chip", "non-PPS with clean news/financial and chip support", ["not_pps_q017", "news_financial_clean", "chip_support_visible"]),
        ("not_pps_q017_clean_chip_pullback", "non-PPS clean chip pullback setup", ["not_pps_q017", "clean_chip_pullback"]),
        ("pps_m003_all", "PPS-M-003 all frequencies", ["has_pps_m_003"]),
        ("pps_m003_tuesday", "PPS-M-003 with Tuesday cadence", ["has_pps_m_003", "freq_weekly_tuesday"]),
        ("pps_m003_tuesday_chip", "PPS-M-003 Tuesday plus chip support", ["has_pps_m_003", "freq_weekly_tuesday", "chip_support_visible"]),
        ("pps_m003_tuesday_clean_chip", "PPS-M-003 Tuesday clean chip setup", ["has_pps_m_003", "freq_weekly_tuesday", "clean_chip_setup"]),
        ("pps_q009_chip", "PPS-Q-009 plus chip support", ["has_pps_q_009", "chip_support_visible"]),
        ("pps_q019_chip", "PPS-Q-019 plus chip support", ["has_pps_q_019", "chip_support_visible"]),
        ("dow_b004_clean_chip", "DOW-B-004 clean chip setup", ["has_dow_b_004", "clean_chip_setup"]),
        ("news_financial_clean", "low-warning news and no financial event matched", ["news_financial_clean"]),
        ("news_financial_clean_chip", "clean news/financial plus chip support", ["news_financial_clean", "chip_support_visible"]),
        ("news_financial_clean_chip_pullback", "clean news/financial plus chip pullback", ["clean_chip_pullback"]),
        ("news_financial_clean_chip_pullback_not_pps", "clean chip pullback excluding PPS-Q-017", ["clean_chip_pullback", "not_pps_q017"]),
        ("chip_support_no_overheat", "chip support and K-line not overheated", ["chip_support_visible", "kline_not_overheated"]),
        ("chip_support_pullback", "chip support and pullback", ["chip_support_visible", "kline_deep_pullback", "kline_not_overheated"]),
        ("chip_low_overhang_pullback", "low upper overhang plus pullback", ["chip_low_overhang", "kline_deep_pullback", "kline_not_overheated"]),
        ("oversold_clean_chip", "oversold clean chip setup", ["kline_oversold", "clean_chip_setup"]),
        ("weak_skill_clean_chip", "weak BookSkill signal with clean chip setup", ["weak_skill_present", "clean_chip_setup"]),
        ("all_grounded_clean_chip", "all triggered skills grounded with clean chip setup", ["all_triggered_grounded_bool", "clean_chip_setup"]),
        ("peer_positive_clean_chip", "peer positive clean chip setup", ["peer_relative_positive", "clean_chip_setup"]),
        ("peer_weak_clean_chip", "peer weak clean chip setup", ["peer_weak", "clean_chip_setup"]),
        ("pps_q017_clean_chip", "PPS-Q-017 clean chip setup, for comparison", ["has_pps_q_017", "clean_chip_setup"]),
        ("pps_q017_news_chip", "PPS-Q-017 low-warning news plus chip support", ["has_pps_q_017", "news_low_warning", "chip_support_visible"]),
    ]
    return [{"rule_id": rule_id, "description": desc, "flags": flags} for rule_id, desc, flags in specs]


def evaluate_rules(data: pd.DataFrame, rules: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = data.copy()
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for rule in rules:
        selected = apply_rule(data, rule["flags"])
        rows.append(metric_row(rule, selected, baseline, scope="ALL"))
        for block in sorted(data["target_block"].dropna().astype(str).unique()):
            block_base = data[data["target_block"].astype(str).eq(block)]
            block_selected = selected[selected["target_block"].astype(str).eq(block)]
            block_rows.append(metric_row(rule, block_selected, block_base, scope=block))
    return pd.DataFrame(rows).round(6), pd.DataFrame(block_rows).round(6)


def enrich_summary(metrics: pd.DataFrame, block_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_metrics = metrics.set_index("rule_id")
    for rule_id, group in block_metrics.groupby("rule_id", sort=True):
        row = all_metrics.loc[rule_id].to_dict() if rule_id in all_metrics.index else {"rule_id": rule_id}
        row["rule_id"] = rule_id
        prior = group[group["scope"].isin(PRIOR_BLOCKS)].copy()
        h2026 = group[group["scope"].eq(FINAL_BLOCK)].copy()
        prior_eval = prior[pd.to_numeric(prior["rows"], errors="coerce").ge(8)].copy()
        row.update(
            {
                "prior_evaluable_blocks": int(len(prior_eval)),
                "prior_selected_rows_sum": int(pd.to_numeric(prior_eval.get("rows"), errors="coerce").fillna(0).sum()),
                "prior_delta_pos_hit": mean_bool(prior_eval.get("delta_pos", pd.Series(dtype=float)).ge(0)),
                "prior_delta_avg_hit": mean_bool(prior_eval.get("delta_avg_pp", pd.Series(dtype=float)).ge(0)),
                "prior_pos20_mean": mean_or_none(prior_eval.get("pos20", pd.Series(dtype=float))),
                "prior_avg20_mean": mean_or_none(prior_eval.get("avg20_pp", pd.Series(dtype=float))),
                "h2026_rows": int(pd.to_numeric(h2026.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "h2026_pos20": first_or_none(h2026.get("pos20", pd.Series(dtype=float))),
                "h2026_avg20_pp": first_or_none(h2026.get("avg20_pp", pd.Series(dtype=float))),
                "h2026_loss_gt5": first_or_none(h2026.get("loss_gt5", pd.Series(dtype=float))),
                "h2026_delta_pos": first_or_none(h2026.get("delta_pos", pd.Series(dtype=float))),
                "h2026_delta_avg_pp": first_or_none(h2026.get("delta_avg_pp", pd.Series(dtype=float))),
            }
        )
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    status_order = {
        "green_candidate_for_ds_sample": 0,
        "yellow_candidate_for_more_panel": 1,
        "h2026_bright_prior_insufficient": 2,
        "diagnostic_or_reject": 3,
    }
    frame["_promotion_sort"] = frame["promotion_status"].map(status_order).fillna(99)
    return frame.sort_values(["_promotion_sort", "rank_score"], ascending=[True, False]).drop(columns=["_promotion_sort"]).round(6)


def metric_row(rule: dict[str, Any], selected: pd.DataFrame, baseline: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    returns = pd.to_numeric(selected.get("return_20d"), errors="coerce").dropna()
    base_returns = pd.to_numeric(baseline.get("return_20d"), errors="coerce").dropna()
    pos = float(returns.gt(0).mean()) if len(returns) else None
    avg = float(returns.mean()) if len(returns) else None
    base_pos = float(base_returns.gt(0).mean()) if len(base_returns) else None
    base_avg = float(base_returns.mean()) if len(base_returns) else None
    return {
        "rule_id": rule["rule_id"],
        "description": rule["description"],
        "flags": ";".join(rule["flags"]),
        "scope": scope,
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
        "research_only": True,
        "not_investment_instruction": True,
    }


def promotion_status(row: dict[str, Any]) -> str:
    h_rows = safe_float(row.get("h2026_rows"))
    h_pos = safe_float(row.get("h2026_pos20"))
    h_avg = safe_float(row.get("h2026_avg20_pp"))
    h_loss = safe_float(row.get("h2026_loss_gt5"))
    prior_blocks = safe_float(row.get("prior_evaluable_blocks"))
    prior_pos_hit = safe_float(row.get("prior_delta_pos_hit"))
    prior_avg_hit = safe_float(row.get("prior_delta_avg_hit"))
    prior_rows = safe_float(row.get("prior_selected_rows_sum"))
    if prior_blocks >= 2 and prior_rows >= 30 and h_rows >= 20 and h_pos >= 0.65 and h_avg >= 5 and h_loss <= 0.18 and prior_pos_hit >= 0.5 and prior_avg_hit >= 0.5:
        return "green_candidate_for_ds_sample"
    if prior_blocks >= 2 and prior_rows >= 24 and h_rows >= 15 and h_pos >= 0.62 and h_avg >= 4 and prior_avg_hit >= 0.5:
        return "yellow_candidate_for_more_panel"
    if h_rows >= 20 and h_pos >= 0.68 and h_avg >= 5:
        return "h2026_bright_prior_insufficient"
    return "diagnostic_or_reject"


def rank_score(row: dict[str, Any]) -> float:
    return (
        safe_float(row.get("h2026_delta_pos")) * 100
        + safe_float(row.get("h2026_delta_avg_pp"))
        + safe_float(row.get("prior_delta_pos_hit")) * 10
        + safe_float(row.get("prior_delta_avg_hit")) * 10
        + min(safe_float(row.get("h2026_rows")), 100) / 20
    )


def build_agent_preview(summary: pd.DataFrame, data: pd.DataFrame, rules: list[dict[str, Any]], *, max_rows: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    rule_map = {rule["rule_id"]: rule for rule in rules}
    keep_status = {"green_candidate_for_ds_sample", "yellow_candidate_for_more_panel", "h2026_bright_prior_insufficient"}
    top = summary[summary["promotion_status"].isin(keep_status)].sort_values("rank_score", ascending=False).head(12)
    previews: list[dict[str, Any]] = []
    per_rule = max(1, max_rows // max(1, len(top)))
    for _, row in top.iterrows():
        rule = rule_map.get(str(row["rule_id"]))
        if not rule:
            continue
        selected = apply_rule(data, rule["flags"]).sort_values(["target_block", "date", "code"]).head(per_rule)
        for _, item in selected.iterrows():
            record = {
                "tool_id": "p0_small_entry_general_channel_scout_v1",
                "rule_id": row["rule_id"],
                "promotion_status": row["promotion_status"],
                "date": str(item.get("date")),
                "code": str(item.get("code")).zfill(6),
                "name": str(item.get("name") or ""),
                "target_block": str(item.get("target_block") or ""),
                "frequency": str(item.get("frequency") or ""),
                "flags": str(row.get("flags") or ""),
                "h2026_pos20_offline_summary": safe_round(row.get("h2026_pos20")),
                "h2026_avg20_offline_summary": safe_round(row.get("h2026_avg20_pp")),
                "prior_delta_hit_summary": f"pos={safe_round(row.get('prior_delta_pos_hit'))};avg={safe_round(row.get('prior_delta_avg_hit'))}",
                "agent_use": "offline_scout_summary_only_not_alpha",
                "forbidden_use": "do_not_use_as_standalone_trade_instruction_or_without_current_evidence",
                "research_only": True,
                "not_investment_instruction": True,
            }
            assert_no_future_fields(record)
            previews.append(record)
    return previews[:max_rows]


def write_outputs(
    prefix: str,
    detail: pd.DataFrame,
    metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    preview: list[dict[str, Any]],
) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "detail": REPORT_DIR / f"{safe}_detail.csv",
        "metrics": REPORT_DIR / f"{safe}_metrics.csv",
        "block_metrics": REPORT_DIR / f"{safe}_block_metrics.csv",
        "summary": REPORT_DIR / f"{safe}_summary.csv",
        "agent_preview": REPORT_DIR / f"{safe}_agent_preview_no_gt.jsonl",
        "report": REPORT_DIR / f"{safe}.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    block_metrics.to_csv(paths["block_metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    with paths["agent_preview"].open("w", encoding="utf-8") as handle:
        for row in preview:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["report"].write_text(render_report(summary, block_metrics, paths), encoding="utf-8")
    return paths


def render_report(summary: pd.DataFrame, block_metrics: pd.DataFrame, paths: dict[str, Path]) -> str:
    lines = [
        "# P0 Small-Entry General Channel Scout",
        "",
        "本地离线 scout：收益字段只用于评估，不进入 Agent evidence。目标是寻找非 PPS-Q-017 或更宽 small-entry 分支，再决定是否进入 DS Flash。",
        "",
        "## Promotion Summary",
        "",
        markdown_table(
            summary,
            [
                "rule_id",
                "promotion_status",
                "rows",
                "pos20",
                "avg20_pp",
                "prior_evaluable_blocks",
                "prior_delta_pos_hit",
                "prior_delta_avg_hit",
                "h2026_rows",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_delta_pos",
                "h2026_delta_avg_pp",
            ],
            max_rows=40,
        ),
        "",
        "## H2026 Bright But Prior Insufficient",
        "",
        markdown_table(
            summary[summary["promotion_status"].astype(str).eq("h2026_bright_prior_insufficient")],
            ["rule_id", "rows", "pos20", "avg20_pp", "prior_evaluable_blocks", "h2026_rows", "h2026_pos20", "h2026_avg20_pp"],
            max_rows=20,
        ),
        "",
        "## Key Block Metrics",
        "",
        markdown_table(
            block_metrics[block_metrics["rule_id"].isin(summary.head(12)["rule_id"])],
            ["rule_id", "scope", "rows", "pos20", "avg20_pp", "delta_pos", "delta_avg_pp", "loss_gt5"],
            max_rows=80,
        ),
        "",
        "## Interpretation",
        "",
        "- `green_candidate_for_ds_sample` 才允许进入小规模 Flash on/off；`yellow_candidate_for_more_panel` 只能先扩 panel 或做 dry-run。",
        "- `h2026_bright_prior_insufficient` 表示最新块好看但时间泛化证据不足，不能直接烧 Pro。",
        "- 本报告会特别关注非 PPS-Q-017、PPS-M-003 频率门、新闻低风险、财报无事件、筹码支撑和 K线不过热组合。",
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def apply_rule(data: pd.DataFrame, flags: list[str]) -> pd.DataFrame:
    if not flags:
        return data.copy()
    mask = pd.Series(True, index=data.index)
    for flag in flags:
        if flag not in data:
            mask &= False
        else:
            mask &= data[flag].fillna(False).astype(bool)
    return data[mask].copy()


def has_strategy(value: Any, strategy_id: str) -> bool:
    target = str(strategy_id).strip()
    return target in {item.strip() for item in str(value or "").split(";") if item.strip()}


def safe_rule_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def safe_prefix(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").fillna(0.0)
    return pd.Series(dtype=float)


def truthy(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == bool:
            return values.fillna(False)
        text = values.fillna("").astype(str).str.lower()
        return text.isin({"1", "true", "yes", "y"})
    return pd.Series(dtype=bool)


def mean_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def first_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def mean_bool(values: Any) -> float | None:
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return float(series.astype(bool).mean())


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_round(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        leaked = sorted(set(map(str, value.keys())) & FUTURE_KEYS)
        if leaked:
            raise ValueError(f"future/result key leaked into preview: {leaked}")
        for item in value.values():
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame.columns]
    data = frame[cols].head(max_rows).fillna("").astype(str)
    rows = data.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


if __name__ == "__main__":
    main()
