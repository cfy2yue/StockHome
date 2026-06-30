from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_JOINED_GT = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "news_financial_interaction_local_v1"

FORBIDDEN_PLAN_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
}

USECOLS = [
    "date",
    "code",
    "name",
    "set",
    "gt_status",
    "return_20d",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "close_above_ma200",
    "drawdown60",
    "atr20_pct",
    "news_count_30d",
    "event_count",
    "news_missing_rate",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
    "news_negative_materiality_30d",
    "news_positive_materiality_30d",
    "news_conflict_intensity_30d",
    "peer_group_news_count_avg",
    "peer_group_news_risk_avg",
    "peer_group_news_opportunity_avg",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "triggered_skills",
    "data_gaps",
    "financial_report_missing_rate",
    "financial_report_event_count",
    "financial_report_join_status",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_event_types",
    "financial_report_available_at",
]

NUMERIC_COLUMNS = [
    "return_20d",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "drawdown60",
    "atr20_pct",
    "news_count_30d",
    "event_count",
    "news_missing_rate",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
    "news_negative_materiality_30d",
    "news_positive_materiality_30d",
    "news_conflict_intensity_30d",
    "peer_group_news_count_avg",
    "peer_group_news_risk_avg",
    "peer_group_news_opportunity_avg",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "financial_report_missing_rate",
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
]

PLAN_COLUMNS = [
    "candidate_rule",
    "time_block",
    "date",
    "code",
    "name",
    "reason_to_test",
    "sample_stock_concentration_note",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "event_count",
    "news_missing_rate",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_report_event_types",
    "triggered_skills",
]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    reason: str
    rule_type: str
    selector: pd.Series


def main() -> None:
    parser = argparse.ArgumentParser(description="Local news x financial x peer interaction search without DeepSeek calls.")
    parser.add_argument("--joined-gt", default=str(DEFAULT_JOINED_GT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-samples-per-rule", type=int, default=4)
    parser.add_argument("--max-plan-rows", type=int, default=32)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    frame = load_rows(Path(args.joined_gt))
    enriched = add_interaction_columns(frame)
    rule_metrics, block_metrics = build_rule_tables(enriched)
    plan = build_sample_plan(enriched, rule_metrics, max_samples_per_rule=args.max_samples_per_rule, max_rows=args.max_plan_rows)
    assert_no_future_plan_columns(plan)

    metrics_path = report_dir / f"{OUTPUT_PREFIX}_rule_metrics.csv"
    block_path = report_dir / f"{OUTPUT_PREFIX}_block_metrics.csv"
    plan_path = report_dir / f"{OUTPUT_PREFIX}_ds_sample_plan.csv"
    summary_path = report_dir / f"{OUTPUT_PREFIX}_summary.md"
    rule_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    block_metrics.to_csv(block_path, index=False, encoding="utf-8-sig")
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    write_summary(summary_path, enriched, rule_metrics, block_metrics, plan)

    print("A股研究Agent")
    print(f"rows={len(enriched)}")
    print(f"rule_metrics={metrics_path}")
    print(f"block_metrics={block_path}")
    print(f"sample_plan={plan_path}")
    print(f"summary={summary_path}")


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    available = set(pd.read_csv(path, nrows=0).columns)
    usecols = [col for col in USECOLS if col in available]
    frame = pd.read_csv(path, dtype={"code": str}, usecols=usecols, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame = frame[pd.to_numeric(frame.get("return_20d"), errors="coerce").notna()].copy()
    for column in USECOLS:
        if column not in frame:
            frame[column] = _default_value(column, frame.index)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["news_missing_rate"] = frame["news_missing_rate"].fillna(1.0)
    frame["financial_report_missing_rate"] = frame["financial_report_missing_rate"].fillna(1.0)
    frame["event_count"] = frame["event_count"].fillna(0)
    frame["news_count_30d"] = frame["news_count_30d"].fillna(0)
    frame["financial_report_event_count"] = frame["financial_report_event_count"].fillna(0)
    return frame.reset_index(drop=True)


def add_interaction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    dates = pd.to_datetime(data["date"], errors="coerce")
    data["time_block"] = dates.map(_time_block)
    status = data["financial_report_join_status"].fillna("").astype(str)
    data["financial_available"] = status.eq("event_window_matched") | data["financial_report_event_count"].gt(0)
    data["news_available"] = data["news_missing_rate"].lt(0.8) & (data["event_count"].add(data["news_count_30d"], fill_value=0).gt(0))
    data["peer_confirmed"] = data["peer_relative_to_group_20d"].ge(0) & data["peer_group_positive_breadth_20d"].ge(0.5)
    data["peer_weak"] = data["peer_relative_to_group_20d"].lt(0) | data["peer_group_positive_breadth_20d"].lt(0.45)
    data["overheat_flag"] = data["prior_return_20d"].ge(20) | data["rsi14"].ge(75) | data["relative_strength_rank"].ge(0.9)
    data["pullback_flag"] = data["prior_return_20d"].le(-5) & data["relative_strength_rank"].le(0.55)
    data["financial_high_risk"] = data["financial_available"] & data["financial_quality_risk_score"].ge(0.6)
    data["financial_nonpositive"] = data["financial_available"] & data["financial_surprise_score"].le(0)
    data["financial_clean_positive"] = (
        data["financial_available"]
        & data["financial_quality_risk_score"].lt(0.3)
        & data["financial_surprise_score"].ge(0.2)
    )
    data["news_high_risk"] = (
        data["news_warning_score"].ge(0.3)
        | data["news_negative_materiality_30d"].ge(0.7)
        | data["news_conflict_intensity_30d"].ge(0.5)
    )
    data["news_high_opportunity"] = data["news_opportunity_score"].ge(0.3) | data["news_positive_materiality_30d"].ge(0.7)
    data["peer_active_self_silent"] = (
        data["peer_group_news_count_avg"].ge(2)
        & (data["event_count"].fillna(0).le(1) | data["news_missing_rate"].ge(0.8))
    )
    data["self_vs_peer_attention_gap"] = data["event_count"].fillna(0) - data["peer_group_news_count_avg"].fillna(0)
    data["routine_official_low_signal"] = (
        data["news_available"]
        & data["official_confirmation_score"].ge(0.8)
        & data["announcement_materiality_score"].ge(0.6)
        & data["news_warning_score"].fillna(0).le(0.05)
        & data["news_opportunity_score"].fillna(0).le(0.05)
    )
    data["policy_support"] = data["news_available"] & data["policy_background_score"].ge(0.2)
    return data


def build_rule_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _metrics(frame)
    specs = interaction_rule_specs(frame)
    metric_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for spec in specs:
        subset = frame[spec.selector.fillna(False)].copy()
        row = {"rule_id": spec.rule_id, "rule_type": spec.rule_type, "reason_to_test": spec.reason, **_metrics(subset)}
        row.update(_baseline_deltas(row, baseline))
        row["block_count"] = int(subset["time_block"].nunique()) if not subset.empty else 0
        row["blocks_pos_ge_0.60"] = _blocks_above(subset, threshold=0.60)
        row["blocks_pos_ge_0.50"] = _blocks_above(subset, threshold=0.50)
        row["avg_news_missing_rate"] = _round(subset["news_missing_rate"].mean()) if not subset.empty else None
        row["status"] = _rule_status(row)
        row["next_action"] = _next_action(row)
        metric_rows.append(row)
        for block, group in subset.groupby("time_block", sort=True):
            block_row = {"rule_id": spec.rule_id, "time_block": block, **_metrics(group)}
            block_rows.append(block_row)
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["status", "delta_positive_20d_rate", "delta_loss_20d_over_5_rate", "rows"],
            ascending=[True, False, True, False],
        )
    return metrics, pd.DataFrame(block_rows)


def interaction_rule_specs(frame: pd.DataFrame) -> list[RuleSpec]:
    return [
        RuleSpec(
            "news_financial_available_v1",
            "普通新闻和财报事件同时可用，测试双通道是否优于单通道。",
            "coverage",
            frame["news_available"] & frame["financial_available"],
        ),
        RuleSpec(
            "financial_nonpositive_news_available_v2",
            "财报偏离不正向但普通新闻可用，测试新闻是否能帮助区分风险/复核。",
            "risk_probe",
            frame["financial_nonpositive"] & frame["news_available"],
        ),
        RuleSpec(
            "financial_nonpositive_news_peer_weak_v1",
            "财报不正向、新闻可用且同行弱，测试是否应更防守。",
            "risk_probe",
            frame["financial_nonpositive"] & frame["news_available"] & frame["peer_weak"],
        ),
        RuleSpec(
            "financial_nonpositive_news_peer_confirmed_v1",
            "财报不正向但同行确认，作为 peer confirmation 对照。",
            "control",
            frame["financial_nonpositive"] & frame["news_available"] & frame["peer_confirmed"],
        ),
        RuleSpec(
            "financial_high_risk_news_available_v1",
            "高财务质量风险且新闻可用，测试财报风险在有新闻时是否仍应降权。",
            "risk_probe",
            frame["financial_high_risk"] & frame["news_available"],
        ),
        RuleSpec(
            "financial_high_risk_news_missing_v1",
            "高财务质量风险但新闻缺失，测试缺新闻是否应转为信息不足。",
            "risk_probe",
            frame["financial_high_risk"] & ~frame["news_available"],
        ),
        RuleSpec(
            "news_high_opportunity_peer_confirmed_financial_clean_v1",
            "新闻机会、同行确认、财报低风险同时出现，测试是否是更可信的机会组合。",
            "opportunity_probe",
            frame["news_high_opportunity"] & frame["peer_confirmed"] & frame["financial_clean_positive"],
        ),
        RuleSpec(
            "news_high_opportunity_peer_weak_or_fin_missing_v1",
            "新闻机会强但同行弱或财报缺失，测试是否是假机会/拥挤噪音。",
            "counterevidence",
            frame["news_high_opportunity"] & (frame["peer_weak"] | ~frame["financial_available"]),
        ),
        RuleSpec(
            "peer_active_self_silent_v1",
            "同行被新闻提及但目标股沉默，测试相对关注缺口。",
            "relative_news",
            frame["peer_active_self_silent"],
        ),
        RuleSpec(
            "routine_official_low_signal_v1",
            "官方/公告很多但风险和机会分都低，测试常规公告是否不应当作 alpha。",
            "control",
            frame["routine_official_low_signal"],
        ),
        RuleSpec(
            "news_risk_high_peer_weak_v1",
            "新闻风险高且同行弱，测试风险扩散/行业弱确认。",
            "risk_probe",
            frame["news_high_risk"] & frame["peer_weak"],
        ),
        RuleSpec(
            "policy_support_peer_confirmed_v1",
            "政策背景和同行确认同时出现，测试政策类新闻是否需要同行确认。",
            "opportunity_probe",
            frame["policy_support"] & frame["peer_confirmed"],
        ),
        RuleSpec(
            "policy_support_peer_weak_v1",
            "有政策背景但同行弱，作为政策机会反证对照。",
            "counterevidence",
            frame["policy_support"] & frame["peer_weak"],
        ),
        RuleSpec(
            "pullback_news_financial_peer_confirmed_v1",
            "回撤后新闻、财报、同行均可见，测试是否比孤立 pullback 更可靠。",
            "opportunity_probe",
            frame["pullback_flag"] & frame["news_available"] & frame["financial_available"] & frame["peer_confirmed"],
        ),
        RuleSpec(
            "overheat_financial_nonpositive_news_available_v1",
            "过热后财报不正向且新闻可用，测试是否比单纯过热更危险。",
            "risk_probe",
            frame["overheat_flag"] & frame["financial_nonpositive"] & frame["news_available"],
        ),
    ]


def build_sample_plan(
    frame: pd.DataFrame,
    rule_metrics: pd.DataFrame,
    *,
    max_samples_per_rule: int,
    max_rows: int,
) -> pd.DataFrame:
    if rule_metrics.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    preferred_status = {"observe_promising", "counterevidence_risk", "observe_control"}
    selected_rules = rule_metrics[rule_metrics["status"].isin(preferred_status)].copy()
    if selected_rules.empty:
        selected_rules = rule_metrics.head(6).copy()
    selected_rule_ids = selected_rules["rule_id"].head(8).tolist()
    specs = {spec.rule_id: spec for spec in interaction_rule_specs(frame)}
    rows: list[pd.DataFrame] = []
    for rule_id in selected_rule_ids:
        spec = specs.get(rule_id)
        if spec is None:
            continue
        subset = frame[spec.selector.fillna(False)].copy()
        if subset.empty:
            continue
        subset["_severity"] = _severity_score(subset, spec.rule_type)
        sample = _diversified_sample(subset, max_rows=max_samples_per_rule)
        sample["candidate_rule"] = spec.rule_id
        sample["reason_to_test"] = spec.reason
        sample["sample_stock_concentration_note"] = _concentration_note(subset)
        rows.append(sample)
    if not rows:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    plan = pd.concat(rows, ignore_index=True)
    plan = _diversified_sample(plan, max_rows=max_rows)
    for column in PLAN_COLUMNS:
        if column not in plan:
            plan[column] = ""
    return plan[PLAN_COLUMNS].copy()


def assert_no_future_plan_columns(plan: pd.DataFrame) -> None:
    leaked = sorted(set(plan.columns) & FORBIDDEN_PLAN_COLUMNS)
    if leaked:
        raise ValueError(f"sample plan contains future/result fields: {leaked}")


def write_summary(path: Path, frame: pd.DataFrame, metrics: pd.DataFrame, block_metrics: pd.DataFrame, plan: pd.DataFrame) -> None:
    baseline = _metrics(frame)
    promising = metrics[metrics["status"].eq("observe_promising")].head(8) if not metrics.empty else pd.DataFrame()
    risk = metrics[metrics["status"].eq("counterevidence_risk")].head(8) if not metrics.empty else pd.DataFrame()
    controls = metrics[metrics["status"].eq("observe_control")].head(8) if not metrics.empty else pd.DataFrame()
    lines = [
        "# News x Financial Interaction Local Search V1",
        "",
        "本报告只做研究辅助，不自动交易，不接券商接口，不输出投资指令。",
        "",
        "## Scope",
        "",
        f"- evaluated_rows: `{baseline['rows']}`",
        f"- unique_stocks: `{baseline['unique_stocks']}`",
        f"- time_blocks: `{frame['time_block'].nunique()}`",
        f"- baseline_positive_20d_rate: `{baseline['positive_20d_rate']}`",
        f"- baseline_avg_return_20d: `{baseline['avg_return_20d']}`",
        f"- baseline_loss_20d_over_5_rate: `{baseline['loss_20d_over_5_rate']}`",
        "",
        "## Candidate Rule Summary",
        "",
        _table(metrics),
        "",
        "## Promising Observations",
        "",
        _table(promising),
        "",
        "## Risk / Counter Evidence",
        "",
        _table(risk),
        "",
        "## Controls",
        "",
        _table(controls),
        "",
        "## Block Metrics",
        "",
        _table(block_metrics),
        "",
        "## Next DS Sample Plan",
        "",
        f"- rows: `{len(plan)}`",
        "- sample_plan file: `reports/date_generalization/news_financial_interaction_local_v1_ds_sample_plan.csv`",
        "- 该 sample plan 不包含 `return_20d`、`gt_status` 或其他未来结果字段。",
        "- 下一步若调用 DeepSeek，建议先 dry-run + leakage audit，再用 Flash 小 shard；变体保留 `no_news`、`keyword_only`、`uncertainty_only_questionnaire`、`quality_only_questionnaire`、`no_financial_report_channel`、`news_plus_financial_report_guarded`。",
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(metrics),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "unique_stocks": 0,
            "top_stock_share": None,
            "avg_return_20d": None,
            "median_return_20d": None,
            "positive_20d_rate": None,
            "loss_20d_over_5_rate": None,
            "return_20d_std": None,
        }
    returns = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    if returns.empty:
        return _metrics(frame.iloc[0:0])
    top_share = frame["code"].value_counts(normalize=True).iloc[0] if "code" in frame and not frame.empty else None
    return {
        "rows": int(len(returns)),
        "unique_stocks": int(frame["code"].nunique()) if "code" in frame else 0,
        "top_stock_share": _round(top_share),
        "avg_return_20d": _round(returns.mean()),
        "median_return_20d": _round(returns.median()),
        "positive_20d_rate": _round((returns > 0).mean()),
        "loss_20d_over_5_rate": _round((returns <= -5).mean()),
        "return_20d_std": _round(returns.std(ddof=0)),
    }


def _baseline_deltas(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_positive_20d_rate": baseline["positive_20d_rate"],
        "baseline_avg_return_20d": baseline["avg_return_20d"],
        "baseline_loss_20d_over_5_rate": baseline["loss_20d_over_5_rate"],
        "delta_positive_20d_rate": _delta(row.get("positive_20d_rate"), baseline.get("positive_20d_rate")),
        "delta_avg_return_20d": _delta(row.get("avg_return_20d"), baseline.get("avg_return_20d")),
        "delta_loss_20d_over_5_rate": _delta(row.get("loss_20d_over_5_rate"), baseline.get("loss_20d_over_5_rate")),
    }


def _rule_status(row: dict[str, Any]) -> str:
    rows = int(row.get("rows") or 0)
    unique = int(row.get("unique_stocks") or 0)
    top_share = row.get("top_stock_share")
    delta_pos = row.get("delta_positive_20d_rate")
    delta_loss = row.get("delta_loss_20d_over_5_rate")
    blocks = int(row.get("block_count") or 0)
    hit60 = int(row.get("blocks_pos_ge_0.60") or 0)
    rule_type = str(row.get("rule_type") or "")
    if rows < 50 or unique < 10 or (top_share is not None and top_share > 0.25):
        return "insufficient_or_concentrated"
    if rule_type == "control":
        return "observe_control"
    if delta_pos is not None and delta_loss is not None and (delta_pos <= -0.05 or delta_loss >= 0.05):
        return "counterevidence_risk"
    if (
        delta_pos is not None
        and delta_loss is not None
        and delta_pos >= 0.03
        and delta_loss <= -0.02
        and hit60 >= max(1, blocks // 2)
    ):
        return "observe_promising"
    return "observe_neutral"


def _next_action(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status == "observe_promising":
        return "small Flash shard with no_news/no_financial controls; do not promote before cross-block validation"
    if status == "counterevidence_risk":
        return "use as counter-evidence candidate; test guarded prompt and zero-exposure handling"
    if status == "observe_control":
        return "keep as neutral/control arm to detect prompt-induced over-optimism"
    if status == "insufficient_or_concentrated":
        return "do not spend DS tokens until sample size and stock diversity improve"
    return "keep in local monitoring; not enough evidence for default rule"


def _blocks_above(frame: pd.DataFrame, *, threshold: float) -> int:
    if frame.empty:
        return 0
    count = 0
    for _, group in frame.groupby("time_block"):
        returns = pd.to_numeric(group["return_20d"], errors="coerce").dropna()
        if len(returns) >= 5 and float((returns > 0).mean()) >= threshold:
            count += 1
    return int(count)


def _diversified_sample(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if frame.empty or max_rows <= 0:
        return frame.iloc[0:0].copy()
    data = frame.copy()
    if "_severity" not in data:
        data["_severity"] = _severity_score(data, "default")
    data = data.sort_values(["_severity", "time_block", "code"], ascending=[False, True, True])
    chosen = []
    seen_codes: set[str] = set()
    block_counts: dict[str, int] = {}
    max_per_block = max(2, max_rows // 3)
    for _, row in data.iterrows():
        code = str(row.get("code"))
        block = str(row.get("time_block"))
        if code in seen_codes:
            continue
        if block_counts.get(block, 0) >= max_per_block:
            continue
        chosen.append(row)
        seen_codes.add(code)
        block_counts[block] = block_counts.get(block, 0) + 1
        if len(chosen) >= max_rows:
            break
    if len(chosen) < max_rows:
        chosen_keys = {(str(row.get("date")), str(row.get("code"))) for row in chosen}
        seen_codes = {str(row.get("code")) for row in chosen}
        for _, row in data.iterrows():
            key = (str(row.get("date")), str(row.get("code")))
            code = str(row.get("code"))
            if key in chosen_keys or code in seen_codes:
                continue
            chosen.append(row)
            chosen_keys.add(key)
            seen_codes.add(code)
            if len(chosen) >= max_rows:
                break
    if len(chosen) < max_rows:
        chosen_keys = {(str(row.get("date")), str(row.get("code"))) for row in chosen}
        for _, row in data.iterrows():
            key = (str(row.get("date")), str(row.get("code")))
            if key in chosen_keys:
                continue
            chosen.append(row)
            chosen_keys.add(key)
            if len(chosen) >= max_rows:
                break
    result = pd.DataFrame(chosen)
    return result.drop(columns=["_severity"], errors="ignore").reset_index(drop=True)


def _severity_score(frame: pd.DataFrame, rule_type: str) -> pd.Series:
    risk = frame["news_warning_score"].fillna(0) + frame["financial_quality_risk_score"].fillna(0)
    opportunity = frame["news_opportunity_score"].fillna(0) + frame["financial_surprise_score"].fillna(0).clip(lower=0)
    peer_gap = frame["peer_relative_to_group_20d"].fillna(0).abs() / 20
    if rule_type in {"risk_probe", "counterevidence"}:
        return risk + peer_gap + frame["news_missing_rate"].fillna(1) * 0.2
    if rule_type == "opportunity_probe":
        return opportunity + frame["peer_group_positive_breadth_20d"].fillna(0)
    return risk + opportunity + peer_gap


def _concentration_note(frame: pd.DataFrame) -> str:
    metrics = _metrics(frame)
    return f"rows={metrics['rows']}; stocks={metrics['unique_stocks']}; top_share={metrics['top_stock_share']}"


def _interpretation_lines(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return ["- No rule metrics generated."]
    lines = []
    promising = metrics[metrics["status"].eq("observe_promising")]
    risk = metrics[metrics["status"].eq("counterevidence_risk")]
    weak = metrics[metrics["status"].eq("insufficient_or_concentrated")]
    if promising.empty:
        lines.append("- 本地搜索没有发现可直接升级的正向 alpha；即使有 observe_promising，也必须先过 DS 消融和跨时间块验证。")
    else:
        ids = ", ".join(promising["rule_id"].head(5).astype(str))
        lines.append(f"- observe_promising 候选：{ids}。这些只说明值得小样本 DS 验证，不是默认策略。")
    if not risk.empty:
        ids = ", ".join(risk["rule_id"].head(5).astype(str))
        lines.append(f"- counterevidence_risk 候选：{ids}。这些更适合进入反证/降权问卷，而不是正向评分。")
    if not weak.empty:
        lines.append("- 多个规则被标为 insufficient_or_concentrated，说明当前新闻/财报历史覆盖仍不足，不能靠少数股票样本训练策略。")
    lines.append("- 下一步应先对 sample plan 做 dry-run 和 evidence leakage audit，再决定是否花 DS Flash token。")
    return lines


def _time_block(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.year <= 2022:
        return f"H{ts.year}_legacy"
    half = 1 if ts.month <= 6 else 2
    return f"H{ts.year}_{half}"


def _default_value(column: str, index: pd.Index) -> pd.Series:
    if column in NUMERIC_COLUMNS:
        default = 1.0 if column.endswith("missing_rate") else 0.0
        return pd.Series(default, index=index)
    return pd.Series("", index=index)


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline):
        return None
    return _round(float(value) - float(baseline))


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _table(frame: pd.DataFrame, *, max_rows: int = 60) -> str:
    if frame.empty:
        return "_empty_"
    return frame.head(max_rows).to_markdown(index=False)


if __name__ == "__main__":
    main()
