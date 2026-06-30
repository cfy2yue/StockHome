from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_JOINED_GT = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_REPORT_DIR = ROOT / "reports" / "date_generalization"

OUTPUT_PREFIX = "financial_report_local_stratified"

USECOLS = [
    "date",
    "code",
    "name",
    "gt_status",
    "return_20d",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "news_missing_rate",
    "triggered_skills",
    "financial_report_event_count",
    "financial_report_join_status",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_event_types",
    "financial_report_available_at",
]

DS_PLAN_COLUMNS = [
    "candidate_rule",
    "time_block",
    "date",
    "code",
    "name",
    "financial_report_event_count",
    "financial_report_event_types",
    "financial_surprise_score",
    "financial_quality_risk_score",
    "financial_disclosure_quality_score",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "news_missing_rate",
    "triggered_skills",
    "reason_to_test",
    "sample_stock_concentration_note",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local stratified backtest analysis for matched financial-report event rows.")
    parser.add_argument("--joined-gt", default=str(DEFAULT_JOINED_GT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--max-samples-per-rule", type=int, default=8)
    parser.add_argument("--diversified-target-rows", type=int, default=24)
    parser.add_argument("--diversified-max-per-stock", type=int, default=2)
    parser.add_argument("--diversified-max-per-block", type=int, default=6)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    matched = load_matched_financial_rows(Path(args.joined_gt))
    enriched = add_strata_columns(matched)
    metrics = build_stratified_metrics(enriched)
    rules = build_rule_candidate_metrics(enriched)
    threshold_search = build_financial_threshold_search(enriched)
    sample_audit, ds_plan = build_candidate_samples(enriched, max_samples_per_rule=args.max_samples_per_rule)
    diversified_audit, diversified_plan = build_diversified_guard_samples(
        enriched,
        target_rows=args.diversified_target_rows,
        max_per_stock=args.diversified_max_per_stock,
        max_per_block=args.diversified_max_per_block,
    )

    metrics_path = report_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    rules_path = report_dir / f"{OUTPUT_PREFIX}_rule_candidates.csv"
    threshold_path = report_dir / "financial_report_threshold_search.csv"
    audit_path = report_dir / f"{OUTPUT_PREFIX}_candidate_sample_audit.csv"
    plan_path = report_dir / "financial_report_next_ds_sample_plan.csv"
    diversified_audit_path = report_dir / "financial_report_risk_guard_diversified_sample_audit.csv"
    diversified_plan_path = report_dir / "financial_report_risk_guard_diversified_sample_plan.csv"
    summary_path = report_dir / f"{OUTPUT_PREFIX}_analysis.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    rules.to_csv(rules_path, index=False, encoding="utf-8-sig")
    threshold_search.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    sample_audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    ds_plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    diversified_audit.to_csv(diversified_audit_path, index=False, encoding="utf-8-sig")
    diversified_plan.to_csv(diversified_plan_path, index=False, encoding="utf-8-sig")
    write_summary(summary_path, enriched, metrics, rules, threshold_search, ds_plan, diversified_plan)

    print("A股研究Agent")
    print(f"matched_rows={len(enriched)}")
    print(f"metrics={metrics_path}")
    print(f"rules={rules_path}")
    print(f"threshold_search={threshold_path}")
    print(f"ds_plan={plan_path}")
    print(f"diversified_ds_plan={diversified_plan_path}")
    print(f"summary={summary_path}")


def load_matched_financial_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    available_cols = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in USECOLS if column in available_cols]
    frame = pd.read_csv(path, dtype={"code": str}, usecols=usecols, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    status = frame.get("financial_report_join_status", pd.Series("", index=frame.index)).fillna("").astype(str)
    count = pd.to_numeric(frame.get("financial_report_event_count", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    matched = frame[status.eq("event_window_matched") | count.gt(0)].copy()
    for column in [
        "return_20d",
        "prior_return_20d",
        "rsi14",
        "relative_strength_rank",
        "news_missing_rate",
        "financial_report_event_count",
        "financial_report_materiality_score",
        "financial_quality_risk_score",
        "financial_surprise_score",
        "financial_disclosure_quality_score",
    ]:
        if column in matched:
            matched[column] = pd.to_numeric(matched[column], errors="coerce")
    return matched.reset_index(drop=True)


def add_strata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    dates = pd.to_datetime(data["date"], errors="coerce")
    data["time_block"] = dates.map(_time_block)
    data["surprise_bucket"] = data["financial_surprise_score"].map(_surprise_bucket)
    data["quality_risk_bucket"] = data["financial_quality_risk_score"].map(_quality_risk_bucket)
    data["report_count_bucket"] = data["financial_report_event_count"].map(_count_bucket)
    data["overheat_flag"] = _overheat(data)
    data["weak_context_flag"] = _weak_context(data)
    data["negative_surprise_overheat_flag"] = (data["financial_surprise_score"] <= -0.4) & data["overheat_flag"]
    data["positive_surprise_weak_context_flag"] = (data["financial_surprise_score"] >= 0.4) & data["weak_context_flag"]
    data["quality_risk_high_flag"] = data["financial_quality_risk_score"] >= 0.6
    data["positive_surprise_low_risk_flag"] = (
        (data["financial_surprise_score"] >= 0.4)
        & (data["financial_quality_risk_score"] < 0.3)
        & (~data["overheat_flag"])
        & (~data["weak_context_flag"])
    )
    data["event_type_primary"] = data["financial_report_event_types"].fillna("").astype(str).map(_primary_event_type)
    return data


def build_stratified_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("time_block", ["time_block"]),
        ("surprise_bucket", ["surprise_bucket"]),
        ("quality_risk_bucket", ["quality_risk_bucket"]),
        ("event_type_primary", ["event_type_primary"]),
        ("time_block__surprise_bucket", ["time_block", "surprise_bucket"]),
        ("surprise__quality_risk", ["surprise_bucket", "quality_risk_bucket"]),
    ]
    rows: list[dict[str, Any]] = []
    for group_name, columns in groups:
        rows.extend(_group_metrics(frame, group_name, columns))
    return pd.DataFrame(rows)


def build_rule_candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "financial_negative_surprise_overheat_guard_v1": frame["negative_surprise_overheat_flag"],
        "financial_positive_surprise_weak_context_guard_v1": frame["positive_surprise_weak_context_flag"],
        "financial_quality_risk_high_guard_v1": frame["quality_risk_high_flag"],
        "financial_positive_surprise_low_risk_candidate_v1": frame["positive_surprise_low_risk_flag"],
        "financial_multi_report_review_v1": frame["financial_report_event_count"] >= 3,
    }
    rows = []
    baseline = _metrics(frame)
    for rule_id, selector in rules.items():
        subset = frame[selector.fillna(False)].copy()
        row = {"rule_id": rule_id, **_metrics(subset)}
        row["baseline_rows"] = baseline["rows"]
        row["baseline_avg_return_20d"] = baseline["avg_return_20d"]
        row["baseline_median_return_20d"] = baseline["median_return_20d"]
        row["baseline_positive_20d_rate"] = baseline["positive_20d_rate"]
        row["baseline_loss_20d_over_5_rate"] = baseline["loss_20d_over_5_rate"]
        row["delta_avg_return_20d"] = _delta(row["avg_return_20d"], baseline["avg_return_20d"])
        row["delta_median_return_20d"] = _delta(row["median_return_20d"], baseline["median_return_20d"])
        row["delta_positive_20d_rate"] = _delta(row["positive_20d_rate"], baseline["positive_20d_rate"])
        row["delta_loss_20d_over_5_rate"] = _delta(row["loss_20d_over_5_rate"], baseline["loss_20d_over_5_rate"])
        row["stock_concentration_note"] = _stock_concentration_note(row)
        row["status"] = _rule_status(rule_id, row)
        row["next_action"] = _rule_next_action(rule_id, row)
        rows.append(row)
    return pd.DataFrame(rows)


def build_financial_threshold_search(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = _metrics(frame)
    selectors = {
        "strict_quality_ge_0.6": frame["financial_quality_risk_score"] >= 0.6,
        "quality_ge_0.4": frame["financial_quality_risk_score"] >= 0.4,
        "quality_ge_0.3": frame["financial_quality_risk_score"] >= 0.3,
        "negative_surprise_le_-0.4": frame["financial_surprise_score"] <= -0.4,
        "mild_negative_surprise_le_-0.1": frame["financial_surprise_score"] <= -0.1,
        "not_positive_surprise_le_0.0": frame["financial_surprise_score"] <= 0.0,
        "nonpositive_surprise_news_available": (frame["financial_surprise_score"] <= 0.0) & (frame["news_missing_rate"] < 0.8),
        "not_strong_positive_surprise_news_available": (frame["financial_surprise_score"] <= 0.2) & (frame["news_missing_rate"] < 0.8),
        "negative_or_quality_ge_0.4": (frame["financial_surprise_score"] <= -0.1) | (frame["financial_quality_risk_score"] >= 0.4),
        "neutral_control_abs_surprise_le_0.2_low_risk": (frame["financial_surprise_score"].abs() <= 0.2) & (frame["financial_quality_risk_score"] < 0.3),
    }
    rows: list[dict[str, Any]] = []
    for rule_id, selector in selectors.items():
        subset = frame[selector.fillna(False)].copy()
        if subset.empty:
            continue
        row = {"rule_id": rule_id, **_metrics(subset)}
        row["time_blocks"] = int(subset["time_block"].nunique()) if "time_block" in subset else 0
        row["top_stock_share"] = _round(subset["code"].value_counts(normalize=True).iloc[0]) if "code" in subset and not subset.empty else None
        row["avg_news_missing_rate"] = _round(subset.get("news_missing_rate", pd.Series(dtype=float)).mean())
        row["delta_avg_return_20d"] = _delta(row["avg_return_20d"], baseline["avg_return_20d"])
        row["delta_positive_20d_rate"] = _delta(row["positive_20d_rate"], baseline["positive_20d_rate"])
        row["delta_loss_20d_over_5_rate"] = _delta(row["loss_20d_over_5_rate"], baseline["loss_20d_over_5_rate"])
        row["status"] = _threshold_status(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["status", "delta_positive_20d_rate", "delta_loss_20d_over_5_rate"], ascending=[True, True, False])


def build_candidate_samples(frame: pd.DataFrame, *, max_samples_per_rule: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        (
            "financial_negative_surprise_overheat_guard_v1",
            frame["negative_surprise_overheat_flag"],
            ["financial_surprise_score", "prior_return_20d", "rsi14"],
            [True, False, False],
            "负惊喜叠加过热，测试是否应降权。",
        ),
        (
            "financial_positive_surprise_weak_context_guard_v1",
            frame["positive_surprise_weak_context_flag"],
            ["financial_surprise_score", "financial_quality_risk_score", "news_missing_rate"],
            [False, False, False],
            "正惊喜但上下文弱，测试正惊喜是否不能单独升级。",
        ),
        (
            "financial_quality_risk_high_guard_v1",
            frame["quality_risk_high_flag"],
            ["financial_quality_risk_score", "financial_surprise_score"],
            [False, True],
            "财务质量风险高，测试是否应作为反证。",
        ),
        (
            "financial_positive_surprise_low_risk_candidate_v1",
            frame["positive_surprise_low_risk_flag"],
            ["financial_surprise_score", "relative_strength_rank"],
            [False, False],
            "正惊喜、低质量风险且上下文不弱，测试是否有正向潜力。",
        ),
        (
            "financial_multi_report_review_v1",
            frame["financial_report_event_count"] >= 3,
            ["financial_report_event_count", "financial_report_materiality_score"],
            [False, False],
            "多份财报/审计/季报同时出现，测试是否需要专门复核。",
        ),
    ]
    rows = []
    for rule_id, selector, sort_cols, ascending, reason in candidates:
        subset = frame[selector.fillna(False)].copy()
        if subset.empty:
            continue
        sort_cols = [col for col in sort_cols if col in subset]
        ascending = ascending[: len(sort_cols)]
        subset = _diversified_sample(
            subset,
            sort_cols=sort_cols,
            ascending=ascending,
            max_rows=max_samples_per_rule,
        )
        subset["candidate_rule"] = rule_id
        subset["reason_to_test"] = reason
        subset["sample_stock_concentration_note"] = _sample_stock_concentration_note(subset)
        rows.append(subset)
    sample_audit = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*DS_PLAN_COLUMNS, "return_20d"])
    ds_plan = sample_audit[[column for column in DS_PLAN_COLUMNS if column in sample_audit]].copy()
    return sample_audit, ds_plan


def build_diversified_guard_samples(
    frame: pd.DataFrame,
    *,
    target_rows: int = 24,
    max_per_stock: int = 2,
    max_per_block: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a safer next-DS sample plan with stock concentration controls.

    This plan is meant for testing `financial_risk_to_zero_guard_v1`.
    It intentionally includes neutral controls so the guard can be checked for
    over-defensiveness, and excludes future-return fields from the DS plan.
    """
    if frame.empty or target_rows <= 0:
        empty = pd.DataFrame(columns=[*DS_PLAN_COLUMNS, "return_20d"])
        return empty, empty[[column for column in DS_PLAN_COLUMNS if column in empty]]

    candidates = [
        (
            "financial_risk_to_zero_guard_v1",
            frame["quality_risk_high_flag"] | frame["negative_surprise_overheat_flag"],
            "高财务风险或负惊喜叠加过热，测试是否应把研究暴露压到0或信息不足不动作。",
            ["financial_quality_risk_score", "financial_surprise_score", "prior_return_20d", "rsi14"],
            [False, True, False, False],
        ),
        (
            "financial_negative_surprise_overheat_guard_v1",
            frame["negative_surprise_overheat_flag"],
            "负惊喜叠加过热，测试财报反证是否压过量价候选。",
            ["financial_surprise_score", "prior_return_20d", "rsi14"],
            [True, False, False],
        ),
        (
            "financial_quality_risk_high_guard_v1",
            frame["quality_risk_high_flag"],
            "财务质量风险高，测试是否应作为风险复核而不是正向证据。",
            ["financial_quality_risk_score", "financial_surprise_score"],
            [False, True],
        ),
        (
            "financial_positive_surprise_weak_context_guard_v1",
            frame["positive_surprise_weak_context_flag"],
            "正惊喜但新闻/同行/Book Skill上下文弱，测试正惊喜不能单独升级。",
            ["financial_surprise_score", "financial_quality_risk_score", "news_missing_rate"],
            [False, False, False],
        ),
        (
            "financial_multi_report_review_v1",
            frame["financial_report_event_count"] >= 3,
            "多报告/审计/问询信息密度高，测试其作为复核触发器是否有用。",
            ["financial_report_event_count", "financial_report_materiality_score"],
            [False, False],
        ),
        (
            "financial_nonpositive_surprise_news_available_v1",
            (frame["financial_surprise_score"] <= 0.0) & (frame["news_missing_rate"] < 0.8),
            "财报惊喜不强且普通新闻可用，测试新闻+财报是否能给出比单独财报更稳的复核判断。",
            ["financial_surprise_score", "news_missing_rate", "financial_disclosure_quality_score"],
            [True, True, False],
        ),
        (
            "financial_report_neutral_control_v1",
            _neutral_financial_selector(frame),
            "有时间安全财报事件但风险/惊喜不极端，用作guard过度防守的neutral control。",
            ["financial_disclosure_quality_score", "financial_report_materiality_score"],
            [False, False],
        ),
    ]
    per_rule_quota = max(1, math.ceil(target_rows / len(candidates)))
    selected_keys: set[tuple[str, str]] = set()
    code_counts: dict[str, int] = {}
    block_counts: dict[str, int] = {}
    selected_frames: list[pd.DataFrame] = []

    for rule_id, selector, reason, sort_cols, ascending in candidates:
        subset = frame[selector.fillna(False)].copy()
        if subset.empty:
            continue
        subset = _sort_for_sample(subset, sort_cols=sort_cols, ascending=ascending)
        picked = _pick_with_caps(
            subset,
            limit=per_rule_quota,
            selected_keys=selected_keys,
            code_counts=code_counts,
            block_counts=block_counts,
            max_per_stock=max_per_stock,
            max_per_block=max_per_block,
        )
        if picked.empty:
            continue
        picked["candidate_rule"] = rule_id
        picked["reason_to_test"] = reason
        selected_frames.append(picked)

    if selected_frames:
        sample = pd.concat(selected_frames, ignore_index=True)
    else:
        sample = pd.DataFrame(columns=frame.columns)

    if len(sample) < target_rows:
        all_candidates = []
        for rule_id, selector, reason, sort_cols, ascending in candidates:
            subset = frame[selector.fillna(False)].copy()
            if subset.empty:
                continue
            subset = _sort_for_sample(subset, sort_cols=sort_cols, ascending=ascending)
            subset["candidate_rule"] = rule_id
            subset["reason_to_test"] = reason
            all_candidates.append(subset)
        if all_candidates:
            fill_source = pd.concat(all_candidates, ignore_index=True)
            fill = _pick_with_caps(
                fill_source,
                limit=target_rows - len(sample),
                selected_keys=selected_keys,
                code_counts=code_counts,
                block_counts=block_counts,
                max_per_stock=max_per_stock,
                max_per_block=max_per_block,
            )
            if not fill.empty:
                sample = pd.concat([sample, fill], ignore_index=True)

    if not sample.empty:
        sample = _sort_diversified_plan(sample).head(target_rows).copy()
        sample["sample_stock_concentration_note"] = _diversified_concentration_note(sample, max_per_stock=max_per_stock, max_per_block=max_per_block)
    else:
        sample["sample_stock_concentration_note"] = ""
    sample_audit = sample.copy()
    ds_plan = sample_audit[[column for column in DS_PLAN_COLUMNS if column in sample_audit]].copy()
    return sample_audit, ds_plan


def write_summary(path: Path, frame: pd.DataFrame, metrics: pd.DataFrame, rules: pd.DataFrame, threshold_search: pd.DataFrame, ds_plan: pd.DataFrame, diversified_plan: pd.DataFrame) -> None:
    lines = [
        "# Financial Report Local Stratified Analysis",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Scope",
        "",
        f"- matched_rows: `{len(frame)}`",
        f"- unique_stocks: `{frame['code'].nunique() if not frame.empty else 0}`",
        f"- date_min: `{frame['date'].min() if not frame.empty else ''}`",
        f"- date_max: `{frame['date'].max() if not frame.empty else ''}`",
        "- selection_policy: 只分析 `financial_report_join_status=event_window_matched` 或事件数大于 0 的时间安全样本。",
        "- no_ds_called: 本层只做本地后验分层，不调用 DeepSeek。",
        "",
        "## Overall Matched Rows",
        "",
        _table(pd.DataFrame([_metrics(frame)])),
        "",
        "## Candidate Rule Metrics",
        "",
        _table(rules),
        "",
        "## Data-driven Findings",
        "",
        *_finding_lines(rules),
        "",
        "## Threshold Search",
        "",
        _table(threshold_search),
        "",
        "## Threshold Interpretation",
        "",
        *_threshold_finding_lines(threshold_search),
        "",
        "## Strongest Strata By Avg Return",
        "",
        _table(_top_metric_rows(metrics, best=True)),
        "",
        "## Weakest Strata By Avg Return",
        "",
        _table(_top_metric_rows(metrics, best=False)),
        "",
        "## Next DS Sample Plan",
        "",
        f"- planned_rows_without_future_return: `{len(ds_plan)}`",
        f"- diversified_guard_plan_rows_without_future_return: `{len(diversified_plan)}`",
        f"- diversified_guard_plan_unique_stocks: `{diversified_plan['code'].nunique() if not diversified_plan.empty and 'code' in diversified_plan else 0}`",
        "- `financial_report_next_ds_sample_plan.csv` 不包含 `return_20d`，避免下一层 Agent prompt 看到未来结果。",
        "- `financial_report_risk_guard_diversified_sample_plan.csv` 用于下一轮 `financial_risk_to_zero_guard_v1`，限制单股集中度，并加入 neutral control。",
        "- `financial_report_local_stratified_candidate_sample_audit.csv` 仅供离线审计，可包含后验收益。",
        "",
        "## Interpretation",
        "",
        "- 本地分层只用于发现候选阈值和失败模式，不能替代真实 Agent 决策 round。",
        "- 如果某条规则样本数小于 20，最多记为 observe，不得升级为默认策略。",
        "- 下一层验证应先 dry-run 检查样本计划，再用 DeepSeek Flash 小样本比较，不直接扩大到大规模。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_metrics(frame: pd.DataFrame, group_name: str, columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"group": group_name}
        row.update({column: value for column, value in zip(columns, keys)})
        row.update(_metrics(group))
        rows.append(row)
    return rows


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_20d", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "rows": int(len(frame)),
        "unique_stocks": int(frame["code"].nunique()) if "code" in frame and not frame.empty else 0,
        "avg_return_20d": _round(values.mean()) if not values.empty else None,
        "median_return_20d": _round(values.median()) if not values.empty else None,
        "positive_20d_rate": _round((values > 0).mean()) if not values.empty else None,
        "loss_20d_over_5_rate": _round((values < -5).mean()) if not values.empty else None,
        "std_return_20d": _round(values.std(ddof=0)) if not values.empty else None,
        "avg_financial_surprise_score": _round(frame.get("financial_surprise_score", pd.Series(dtype=float)).mean()) if not frame.empty else None,
        "avg_financial_quality_risk_score": _round(frame.get("financial_quality_risk_score", pd.Series(dtype=float)).mean()) if not frame.empty else None,
    }


def _rule_status(rule_id: str, row: dict[str, Any]) -> str:
    if row["rows"] < 20:
        return "observe_small_n"
    risk_worse = _risk_profile_worse(row)
    concentrated = row.get("unique_stocks", 0) < 5
    if concentrated and risk_worse:
        return "observe_concentrated_risk_guard"
    if concentrated:
        return "observe_concentrated"
    if "guard" in rule_id and risk_worse:
        return "observe_as_risk_guard"
    if "candidate" in rule_id and _positive_candidate_better(row):
        return "observe_candidate"
    return "observe"


def _rule_next_action(rule_id: str, row: dict[str, Any]) -> str:
    if row["rows"] < 20:
        return "样本不足，先不调用 DS 扩大；等待缓存扩展或并入人工复核。"
    if row.get("unique_stocks", 0) < 5:
        return "样本集中度过高，只能做跨日期复核；下一步优先扩缓存或用分层抽样补股票。"
    if "positive" in rule_id:
        return "抽样进入下一层小规模 Flash，验证是否仍无主动暴露。"
    return "作为反证候选进入下一层小规模 Flash，验证是否降低亏损。"


def _time_block(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    ts = pd.Timestamp(value)
    year = ts.year
    half = 1 if ts.month <= 6 else 2
    return f"H{year}_{half}"


def _surprise_bucket(value: Any) -> str:
    value = _number(value)
    if pd.isna(value):
        return "missing"
    if value <= -0.4:
        return "negative_le_-0.4"
    if value < -0.1:
        return "mild_negative"
    if value <= 0.1:
        return "neutral"
    if value < 0.4:
        return "mild_positive"
    return "positive_ge_0.4"


def _quality_risk_bucket(value: Any) -> str:
    value = _number(value)
    if pd.isna(value):
        return "missing"
    if value >= 0.6:
        return "high_ge_0.6"
    if value >= 0.3:
        return "medium_0.3_0.6"
    return "low_lt_0.3"


def _count_bucket(value: Any) -> str:
    value = _number(value)
    if pd.isna(value) or value <= 0:
        return "none"
    if value == 1:
        return "one"
    if value <= 3:
        return "two_to_three"
    return "four_plus"


def _primary_event_type(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return "missing"
    for candidate in ["financial_correction", "financial_inquiry", "performance_forecast", "performance_express", "audit_report", "annual_report", "semi_annual_metrics", "quarterly_metrics", "quarterly_report"]:
        if candidate in text:
            return candidate
    return text.split(";")[0]


def _overheat(frame: pd.DataFrame) -> pd.Series:
    prior = pd.to_numeric(frame.get("prior_return_20d", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    rsi = pd.to_numeric(frame.get("rsi14", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    return (prior >= 20) | (rsi >= 70)


def _weak_context(frame: pd.DataFrame) -> pd.Series:
    rel = pd.to_numeric(frame.get("relative_strength_rank", pd.Series(1, index=frame.index)), errors="coerce").fillna(1)
    news_missing = pd.to_numeric(frame.get("news_missing_rate", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    quality_risk = pd.to_numeric(frame.get("financial_quality_risk_score", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    skills = frame.get("triggered_skills", pd.Series("", index=frame.index)).fillna("").astype(str)
    return (rel <= 0.2) | (news_missing >= 0.8) | (quality_risk >= 0.4) | skills.eq("")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    if pd.isna(value) or pd.isna(baseline):
        return None
    return _round(float(value) - float(baseline))


def _round(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


def _top_metric_rows(metrics: pd.DataFrame, *, best: bool) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    data = metrics[metrics["rows"].fillna(0).astype(int) >= 20].copy()
    if data.empty:
        data = metrics.copy()
    return data.sort_values("avg_return_20d", ascending=not best).head(10)


def _finding_lines(rules: pd.DataFrame) -> list[str]:
    if rules.empty:
        return ["- 无候选规则数据。"]
    lines = []
    for row in rules.to_dict("records"):
        rule_id = row.get("rule_id", "")
        rows = row.get("rows", 0)
        stocks = row.get("unique_stocks", 0)
        pos_delta = row.get("delta_positive_20d_rate")
        loss_delta = row.get("delta_loss_20d_over_5_rate")
        status = row.get("status", "observe")
        if rule_id == "financial_positive_surprise_low_risk_candidate_v1" and rows == 0:
            lines.append("- `financial_positive_surprise_low_risk_candidate_v1` 在当前严格条件下样本为 0，不能把财报正惊喜作为独立正向信号。")
            continue
        if status == "observe_concentrated_risk_guard":
            lines.append(
                f"- `{rule_id}` 呈现风险特征但股票集中度过高：rows={int(rows)}, "
                f"unique_stocks={int(stocks)}, positive_rate_delta={_format_metric(pos_delta)}, "
                f"loss_rate_delta={_format_metric(loss_delta)}；下一步只能做小样本复核或扩缓存，不能直接固化为默认规则。"
            )
            continue
        if rule_id == "financial_multi_report_review_v1":
            lines.append(
                "- `financial_multi_report_review_v1` 覆盖较宽：rows={}, unique_stocks={}, "
                "positive_rate_delta={}, loss_rate_delta={}；目前更像复核触发器，不是单独决策规则。".format(
                    int(rows),
                    int(stocks),
                    _format_metric(pos_delta),
                    _format_metric(loss_delta),
                )
            )
    lines.append("- 总体结论：财报通道当前最适合做高可信复核、风险和不确定性通道；没有证据支持把它作为独立正向 alpha。")
    return lines


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _risk_profile_worse(row: dict[str, Any]) -> bool:
    positive_delta = row.get("delta_positive_20d_rate")
    loss_delta = row.get("delta_loss_20d_over_5_rate")
    median_delta = row.get("delta_median_return_20d")
    return (
        (positive_delta is not None and positive_delta <= -0.05)
        or (loss_delta is not None and loss_delta >= 0.05)
        or (median_delta is not None and median_delta <= -1.0)
    )


def _positive_candidate_better(row: dict[str, Any]) -> bool:
    positive_delta = row.get("delta_positive_20d_rate")
    loss_delta = row.get("delta_loss_20d_over_5_rate")
    avg_delta = row.get("delta_avg_return_20d")
    return (
        positive_delta is not None
        and positive_delta >= 0.03
        and (loss_delta is None or loss_delta <= 0.02)
        and (avg_delta is None or avg_delta >= 0)
    )


def _threshold_status(row: dict[str, Any]) -> str:
    if row.get("unique_stocks", 0) >= 20 and row.get("top_stock_share", 1) <= 0.1 and row.get("delta_positive_20d_rate", 0) < 0:
        return "broad_observe"
    if row.get("unique_stocks", 0) < 10 or row.get("top_stock_share", 0) > 0.2:
        return "concentrated_observe_only"
    if row.get("delta_positive_20d_rate", 0) >= 0:
        return "not_risk_signal"
    return "observe"


def _threshold_finding_lines(threshold_search: pd.DataFrame) -> list[str]:
    if threshold_search.empty:
        return ["- 无阈值扫描结果。"]
    lines = []
    strict = threshold_search[threshold_search["rule_id"].isin(["strict_quality_ge_0.6", "negative_surprise_le_-0.4"])]
    for row in strict.to_dict("records"):
        lines.append(
            f"- `{row['rule_id']}` 风险较强但集中：rows={int(row['rows'])}, "
            f"unique_stocks={int(row['unique_stocks'])}, top_stock_share={_format_metric(row.get('top_stock_share'))}；"
            "只能作为小样本护栏候选，不能原样扩成大规模正向规则。"
        )
    broad = threshold_search[threshold_search["rule_id"].isin(["nonpositive_surprise_news_available", "not_strong_positive_surprise_news_available"])]
    for row in broad.to_dict("records"):
        lines.append(
            f"- `{row['rule_id']}` 更分散：rows={int(row['rows'])}, unique_stocks={int(row['unique_stocks'])}, "
            f"time_blocks={int(row['time_blocks'])}, pos_delta={_format_metric(row.get('delta_positive_20d_rate'))}；"
            "适合作为新闻+财报通道样本，不适合作为跨全时期日期泛化证明。"
        )
    lines.append("- 下一层样本应同时包含严格风险、新闻可用宽风险和 neutral control，避免只在少数高风险股票上过拟合。")
    return lines


def _stock_concentration_note(row: dict[str, Any]) -> str:
    if row.get("rows", 0) >= 20 and row.get("unique_stocks", 0) < 5:
        return "concentrated_unique_stocks_lt_5"
    return ""


def _sample_stock_concentration_note(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    if frame["code"].nunique() < min(5, len(frame)):
        return "sample_concentrated_by_code"
    return ""


def _diversified_sample(
    frame: pd.DataFrame,
    *,
    sort_cols: list[str],
    ascending: list[bool],
    max_rows: int,
) -> pd.DataFrame:
    sorted_frame = frame.sort_values(sort_cols + ["date", "code"], ascending=ascending + [True, True])
    sorted_frame = sorted_frame.drop_duplicates(["date", "code"]).copy()
    selected: list[int] = []
    seen_codes: set[str] = set()
    seen_block_codes: set[tuple[str, str]] = set()

    for idx, row in sorted_frame.iterrows():
        if len(selected) >= max_rows:
            break
        code = str(row.get("code", ""))
        if code in seen_codes:
            continue
        selected.append(idx)
        seen_codes.add(code)
        seen_block_codes.add((str(row.get("time_block", "")), code))

    for idx, row in sorted_frame.iterrows():
        if len(selected) >= max_rows:
            break
        if idx in selected:
            continue
        code = str(row.get("code", ""))
        block_code = (str(row.get("time_block", "")), code)
        if block_code in seen_block_codes:
            continue
        selected.append(idx)
        seen_block_codes.add(block_code)

    for idx, _row in sorted_frame.iterrows():
        if len(selected) >= max_rows:
            break
        if idx in selected:
            continue
        selected.append(idx)

    return sorted_frame.loc[selected].reset_index(drop=True)


def _neutral_financial_selector(frame: pd.DataFrame) -> pd.Series:
    surprise = pd.to_numeric(frame.get("financial_surprise_score", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    risk = pd.to_numeric(frame.get("financial_quality_risk_score", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    disclosure = pd.to_numeric(frame.get("financial_disclosure_quality_score", pd.Series(0.5, index=frame.index)), errors="coerce").fillna(0)
    count = pd.to_numeric(frame.get("financial_report_event_count", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    return count.gt(0) & surprise.abs().le(0.2) & risk.lt(0.6) & disclosure.ge(0.5)


def _sort_for_sample(frame: pd.DataFrame, *, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    cols = [col for col in sort_cols if col in frame]
    asc = ascending[: len(cols)]
    return frame.sort_values(cols + ["time_block", "date", "code"], ascending=asc + [True, True, True]).drop_duplicates(["date", "code"]).copy()


def _pick_with_caps(
    frame: pd.DataFrame,
    *,
    limit: int,
    selected_keys: set[tuple[str, str]],
    code_counts: dict[str, int],
    block_counts: dict[str, int],
    max_per_stock: int,
    max_per_block: int,
) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return pd.DataFrame(columns=frame.columns)
    selected: list[int] = []
    seen_blocks: set[str] = set()

    def try_pick(require_new_block: bool) -> None:
        for idx, row in frame.iterrows():
            if len(selected) >= limit:
                return
            key = (str(row.get("date")), str(row.get("code")).zfill(6))
            code = key[1]
            block = str(row.get("time_block", ""))
            if key in selected_keys or code_counts.get(code, 0) >= max_per_stock:
                continue
            if max_per_block > 0 and block_counts.get(block, 0) >= max_per_block:
                continue
            if require_new_block and block in seen_blocks:
                continue
            selected.append(idx)
            selected_keys.add(key)
            code_counts[code] = code_counts.get(code, 0) + 1
            block_counts[block] = block_counts.get(block, 0) + 1
            seen_blocks.add(block)

    try_pick(require_new_block=True)
    try_pick(require_new_block=False)
    return frame.loc[selected].reset_index(drop=True)


def _sort_diversified_plan(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["_block_order"] = data["time_block"].map(_time_block_order)
    return data.sort_values(["_block_order", "candidate_rule", "date", "code"]).drop(columns=["_block_order"]).reset_index(drop=True)


def _time_block_order(block: Any) -> int:
    text = str(block)
    if text == "unknown":
        return 999
    try:
        year = int(text[1:5])
        half = int(text[-1])
        return year * 2 + half
    except (TypeError, ValueError):
        return 999


def _diversified_concentration_note(frame: pd.DataFrame, *, max_per_stock: int, max_per_block: int) -> str:
    if frame.empty or "code" not in frame:
        return ""
    unique = frame["code"].nunique()
    top_share = frame["code"].value_counts(normalize=True).iloc[0]
    notes = []
    if unique < 10:
        notes.append("unique_stocks_lt_10")
    if top_share > 0.2:
        notes.append("top_stock_share_gt_20pct")
    if frame["code"].value_counts().max() > max_per_stock:
        notes.append("max_per_stock_exceeded")
    if max_per_block > 0 and "time_block" in frame and frame["time_block"].value_counts().max() > max_per_block:
        notes.append("max_per_block_exceeded")
    return ";".join(notes) if notes else "diversified_ok"


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    return df.to_markdown(index=False)


if __name__ == "__main__":
    main()
