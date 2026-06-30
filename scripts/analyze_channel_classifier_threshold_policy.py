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


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_SCORED_DETAIL = DEFAULT_OUTPUT_DIR / "channel_rule_outcome_classifier_v1_scored_detail.csv"


FUTURE_NOTE = (
    "本报告使用 return_20d/pool_excess_20d 做离线阈值诊断；这些字段不得写回 Agent evidence。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline hard-counter probability policy diagnostics.")
    parser.add_argument("--scored-detail", default=str(DEFAULT_SCORED_DETAIL))
    parser.add_argument("--decision-ledger", default="")
    parser.add_argument("--audit-detail", default="")
    parser.add_argument("--output-prefix", default="channel_hard_counter_threshold_policy_v1")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_prefix(args.output_prefix)

    scored = load_scored_detail(Path(args.scored_detail))
    threshold_metrics = build_threshold_metrics(scored)
    block_metrics = build_block_metrics(scored)
    bin_metrics = build_probability_bin_metrics(scored)

    threshold_path = output_dir / f"{prefix}_threshold_metrics.csv"
    block_path = output_dir / f"{prefix}_block_metrics.csv"
    bin_path = output_dir / f"{prefix}_probability_bins.csv"
    threshold_metrics.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    block_metrics.to_csv(block_path, index=False, encoding="utf-8-sig")
    bin_metrics.to_csv(bin_path, index=False, encoding="utf-8-sig")

    paired = pd.DataFrame()
    if args.decision_ledger and args.audit_detail:
        paired = build_paired_decision_diagnostics(Path(args.decision_ledger), Path(args.audit_detail))
        paired.to_csv(output_dir / f"{prefix}_paired_decision_diagnostics.csv", index=False, encoding="utf-8-sig")

    write_report(
        output_dir / f"{prefix}.md",
        scored=scored,
        threshold_metrics=threshold_metrics,
        block_metrics=block_metrics,
        bin_metrics=bin_metrics,
        paired=paired,
        threshold_path=threshold_path,
        block_path=block_path,
        bin_path=bin_path,
    )

    print("A股研究Agent")
    print(f"scored_rows={len(scored)}")
    print(f"wrote: {output_dir / f'{prefix}.md'}")


def load_scored_detail(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {
        "date",
        "code",
        "valid_block",
        "return_20d",
        "pool_excess_20d",
        "logistic_channel_outcome__prob_hard_counter",
        "logistic_channel_outcome__prob_positive_support",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for col in [
        "return_20d",
        "pool_excess_20d",
        "logistic_channel_outcome__prob_hard_counter",
        "logistic_channel_outcome__prob_positive_support",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def build_threshold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = [0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    for threshold in thresholds:
        selected = frame[frame["logistic_channel_outcome__prob_hard_counter"] >= threshold]
        rows.append(_metric_row(selected, f"hard_prob_ge_{threshold:.2f}", threshold=threshold))
    return pd.DataFrame(rows)


def build_block_metrics(frame: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    selected = frame[frame["logistic_channel_outcome__prob_hard_counter"] >= threshold]
    rows = []
    for block, group in selected.groupby("valid_block", dropna=False, sort=True):
        rows.append(_metric_row(group, f"{block}_hard_prob_ge_{threshold:.2f}", threshold=threshold, valid_block=block))
    return pd.DataFrame(rows)


def build_probability_bin_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    prob = frame["logistic_channel_outcome__prob_hard_counter"]
    bins = [-0.001, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98, 1.001]
    labels = ["0-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-0.90", "0.90-0.95", "0.95-0.98", "0.98-1.00"]
    work = frame.copy()
    work["hard_prob_bin"] = pd.cut(prob, bins=bins, labels=labels, include_lowest=True)
    rows = []
    for bin_name, group in work.groupby("hard_prob_bin", dropna=False, sort=False):
        rows.append(_metric_row(group, str(bin_name), threshold=None))
    return pd.DataFrame(rows)


def build_paired_decision_diagnostics(decision_ledger: Path, audit_detail: Path) -> pd.DataFrame:
    rows = []
    for line in decision_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(
            {
                "variant": raw.get("variant"),
                "task_mode": raw.get("task_mode"),
                "valid_block": raw.get("valid_block"),
                "date": raw.get("decision_date"),
                "code": normalize_code(raw.get("code")),
                "name": raw.get("name"),
                "grade": raw.get("research_grade"),
                "action": raw.get("simulated_action"),
                "weight": pd.to_numeric(raw.get("simulated_weight_change"), errors="coerce"),
                "quant_tool_adoption_decision": raw.get("quant_tool_adoption_decision"),
                "quant_tool_override_reasons": raw.get("quant_tool_override_reasons"),
            }
        )
    decisions = pd.DataFrame(rows)
    if decisions.empty:
        return decisions

    audit = pd.read_csv(audit_detail, dtype={"code": str})
    audit["code"] = audit["code"].map(normalize_code)
    for col in ["return_20d", "pool_excess_20d", "hard_counter_probability", "positive_support_probability"]:
        if col in audit:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")

    variants = ["full_agent_with_hard_counter_tool", "full_agent_without_channel_classifier"]
    subset = decisions[decisions["variant"].isin(variants)].copy()
    wide = subset.pivot_table(
        index=["task_mode", "valid_block", "date", "code"],
        columns="variant",
        values=["grade", "action", "weight"],
        aggfunc="first",
    )
    wide.columns = [f"{field}__{variant}" for field, variant in wide.columns]
    wide = wide.reset_index()
    keep = [
        "date",
        "code",
        "valid_block",
        "name",
        "stratum",
        "rule_outcome_label",
        "hard_counter_probability",
        "positive_support_probability",
        "return_20d",
        "pool_excess_20d",
        "conflict_count",
    ]
    available = [col for col in keep if col in audit]
    wide = wide.merge(audit[available], on=["date", "code", "valid_block"], how="left")
    with_col = "weight__full_agent_with_hard_counter_tool"
    without_col = "weight__full_agent_without_channel_classifier"
    wide["weight_delta_with_minus_without"] = pd.to_numeric(wide.get(with_col), errors="coerce").fillna(0) - pd.to_numeric(wide.get(without_col), errors="coerce").fillna(0)
    wide["lowered_positive_posterior"] = (wide["weight_delta_with_minus_without"] < 0) & pd.to_numeric(wide.get("return_20d"), errors="coerce").gt(0)
    wide["lowered_negative_posterior"] = (wide["weight_delta_with_minus_without"] < 0) & pd.to_numeric(wide.get("return_20d"), errors="coerce").lt(0)
    wide["raised_positive_posterior"] = (wide["weight_delta_with_minus_without"] > 0) & pd.to_numeric(wide.get("return_20d"), errors="coerce").gt(0)
    wide["raised_negative_posterior"] = (wide["weight_delta_with_minus_without"] > 0) & pd.to_numeric(wide.get("return_20d"), errors="coerce").lt(0)
    return wide


def _metric_row(group: pd.DataFrame, name: str, *, threshold: float | None, valid_block: str | None = None) -> dict[str, Any]:
    returns = pd.to_numeric(group.get("return_20d", pd.Series(dtype=float)), errors="coerce")
    excess = pd.to_numeric(group.get("pool_excess_20d", pd.Series(dtype=float)), errors="coerce")
    hard_prob = pd.to_numeric(group.get("logistic_channel_outcome__prob_hard_counter", pd.Series(dtype=float)), errors="coerce")
    pos_prob = pd.to_numeric(group.get("logistic_channel_outcome__prob_positive_support", pd.Series(dtype=float)), errors="coerce")
    return {
        "segment": name,
        "valid_block": valid_block or "all",
        "hard_probability_threshold": threshold,
        "rows": int(len(group)),
        "unique_stocks": int(group["code"].nunique()) if "code" in group else 0,
        "coverage_dates": int(group["date"].nunique()) if "date" in group else 0,
        "positive_20d_rate": _mean_bool(returns.gt(0)),
        "loss_gt5_rate": _mean_bool(returns.le(-5)),
        "avg_return_20d": _round_mean(returns),
        "pool_excess_20d": _round_mean(excess),
        "mean_hard_probability": _round_mean(hard_prob),
        "mean_positive_support_probability": _round_mean(pos_prob),
        "research_only": True,
        "not_investment_instruction": True,
    }


def write_report(
    path: Path,
    *,
    scored: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    bin_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    threshold_path: Path,
    block_path: Path,
    bin_path: Path,
) -> None:
    q = scored["logistic_channel_outcome__prob_hard_counter"].quantile([0.50, 0.75, 0.90, 0.95, 0.98, 0.99])
    lines = [
        "# Channel Hard-Counter Threshold Policy v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Scope",
        "",
        f"- scored_rows: `{len(scored)}`",
        f"- valid_blocks: `{','.join(map(str, sorted(scored['valid_block'].dropna().unique())))}`",
        f"- future_label_boundary: {FUTURE_NOTE}",
        f"- threshold_metrics: `{threshold_path.relative_to(ROOT)}`",
        f"- block_metrics: `{block_path.relative_to(ROOT)}`",
        f"- probability_bins: `{bin_path.relative_to(ROOT)}`",
        "",
        "## Probability Quantiles",
        "",
        _table(q.reset_index().rename(columns={"index": "quantile", "logistic_channel_outcome__prob_hard_counter": "hard_counter_probability"})),
        "",
        "## Threshold Metrics",
        "",
        _table(threshold_metrics),
        "",
        "## Block Metrics At hard_prob>=0.80",
        "",
        _table(block_metrics),
        "",
        "## Probability Bins",
        "",
        _table(bin_metrics),
        "",
        "## Paired DS Decision Diagnostics",
        "",
    ]
    if paired.empty:
        lines.append("- paired_decision_diagnostics: `not_provided`")
    else:
        summary = paired.groupby("task_mode").agg(
            paired_rows=("code", "count"),
            lowered_negative=("lowered_negative_posterior", "sum"),
            lowered_positive=("lowered_positive_posterior", "sum"),
            raised_positive=("raised_positive_posterior", "sum"),
            raised_negative=("raised_negative_posterior", "sum"),
            mean_weight_delta=("weight_delta_with_minus_without", "mean"),
        ).reset_index()
        lines.extend([_table(summary), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- hard-counter 概率不能作为单一硬阈值。高概率段整体更像风险复核层，但不同时间块差异明显。",
            "- `hard_prob>=0.95` 更接近高亏损风险层；`0.80-0.95` 混有大量可反弹/soft-gap 样本，应作为黄色复核而不是直接归零。",
            "- 若进入 Agent evidence，应表达为 `risk_tier / required_confirmation / known_false_veto_risk`，而不是 `must_remove`。",
            "- 组合模式必须额外检查 missed-positive cost；单支模式可以更积极用作排雷复核，但仍不得单独改变最终研究分级。",
            "- 下一轮应训练 calibrated guard：hard-counter 只在缺少正向确认且 BookSkill/新闻/财报/同行/筹码冲突共同成立时升级为强反证。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_code(value: Any) -> str:
    text = str(value)
    if "." in text:
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def _round_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 6)


def _mean_bool(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.fillna(False).mean()), 6)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def safe_prefix(prefix: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in prefix.strip())
    return clean or "channel_hard_counter_threshold_policy_v1"


if __name__ == "__main__":
    main()
