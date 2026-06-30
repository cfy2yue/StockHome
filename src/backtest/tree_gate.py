from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FEATURES = [
    "total_score",
    "trend_score",
    "book_score",
    "counter_score",
    "prior_return_20d",
    "relative_strength_rank",
    "rsi14",
    "macd_hist",
    "volume_ratio20",
    "drawdown60",
    "ma200_slope20",
    "atr20_pct",
    "news_net_materiality_30d",
    "news_risk_event_score_30d",
    "news_opportunity_event_score_30d",
    "news_evidence_quality_score_30d",
    "news_conflict_intensity_30d",
    "peer_group_avg_return_20d",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "peer_group_above_ma200_rate",
    "peer_group_news_risk_avg",
    "peer_group_news_opportunity_avg",
]
TARGET_AVG_20D = 8.0
TARGET_POSITIVE_20D = 0.65
MIN_VALID_SAMPLES = 80
MIN_TEST_SAMPLES = 80


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def formula(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.4f}"


def write_tree_gate_report(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train = _read(output_dir / "epoch2" / "ground_truth.csv")
    test = _read(output_dir / "test" / "ground_truth.csv")
    if train.empty or test.empty:
        result = pd.DataFrame()
        result.to_csv(output_dir / "tree_gate_optimization.csv", index=False, encoding="utf-8-sig")
        return result

    search, valid = _time_split(train)
    rows = []
    for depth in [1, 2, 3]:
        conditions = _learn_rule(search, depth)
        rows.append(_row(f"tree_gate_depth{depth}", conditions, search, valid, test))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["target_hit_on_valid"] = (result["valid_avg_return_20d"] >= TARGET_AVG_20D) & (
            result["valid_positive_20d_rate"] >= TARGET_POSITIVE_20D
        ) & (result["valid_sample_count"] >= MIN_VALID_SAMPLES)
        result["target_hit_on_test"] = result["target_hit_on_valid"] & (result["test_avg_return_20d"] >= TARGET_AVG_20D) & (
            result["test_positive_20d_rate"] >= TARGET_POSITIVE_20D
        ) & (result["test_sample_count"] >= MIN_TEST_SAMPLES)
        result = result.sort_values(["target_hit_on_valid", "valid_stability_score"], ascending=False)
    result.to_csv(output_dir / "tree_gate_optimization.csv", index=False, encoding="utf-8-sig")
    (output_dir / "tree_gate_optimization.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    dates = sorted(out["date"].dropna().unique())
    if len(dates) < 4:
        return out, out.iloc[0:0].copy()
    cutoff = dates[int(len(dates) * 0.65)]
    return out[out["date"] <= cutoff].copy(), out[out["date"] > cutoff].copy()


def _learn_rule(df: pd.DataFrame, max_depth: int) -> list[Condition]:
    conditions: list[Condition] = []
    current = df.copy()
    min_samples = max(80, int(len(df) * 0.04))
    for _ in range(max_depth):
        candidate = _best_condition(current, min_samples)
        if candidate is None:
            break
        next_current = current[_mask(current, [candidate])]
        if len(next_current) < min_samples:
            break
        conditions.append(candidate)
        current = next_current
    return conditions


def _best_condition(df: pd.DataFrame, min_samples: int) -> Condition | None:
    best: tuple[float, Condition] | None = None
    for feature in FEATURES:
        if feature not in df:
            continue
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.nunique() < 3:
            continue
        for threshold in values.quantile([0.2, 0.35, 0.5, 0.65, 0.8]).dropna().unique():
            for op in [">=", "<="]:
                condition = Condition(feature, op, float(threshold))
                subset = df[_mask(df, [condition])]
                if len(subset) < min_samples:
                    continue
                score = _selection_score(subset)
                if best is None or score > best[0]:
                    best = (score, condition)
    return None if best is None else best[1]


def _mask(df: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        values = pd.to_numeric(df.get(condition.feature), errors="coerce")
        if condition.op == ">=":
            mask &= values >= condition.threshold
        else:
            mask &= values <= condition.threshold
    return mask.fillna(False)


def _row(name: str, conditions: list[Condition], search: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    formula = " and ".join(condition.formula() for condition in conditions) or "all"
    row = {"gate_name": name, "formula": formula, "depth": len(conditions)}
    for prefix, df in [("search", search), ("valid", valid), ("test", test)]:
        metrics = _metrics(df[_mask(df, conditions)])
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return row


def _metrics(df: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(df.get("return_20d"), errors="coerce").dropna()
    if values.empty:
        return {
            "sample_count": 0,
            "avg_return_20d": None,
            "positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "stability_score": None,
        }
    loss_rate = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    avg = float(values.mean())
    return {
        "sample_count": int(len(values)),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss_rate, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss_rate, 4),
    }


def _selection_score(df: pd.DataFrame) -> float:
    metrics = _metrics(df)
    if metrics["avg_return_20d"] is None:
        return -9999
    return (
        float(metrics["avg_return_20d"])
        + float(metrics["positive_20d_rate"]) * 10
        + float(metrics["stability_score"]) * 0.3
        - abs(float(metrics["std_return_20d"])) * 0.05
    )


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# Tree Gate 优化",
        "",
        "本报告只用于研究辅助。浅层 tree gate 只在训练集前段学习分叉阈值，训练集后段验证，test 只做锁定后检验。",
        "",
        "| Gate | 公式 | search样本 | search20日均值 | valid样本 | valid20日均值 | valid正收益率 | test样本 | test20日均值 | test正收益率 | test稳定性 | 验证+test达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result.iterrows():
        hit = "是" if bool(row.get("target_hit_on_test")) else "否"
        lines.append(
            f"| {row['gate_name']} | `{row['formula']}` | {int(row['search_sample_count'])} | {_fmt(row['search_avg_return_20d'])} | "
            f"{int(row['valid_sample_count'])} | {_fmt(row['valid_avg_return_20d'])} | {_fmt(row['valid_positive_20d_rate'])} | "
            f"{int(row['test_sample_count'])} | {_fmt(row['test_avg_return_20d'])} | {_fmt(row['test_positive_20d_rate'])} | "
            f"{_fmt(row['test_stability_score'])} | {hit} |"
        )
    if not result.empty and not bool(result["target_hit_on_test"].any()):
        lines += [
            "",
            "## 结论",
            "",
            f"- 当前 tree gate 未在锁定 test 上同时达到 20日均值 {TARGET_AVG_20D}% 和 20日正收益率 {TARGET_POSITIVE_20D:.0%}。",
            "- 这些阈值只能作为下一轮大样本验证的候选分叉，不应直接升级为正式规则。",
        ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
