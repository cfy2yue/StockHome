from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TOP_NS = (5, 10, 20)


def write_pool_selection_report(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for split in ["epoch2", "test"]:
        df = _read(output_dir / split / "ground_truth.csv")
        if df.empty:
            continue
        rows.extend(_evaluate_split(split, df))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "pool_selection_report.csv", index=False, encoding="utf-8-sig")
    (output_dir / "pool_selection_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _evaluate_split(split: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    base = df[df["gt_status"].astype(str) == "evaluated"].copy() if "gt_status" in df else df.copy()
    if base.empty:
        return []
    base["pool_score_book"] = pd.to_numeric(base.get("total_score"), errors="coerce").fillna(0)
    base["pool_score_momentum"] = pd.to_numeric(base.get("relative_strength_rank"), errors="coerce").fillna(0) * 10
    base["pool_score_composite"] = _composite_score(base)
    rows = [_strategy_metrics(split, "全候选池等权基线", "all candidates per decision date", base)]
    for top_n in TOP_NS:
        rows.append(_strategy_metrics(split, f"总分Top{top_n}", f"rank total_score top {top_n}", _top_n(base, "pool_score_book", top_n)))
        rows.append(_strategy_metrics(split, f"相对强弱Top{top_n}", f"rank relative_strength_rank top {top_n}", _top_n(base, "pool_score_momentum", top_n)))
        rows.append(_strategy_metrics(split, f"综合评分Top{top_n}", f"rank composite score top {top_n}", _top_n(base, "pool_score_composite", top_n)))
    return rows


def _composite_score(df: pd.DataFrame) -> pd.Series:
    total = pd.to_numeric(df.get("total_score"), errors="coerce").fillna(0)
    rs = pd.to_numeric(df.get("relative_strength_rank"), errors="coerce").fillna(0) * 10
    above = df.get("close_above_ma200", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1"]).astype(float) * 10
    peer = pd.to_numeric(df.get("peer_relative_to_group_20d"), errors="coerce").fillna(0).clip(-20, 20) / 4 + 5
    news_risk = pd.to_numeric(df.get("news_risk_event_score_30d"), errors="coerce").fillna(0).clip(0, 20)
    return total * 0.30 + rs * 0.25 + above * 0.15 + peer * 0.15 - news_risk * 0.15


def _top_n(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ordered = df.sort_values(["date", score_col, "code"], ascending=[True, False, True])
    return ordered.groupby("date", group_keys=False).head(top_n).copy()


def _strategy_metrics(split: str, name: str, formula: str, selected: pd.DataFrame) -> dict[str, Any]:
    daily = selected.groupby("date").agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
    values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
    if values.empty:
        avg = positive = std = loss = stability = None
    else:
        avg = round(float(values.mean()), 4)
        positive = round(float((values > 0).mean()), 4)
        std = round(float(values.std(ddof=0)), 4)
        loss = round(float((values <= -5).mean()), 4)
        stability = round(float(values.mean() - 0.5 * values.std(ddof=0) - 10 * (values <= -5).mean()), 4)
    return {
        "split": split,
        "strategy": name,
        "formula": formula,
        "decision_dates": int(len(daily)),
        "avg_selected_count": round(float(daily["selected_count"].mean()), 4) if not daily.empty else 0,
        "avg_return_20d": avg,
        "positive_20d_rate": positive,
        "std_return_20d": std,
        "loss_20d_over_5_rate": loss,
        "stability_score": stability,
    }


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 候选池筛选回测",
        "",
        "本报告评估多股票候选池内的等权筛选能力，只用于研究辅助，不构成买卖指令。",
        "",
        "| 数据集 | 策略 | 决策期数 | 平均入选数 | 20日均值 | 20日正收益率 | 20日波动 | 稳定性分 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if result.empty:
        lines.append("| NA | NA | 0 | 0 | NA | NA | NA | NA |")
        return "\n".join(lines)
    ordered = result.sort_values(["split", "avg_return_20d"], ascending=[True, False])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['split']} | {row['strategy']} | {int(row['decision_dates'])} | {_fmt(row['avg_selected_count'])} | "
            f"{_fmt(row['avg_return_20d'])} | {_fmt(row['positive_20d_rate'])} | {_fmt(row['std_return_20d'])} | {_fmt(row['stability_score'])} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- 每个决策日从候选池选 Top N，使用未来 20 个交易日等权平均收益验证。",
        "- `综合评分` = 总分、相对强弱、200DMA、同组相对强弱、新闻风险的预设组合，不用 test 调参。",
        "- 该报告回答“多股票候选池筛选”问题，不等同于单一股票深度研究。",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
