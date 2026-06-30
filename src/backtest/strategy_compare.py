from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ORIGINAL_NAME = "原始BookSkill评分"
ADAPTED_NAME = "优化后量化共振"


def write_strategy_comparison(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for split in ["epoch1", "epoch2", "test"]:
        df = _read(output_dir / split / "ground_truth.csv")
        if df.empty:
            continue
        rows.append(_metrics(split, ORIGINAL_NAME, _original_mask(df), df))
        rows.append(_metrics(split, ADAPTED_NAME, _adapted_mask(df), df))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "strategy_comparison.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _original_mask(df: pd.DataFrame) -> pd.Series:
    return df["rating"].isin(["放入观察", "继续深挖"])


def _adapted_mask(df: pd.DataFrame) -> pd.Series:
    skills = df["triggered_skills"].fillna("")
    return (
        skills.str.contains("PPS-Q-017", regex=False)
        & skills.str.contains("PPS-Q-019", regex=False)
        & skills.str.contains("DOW-B-017", regex=False)
        & (pd.to_numeric(df.get("prior_return_20d"), errors="coerce") > 0)
        & (pd.to_numeric(df.get("relative_strength_rank"), errors="coerce") >= 0.67)
        & (df.get("close_above_ma200").astype(str).str.lower().isin(["true", "1"]))
    )


def _metrics(split: str, strategy: str, mask: pd.Series, df: pd.DataFrame) -> dict[str, Any]:
    sub = df[mask].copy()
    return {
        "split": split,
        "strategy": strategy,
        "sample_count": int(len(sub)),
        "avg_return_5d": _mean(sub, "return_5d"),
        "avg_return_10d": _mean(sub, "return_10d"),
        "avg_return_20d": _mean(sub, "return_20d"),
        "positive_5d_rate": _positive(sub, "return_5d"),
        "positive_20d_rate": _positive(sub, "return_20d"),
        "std_return_20d": _std(sub, "return_20d"),
        "loss_20d_over_5_rate": _loss_rate(sub, "return_20d", -5),
        "stability_score": _stability(sub),
    }


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float(values.mean()), 4)


def _std(df: pd.DataFrame, col: str) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float(values.std(ddof=0)), 4)


def _positive(df: pd.DataFrame, col: str) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float((values > 0).mean()), 4)


def _loss_rate(df: pd.DataFrame, col: str, threshold: float) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float((values <= threshold).mean()), 4)


def _stability(df: pd.DataFrame) -> float | None:
    values = _values(df, "return_20d")
    if values.empty:
        return None
    # 越高越稳：20日均值减去一半波动，再惩罚大亏比例。
    return round(float(values.mean() - 0.5 * values.std(ddof=0) - 10 * (values <= -5).mean()), 4)


def _values(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def comparison_summary(df: pd.DataFrame, split: str = "test") -> dict[str, Any]:
    target = df[df["split"] == split]
    if target.empty:
        return {}
    original = target[target["strategy"] == ORIGINAL_NAME].head(1)
    adapted = target[target["strategy"] == ADAPTED_NAME].head(1)
    if original.empty or adapted.empty:
        return {}
    o = original.iloc[0]
    a = adapted.iloc[0]
    return {
        "split": split,
        "sample_count_delta": int(a["sample_count"]) - int(o["sample_count"]),
        "avg_return_20d_delta": _delta(a, o, "avg_return_20d"),
        "positive_20d_rate_delta": _delta(a, o, "positive_20d_rate"),
        "std_return_20d_delta": _delta(a, o, "std_return_20d"),
        "loss_20d_over_5_rate_delta": _delta(a, o, "loss_20d_over_5_rate"),
        "stability_score_delta": _delta(a, o, "stability_score"),
    }


def _delta(a: pd.Series, o: pd.Series, col: str) -> float | None:
    if pd.isna(a.get(col)) or pd.isna(o.get(col)):
        return None
    return round(float(a[col]) - float(o[col]), 4)


def _markdown(df: pd.DataFrame) -> str:
    lines = [
        "# 原始策略 vs 优化后量化策略对比",
        "",
        "本对比只用于研究辅助，不构成买卖指令。",
        "",
        "定义：",
        "",
        "- 原始BookSkill评分：当前系统给出“放入观察/继续深挖”的样本。",
        "- 优化后量化共振：同时触发 `PPS-Q-017`、`PPS-Q-019`、`DOW-B-017`，且 20 日相对强弱排名 >= 0.67、处于 200DMA 上方。",
        "",
        "| 数据集 | 策略 | 样本数 | 5日均值 | 20日均值 | 20日正收益率 | 20日波动 | 20日<-5%比例 | 稳定性分 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['split']} | {row['strategy']} | {int(row['sample_count'])} | {_fmt(row['avg_return_5d'])} | "
            f"{_fmt(row['avg_return_20d'])} | {_fmt(row['positive_20d_rate'])} | {_fmt(row['std_return_20d'])} | "
            f"{_fmt(row['loss_20d_over_5_rate'])} | {_fmt(row['stability_score'])} |"
        )
    summary = comparison_summary(df, "test")
    if summary:
        lines += [
            "",
            "## Test 集差异",
            "",
            f"- 20日均值变化：{_fmt(summary['avg_return_20d_delta'])}",
            f"- 20日正收益率变化：{_fmt(summary['positive_20d_rate_delta'])}",
            f"- 20日波动变化：{_fmt(summary['std_return_20d_delta'])}",
            f"- 20日跌幅超过5%的比例变化：{_fmt(summary['loss_20d_over_5_rate_delta'])}",
            f"- 稳定性分变化：{_fmt(summary['stability_score_delta'])}",
        ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"

