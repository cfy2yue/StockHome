from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .news_vector import NEWS_VECTOR_DIMENSIONS


def write_news_feature_report(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for split in ["epoch2", "test"]:
        df = _read(output_dir / split / "ground_truth.csv")
        if df.empty:
            continue
        rows.extend(_split_rows(split, df))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "news_feature_report.csv", index=False, encoding="utf-8-sig")
    (output_dir / "news_feature_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _split_rows(split: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = len(df)
    for feature in NEWS_VECTOR_DIMENSIONS:
        if feature not in df:
            continue
        values = pd.to_numeric(df[feature], errors="coerce").fillna(0)
        active = df[values != 0].copy()
        inactive = df[values == 0].copy()
        rows.append(
            {
                "split": split,
                "feature": feature,
                "active_count": int(len(active)),
                "coverage_rate": round(len(active) / total, 4) if total else 0,
                "active_avg_return_20d": _mean(active, "return_20d"),
                "inactive_avg_return_20d": _mean(inactive, "return_20d"),
                "active_positive_20d_rate": _positive(active, "return_20d"),
                "inactive_positive_20d_rate": _positive(inactive, "return_20d"),
                "delta_avg_return_20d": _delta_mean(active, inactive, "return_20d"),
            }
        )
    return rows


def _values(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float(values.mean()), 4)


def _positive(df: pd.DataFrame, col: str) -> float | None:
    values = _values(df, col)
    return None if values.empty else round(float((values > 0).mean()), 4)


def _delta_mean(active: pd.DataFrame, inactive: pd.DataFrame, col: str) -> float | None:
    active_mean = _mean(active, col)
    inactive_mean = _mean(inactive, col)
    if active_mean is None or inactive_mean is None:
        return None
    return round(active_mean - inactive_mean, 4)


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 新闻向量特征报告",
        "",
        "本报告只用于研究辅助。新闻特征来自免费公开源缓存，并在回测时按决策日时间窗过滤。",
        "",
        "| 数据集 | 新闻维度 | 覆盖样本 | 覆盖率 | active 20日均值 | inactive 20日均值 | 差值 | active 正收益率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if result.empty:
        lines.append("| NA | NA | 0 | 0 | NA | NA | NA | NA |")
        return "\n".join(lines)
    top = result.sort_values(["split", "coverage_rate", "delta_avg_return_20d"], ascending=[True, False, False]).groupby("split").head(16)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['split']} | `{row['feature']}` | {int(row['active_count'])} | {_fmt(row['coverage_rate'])} | "
            f"{_fmt(row['active_avg_return_20d'])} | {_fmt(row['inactive_avg_return_20d'])} | {_fmt(row['delta_avg_return_20d'])} | "
            f"{_fmt(row['active_positive_20d_rate'])} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- 覆盖率低说明当前新闻缓存偏近期，不能把新闻维度的历史效果过度外推。",
        "- `active` 只是该新闻维度非零，不等于因果关系；需要更长历史新闻/公告回填后再作为正式 gate。",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
