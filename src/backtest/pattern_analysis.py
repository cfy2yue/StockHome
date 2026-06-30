from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def write_pattern_report(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    epoch1 = _read(output_dir / "epoch1" / "ground_truth.csv")
    epoch2 = _read(output_dir / "epoch2" / "ground_truth.csv")
    test = _read(output_dir / "test" / "ground_truth.csv")
    skill_rows = []
    for skill_id in sorted(set(_skills(epoch1)) | set(_skills(epoch2)) | set(_skills(test))):
        row = {"strategy_id": skill_id}
        for label, df in [("train_epoch1", epoch1), ("train_epoch2", epoch2), ("test", test)]:
            stats = _skill_stats(df, skill_id)
            row.update({f"{label}_{k}": v for k, v in stats.items()})
        row["judgement"] = _judge(row)
        skill_rows.append(row)
    skill_df = pd.DataFrame(skill_rows)
    skill_df.to_csv(output_dir / "book_skill_pattern_stats.csv", index=False, encoding="utf-8-sig")

    sector = _sector_stats(epoch2, test)
    (output_dir / "pattern_report.md").write_text(_markdown(skill_df, sector), encoding="utf-8")
    return {"skills": skill_rows, "sector": sector}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _skills(df: pd.DataFrame) -> list[str]:
    if df.empty or "triggered_skills" not in df:
        return []
    out = []
    for value in df["triggered_skills"].fillna(""):
        out.extend([part for part in str(value).split(";") if part])
    return out


def _skill_stats(df: pd.DataFrame, skill_id: str) -> dict[str, Any]:
    if df.empty or "triggered_skills" not in df:
        return _empty()
    sub = df[df["triggered_skills"].fillna("").str.contains(skill_id, regex=False)].copy()
    if sub.empty:
        return _empty()
    return {
        "trigger_count": int(len(sub)),
        "gt_pass_rate": _mean(sub, "gt_pass"),
        "avg_return_5d": _mean(sub, "return_5d"),
        "avg_return_10d": _mean(sub, "return_10d"),
        "avg_return_20d": _mean(sub, "return_20d"),
        "positive_5d_rate": _positive_rate(sub, "return_5d"),
        "positive_20d_rate": _positive_rate(sub, "return_20d"),
    }


def _empty() -> dict[str, Any]:
    return {
        "trigger_count": 0,
        "gt_pass_rate": None,
        "avg_return_5d": None,
        "avg_return_10d": None,
        "avg_return_20d": None,
        "positive_5d_rate": None,
        "positive_20d_rate": None,
    }


def _mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _positive_rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float((values > 0).mean()), 4)


def _judge(row: dict[str, Any]) -> str:
    test_n = row.get("test_trigger_count", 0) or 0
    train_n = (row.get("train_epoch1_trigger_count", 0) or 0) + (row.get("train_epoch2_trigger_count", 0) or 0)
    test_r20 = row.get("test_avg_return_20d")
    train_r20 = row.get("train_epoch2_avg_return_20d")
    if train_n < 30:
        return "样本不足，暂不复用"
    if test_n < 10:
        return "test触发不足，需要扩大样本"
    if test_r20 is not None and train_r20 is not None and test_r20 * train_r20 < 0:
        return "test反证，暂不复用"
    if test_r20 is not None and abs(test_r20) >= 3:
        return "候选规律，可进入复核"
    return "方向较弱，仅作观察"


def _sector_stats(epoch2: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for label, df in [("train_epoch2", epoch2), ("test", test)]:
        if df.empty:
            continue
        grouped = {}
        for sector, sub in df.groupby("sector_group"):
            grouped[str(sector)] = {
                "count": int(len(sub)),
                "gt_pass_rate": _mean(sub, "gt_pass"),
                "avg_return_20d": _mean(sub, "return_20d"),
            }
        out[label] = grouped
    return out


def _markdown(skill_df: pd.DataFrame, sector: dict[str, Any]) -> str:
    lines = [
        "# 轻量回测规律分析报告",
        "",
        "本报告用于沉淀可复用研究规律，不构成买卖指令。",
        "",
        "## Book Skill 触发规律",
        "",
        "| 策略ID | test触发数 | test 20日均值 | test 20日正收益率 | 判断 |",
        "|---|---:|---:|---:|---|",
    ]
    if not skill_df.empty:
        ordered = skill_df.sort_values(["test_trigger_count", "train_epoch2_trigger_count"], ascending=False)
        for _, row in ordered.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {int(row.get('test_trigger_count') or 0)} | "
                f"{_fmt(row.get('test_avg_return_20d'))} | {_fmt(row.get('test_positive_20d_rate'))} | {row.get('judgement')} |"
            )
    lines += ["", "## 板块差异", ""]
    for label, data in sector.items():
        lines.append(f"### {label}")
        for sector_name, stats in data.items():
            lines.append(
                f"- {sector_name}: 样本 {stats['count']}，GT通过率 {_fmt(stats['gt_pass_rate'])}，20日均值 {_fmt(stats['avg_return_20d'])}"
            )
        lines.append("")
    lines += [
        "## 使用建议",
        "",
        "- 优先复核 test 中触发不少于 10 次且 20 日方向与训练集一致的规则。",
        "- 若训练集和 test 方向相反，归档为暂不复用，不进入判断模块。",
        "- 当前财务字段因缺少可靠披露日未参与评分，后续补齐后需重新回测。",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"

