from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .pool_optimizer import (
    FORMULAS,
    MIN_VALID_DATES,
    TARGET_AVG_20D,
    TARGET_POSITIVE_20D,
    TOP_NS,
    _date_gates,
    _fmt,
    _metrics,
    _row,
    _select,
    _time_split,
)


def write_pool_walkforward_report(output_dir: str | Path, folds: int = 4) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train = _read(output_dir / "epoch2" / "ground_truth.csv")
    if train.empty:
        result = pd.DataFrame()
        result.to_csv(output_dir / "pool_walkforward_report.csv", index=False, encoding="utf-8-sig")
        return result
    result = _walk_forward(train, folds=folds)
    result.to_csv(output_dir / "pool_walkforward_report.csv", index=False, encoding="utf-8-sig")
    (output_dir / "pool_walkforward_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _walk_forward(df: pd.DataFrame, folds: int) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    dates = sorted(out["date"].dropna().unique())
    if len(dates) < 80:
        return pd.DataFrame()
    start_idx = max(80, int(len(dates) * 0.40))
    fold_dates = _split_dates(dates[start_idx:], folds)
    rows = []
    for fold_idx, dates_chunk in enumerate(fold_dates, 1):
        if len(dates_chunk) < 10:
            continue
        train_df = out[out["date"] < dates_chunk[0]].copy()
        oos_df = out[out["date"].isin(dates_chunk)].copy()
        if train_df.empty or oos_df.empty:
            continue
        selected = _select_best_candidate(train_df, oos_df)
        if not selected:
            continue
        selected.update(
            {
                "fold": fold_idx,
                "oos_start": pd.Timestamp(dates_chunk[0]).date().isoformat(),
                "oos_end": pd.Timestamp(dates_chunk[-1]).date().isoformat(),
                "train_decision_dates": int(train_df["date"].nunique()),
            }
        )
        rows.append(selected)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["oos_target_hit"] = (result["oos_avg_return_20d"] >= TARGET_AVG_20D) & (
        result["oos_positive_20d_rate"] >= TARGET_POSITIVE_20D
    )
    return result


def _split_dates(dates: list[Any], folds: int) -> list[list[Any]]:
    return [list(chunk) for chunk in pd.Series(dates).groupby(pd.Series(range(len(dates))) * folds // max(1, len(dates))).apply(list)]


def _select_best_candidate(train_df: pd.DataFrame, oos_df: pd.DataFrame) -> dict[str, Any] | None:
    inner_search, inner_valid = _time_split(train_df)
    if inner_search.empty or inner_valid.empty:
        return None
    candidates = []
    for formula in FORMULAS:
        for top_n in TOP_NS:
            for gate in _date_gates(inner_search):
                row = _row(formula, top_n, gate, inner_search, inner_valid, oos_df)
                row["selection_score"] = _selection_score(row)
                candidates.append((row, formula, top_n, gate))
    candidates = [item for item in candidates if item[0]["selection_score"] > -9999]
    if not candidates:
        return None
    row, formula, top_n, gate = max(candidates, key=lambda item: item[0]["selection_score"])
    selected = _select(oos_df, formula, top_n, gate)
    oos_metrics = _metrics(selected)
    return {
        "formula_name": formula.name,
        "top_n": top_n,
        "date_gate": gate.name,
        "date_gate_formula": gate.formula,
        "inner_valid_decision_dates": row.get("valid_decision_dates"),
        "inner_valid_avg_return_20d": row.get("valid_avg_return_20d"),
        "inner_valid_positive_20d_rate": row.get("valid_positive_20d_rate"),
        "inner_valid_stability_score": row.get("valid_stability_score"),
        **{f"oos_{key}": value for key, value in oos_metrics.items()},
    }


def _selection_score(row: dict[str, Any]) -> float:
    if int(row.get("valid_decision_dates") or 0) < MIN_VALID_DATES:
        return -9999
    avg = _safe(row.get("valid_avg_return_20d"), -999)
    positive = _safe(row.get("valid_positive_20d_rate"), 0)
    stability = _safe(row.get("valid_stability_score"), -999)
    std = _safe(row.get("valid_std_return_20d"), 999)
    return avg + positive * 8 + stability * 0.3 - std * 0.05


def _safe(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 候选池滚动 Walk-Forward 验证",
        "",
        "本报告只用于研究辅助。每个 fold 只使用之前历史选择公式和环境 gate，再在下一时间段做 out-of-sample 检验。",
        "",
        "| Fold | OOS区间 | 公式 | Gate | TopN | inner valid均值 | inner valid正收益率 | OOS均值 | OOS正收益率 | OOS稳定性 | 达标 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if result.empty:
        lines.append("| 0 | NA | NA | NA | 0 | NA | NA | NA | NA | NA | 否 |")
        return "\n".join(lines)
    for _, row in result.iterrows():
        hit = "是" if bool(row.get("oos_target_hit")) else "否"
        lines.append(
            f"| {int(row['fold'])} | {row['oos_start']}~{row['oos_end']} | {row['formula_name']} | {row['date_gate']} | {int(row['top_n'])} | "
            f"{_fmt(row.get('inner_valid_avg_return_20d'))} | {_fmt(row.get('inner_valid_positive_20d_rate'))} | "
            f"{_fmt(row.get('oos_avg_return_20d'))} | {_fmt(row.get('oos_positive_20d_rate'))} | {_fmt(row.get('oos_stability_score'))} | {hit} |"
        )
    avg = result["oos_avg_return_20d"].mean()
    positive = result["oos_positive_20d_rate"].mean()
    stability = result["oos_stability_score"].mean()
    lines += [
        "",
        "## 汇总",
        "",
        f"- OOS 平均 20日均值：{_fmt(avg)}",
        f"- OOS 平均 20日正收益率：{_fmt(positive)}",
        f"- OOS 平均稳定性分：{_fmt(stability)}",
        f"- 达标 fold 数：{int(result['oos_target_hit'].sum())}/{len(result)}",
    ]
    if not bool(result["oos_target_hit"].all()):
        lines.append("- 未在所有滚动 fold 中达标，不应升级为正式候选池策略。")
    return "\n".join(lines)
