from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .pool_optimizer import _fmt, _metrics, _select
from .rebound_diagnostics import (
    REBOUND_FORMULAS,
    TARGET_AVG_20D,
    TARGET_POSITIVE_20D,
    _evaluate,
    _read,
    _rebound_gates,
)


MIN_FOLDS = 4
MIN_TEST_DATES = 20


def write_rebound_validation_report(output_dir: str | Path, folds: int = 4) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train, train_source = _read_train(output_dir)
    test = _read(output_dir / "test" / "ground_truth.csv")
    if train.empty or test.empty:
        result = pd.DataFrame()
        _write(output_dir, result)
        return result

    diagnostics = _evaluate(train, folds)
    if diagnostics.empty:
        result = pd.DataFrame()
        _write(output_dir, result)
        return result

    family = _family_summary(diagnostics)
    if family.empty:
        result = pd.DataFrame()
        _write(output_dir, result)
        return result
    selected_families = _select_families(family)
    if not selected_families:
        result = pd.DataFrame()
        _write(output_dir, result)
        return result

    baseline = _load_baselines(output_dir, test)
    rows = []
    for selector_name, selector_rule, selected_family in selected_families:
        formula = _find_formula(str(selected_family["formula_name"]))
        gate = _find_gate(train, str(selected_family["date_gate"]))
        if formula is None or gate is None:
            continue
        selected_test = _select(test, formula, int(selected_family["top_n"]), gate)
        test_metrics = _metrics(selected_test)
        row = {
            "selector_name": selector_name,
            "selector_rule": selector_rule,
            "train_source": train_source,
            "formula_name": selected_family["formula_name"],
            "date_gate": selected_family["date_gate"],
            "top_n": int(selected_family["top_n"]),
            "locked_gate_formula": gate.formula,
            "train_folds": int(selected_family["folds"]),
            "train_hit_rate": round(float(selected_family["hit_rate"]), 4),
            "train_avg_return_20d": round(float(selected_family["avg_return_20d"]), 4),
            "train_positive_20d_rate": round(float(selected_family["positive_20d_rate"]), 4),
            "train_stability_score": round(float(selected_family["stability_score"]), 4),
            "train_avg_decision_dates": round(float(selected_family["avg_decision_dates"]), 4),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **baseline,
        }
        row["test_target_hit"] = (
            _safe(row.get("test_avg_return_20d")) >= TARGET_AVG_20D
            and _safe(row.get("test_positive_20d_rate")) >= TARGET_POSITIVE_20D
            and int(row.get("test_decision_dates") or 0) >= MIN_TEST_DATES
        )
        row["test_vs_pool_avg_return_delta"] = _delta(row.get("test_avg_return_20d"), row.get("pool_baseline_avg_return_20d"))
        row["test_vs_pool_positive_delta"] = _delta(row.get("test_positive_20d_rate"), row.get("pool_baseline_positive_20d_rate"))
        row["test_vs_pool_stability_delta"] = _delta(row.get("test_stability_score"), row.get("pool_baseline_stability_score"))
        row["test_vs_rolling_hold_avg_return_delta"] = _delta(row.get("test_avg_return_20d"), row.get("rolling_hold_avg_return_20d"))
        row["test_vs_rolling_hold_positive_delta"] = _delta(row.get("test_positive_20d_rate"), row.get("rolling_hold_positive_20d_rate"))
        row["test_vs_rolling_hold_stability_delta"] = _delta(row.get("test_stability_score"), row.get("rolling_hold_stability_score"))
        row["promotion_candidate"] = (
            row["test_target_hit"]
            and _safe(row.get("train_avg_return_20d")) >= TARGET_AVG_20D
            and _safe(row.get("train_positive_20d_rate")) >= TARGET_POSITIVE_20D
            and _safe(row.get("test_vs_pool_avg_return_delta")) > 0
            and _safe(row.get("test_vs_pool_stability_delta")) > 0
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    _write(output_dir, result)
    return result


def _write(output_dir: Path, result: pd.DataFrame) -> None:
    result.to_csv(output_dir / "rebound_validation.csv", index=False, encoding="utf-8-sig")
    (output_dir / "rebound_validation.md").write_text(_markdown(result), encoding="utf-8")


def _read_train(output_dir: Path) -> tuple[pd.DataFrame, str]:
    for split in ["epoch2", "epoch1"]:
        df = _read(output_dir / split / "ground_truth.csv")
        if not df.empty:
            return df, split
    return pd.DataFrame(), ""


def _family_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        diagnostics.groupby(["formula_name", "date_gate", "top_n"])
        .agg(
            folds=("fold", "count"),
            hit_rate=("oos_target_hit", "mean"),
            avg_return_20d=("oos_avg_return_20d", "mean"),
            positive_20d_rate=("oos_positive_20d_rate", "mean"),
            stability_score=("oos_stability_score", "mean"),
            avg_decision_dates=("oos_decision_dates", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["folds"] >= MIN_FOLDS].copy()
    if summary.empty:
        return pd.DataFrame()
    summary["selection_score"] = (
        summary["avg_return_20d"].fillna(-999)
        + summary["positive_20d_rate"].fillna(0) * 8
        + summary["stability_score"].fillna(-999) * 0.3
        + summary["hit_rate"].fillna(0) * 6
    )
    return summary.sort_values(["hit_rate", "selection_score", "avg_decision_dates"], ascending=False)


def _select_families(summary: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    selectors = [
        (
            "hit_rate_first",
            "训练fold达标率优先，其次综合得分",
            summary,
            ["hit_rate", "selection_score", "avg_decision_dates"],
            [False, False, False],
        ),
        (
            "strict_target_margin",
            "训练均值>=8且正收益率>=65%，综合得分优先",
            summary[(summary["avg_return_20d"] >= TARGET_AVG_20D) & (summary["positive_20d_rate"] >= TARGET_POSITIVE_20D)],
            ["selection_score", "hit_rate", "avg_decision_dates"],
            [False, False, False],
        ),
        (
            "trend_protected_target",
            "趋势未破坏策略族中，训练均值>=8且正收益率>=65%",
            summary[
                (summary["formula_name"] == "反弹:趋势未破坏")
                & (summary["avg_return_20d"] >= TARGET_AVG_20D)
                & (summary["positive_20d_rate"] >= TARGET_POSITIVE_20D)
            ],
            ["avg_return_20d", "positive_20d_rate", "stability_score"],
            [False, False, False],
        ),
        (
            "peer_rebound_stability",
            "同组回暖策略族中，训练均值>=7.5且正收益率>=75%",
            summary[
                (summary["formula_name"] == "反弹:同组回暖小组")
                & (summary["avg_return_20d"] >= 7.5)
                & (summary["positive_20d_rate"] >= 0.75)
            ],
            ["stability_score", "positive_20d_rate", "avg_return_20d"],
            [False, False, False],
        ),
    ]
    rows: list[tuple[str, str, pd.Series]] = []
    seen: set[tuple[str, str, int]] = set()
    for name, rule, frame, by, ascending in selectors:
        if frame.empty:
            continue
        selected = frame.sort_values(by, ascending=ascending).iloc[0]
        key = (str(selected["formula_name"]), str(selected["date_gate"]), int(selected["top_n"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append((name, rule, selected))
    return rows


def _find_formula(name: str):
    return next((formula for formula in REBOUND_FORMULAS if formula.name == name), None)


def _find_gate(train: pd.DataFrame, name: str):
    return next((gate for gate in _rebound_gates(train) if gate.name == name), None)


def _load_baselines(output_dir: Path, test: pd.DataFrame) -> dict[str, Any]:
    pool = _read(output_dir / "pool_selection_report.csv")
    hold = _read(output_dir / "baseline_comparison.csv")
    result: dict[str, Any] = {
        "pool_baseline_avg_return_20d": None,
        "pool_baseline_positive_20d_rate": None,
        "pool_baseline_stability_score": None,
        "rolling_hold_avg_return_20d": None,
        "rolling_hold_positive_20d_rate": None,
        "rolling_hold_stability_score": None,
    }
    if not pool.empty:
        base = pool[(pool["split"] == "test") & (pool["strategy"] == "全候选池等权基线")].head(1)
        if not base.empty:
            row = base.iloc[0]
            result.update(
                {
                    "pool_baseline_avg_return_20d": row.get("avg_return_20d"),
                    "pool_baseline_positive_20d_rate": row.get("positive_20d_rate"),
                    "pool_baseline_stability_score": row.get("stability_score"),
                }
            )
    if not hold.empty:
        base = hold[(hold["split"] == "test") & (hold["gate_name"] == "20日滚动长期持有基线")].head(1)
        if not base.empty:
            row = base.iloc[0]
            result.update(
                {
                    "rolling_hold_avg_return_20d": row.get("avg_return_20d"),
                    "rolling_hold_positive_20d_rate": row.get("positive_20d_rate"),
                    "rolling_hold_stability_score": row.get("stability_score"),
                }
            )
    if result["rolling_hold_avg_return_20d"] is None and not test.empty:
        values = pd.to_numeric(test.get("return_20d"), errors="coerce").dropna()
        if not values.empty:
            loss = float((values <= -5).mean())
            std = float(values.std(ddof=0))
            avg = float(values.mean())
            result.update(
                {
                    "rolling_hold_avg_return_20d": round(avg, 4),
                    "rolling_hold_positive_20d_rate": round(float((values > 0).mean()), 4),
                    "rolling_hold_stability_score": round(avg - 0.5 * std - 10 * loss, 4),
                }
            )
    return result


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    return round(float(left) - float(right), 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 反弹策略锁定 Test 验证",
        "",
        "本报告只用训练集 walk-forward 结果选择策略族，然后锁定公式和环境 gate，在 test 集运行一次。它用于检查反弹策略是否真正跑过基线，不构成任何交易指令。",
        "",
    ]
    if result.empty:
        lines.append("- 尚无足够数据完成锁定验证。")
        return "\n".join(lines)
    row = result.iloc[0]
    hit = "是" if bool(row.get("test_target_hit")) else "否"
    lines += [
        "## 锁定规则",
        "",
        "| 选择器 | 训练来源 | 公式 | Gate | TopN | 训练均值 | 训练正收益率 | Test均值 | Test正收益率 | Test稳定性 | 相对全池稳定性 | 达标 | 候选升级 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in result.iterrows():
        hit = "是" if bool(row.get("test_target_hit")) else "否"
        promote = "是" if bool(row.get("promotion_candidate")) else "否"
        lines.append(
            f"| {row.get('selector_name')} | {row.get('train_source')} | {row.get('formula_name')} | {row.get('date_gate')} | {int(row.get('top_n'))} | "
            f"{_fmt(row.get('train_avg_return_20d'))} | {_fmt(row.get('train_positive_20d_rate'))} | "
            f"{_fmt(row.get('test_avg_return_20d'))} | {_fmt(row.get('test_positive_20d_rate'))} | "
            f"{_fmt(row.get('test_stability_score'))} | {_fmt(row.get('test_vs_pool_stability_delta'))} | {hit} | {promote} |"
        )
    best = result.sort_values(["promotion_candidate", "test_target_hit", "test_avg_return_20d"], ascending=False).iloc[0]
    hit = "是" if bool(best.get("test_target_hit")) else "否"
    lines += [
        "",
        "## 最强锁定结果",
        "",
        "| 决策期数 | 平均入选数 | 20日均值 | 20日正收益率 | 20日波动 | 20日-5%比例 | 稳定性分 | 达标 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| {int(best.get('test_decision_dates') or 0)} | {_fmt(best.get('test_avg_selected_count'))} | {_fmt(best.get('test_avg_return_20d'))} | "
        f"{_fmt(best.get('test_positive_20d_rate'))} | {_fmt(best.get('test_std_return_20d'))} | {_fmt(best.get('test_loss_20d_over_5_rate'))} | "
        f"{_fmt(best.get('test_stability_score'))} | {hit} |",
        "",
        "## 相对基线",
        "",
        f"- 相对全候选池等权基线：20日均值变化 {_fmt(best.get('test_vs_pool_avg_return_delta'))}，正收益率变化 {_fmt(best.get('test_vs_pool_positive_delta'))}，稳定性变化 {_fmt(best.get('test_vs_pool_stability_delta'))}。",
        f"- 相对20日滚动长期持有基线：20日均值变化 {_fmt(best.get('test_vs_rolling_hold_avg_return_delta'))}，正收益率变化 {_fmt(best.get('test_vs_rolling_hold_positive_delta'))}，稳定性变化 {_fmt(best.get('test_vs_rolling_hold_stability_delta'))}。",
    ]
    target_count = int(result.get("test_target_hit", pd.Series(dtype=bool)).astype(bool).sum())
    if bool(result.get("promotion_candidate", pd.Series(dtype=bool)).any()):
        lines.append("- 至少一个训练预定义选择器在锁定 test 上达到目标并跑过基线，可进入扩大股票池/更长历史复核。")
    elif target_count:
        lines.append("- 有训练预定义选择器在锁定 test 上达到目标，但训练侧均值或升级门槛不足，先作为扩大复核候选，不升级为正式候选池策略。")
    else:
        lines.append("- 尚无训练预定义选择器同时满足 test 目标和基线改进，不升级为正式候选池策略。")
    return "\n".join(lines)
