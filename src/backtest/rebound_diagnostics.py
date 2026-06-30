from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .pool_optimizer import DateGate, Formula, _date_features, _fmt, _metrics, _select


TARGET_AVG_20D = 8.0
TARGET_POSITIVE_20D = 0.65
TOP_NS = (3, 5, 10)

REBOUND_FORMULAS = [
    Formula(
        "反弹:同组回暖小组",
        {
            "drawdown60": -0.30,
            "prior_return_20d": -0.20,
            "peer_group_positive_breadth_20d": 0.25,
            "peer_group_above_ma200_rate": 0.15,
            "news_risk_event_score_30d": -0.10,
        },
        "深跌 + 同组广度修复，用于观察超跌但行业环境改善的候选池。",
    ),
    Formula(
        "反弹:低波动修复",
        {
            "drawdown60": -0.25,
            "prior_return_20d": -0.20,
            "atr20_pct": -0.25,
            "counter_score": 0.20,
            "news_risk_event_score_30d": -0.10,
        },
        "深跌 + 低波动 + 反证少，用于区分修复和继续失控下跌。",
    ),
    Formula(
        "反弹:低新闻风险",
        {
            "drawdown60": -0.25,
            "prior_return_20d": -0.20,
            "news_risk_event_score_30d": -0.35,
            "counter_score": 0.20,
        },
        "深跌但规避新闻/公告风险，用于检验新闻通道是否能降低回撤。",
    ),
    Formula(
        "反弹:趋势未破坏",
        {
            "drawdown60": -0.20,
            "prior_return_20d": -0.20,
            "close_above_ma200": 0.25,
            "ma200_slope20": 0.20,
            "news_risk_event_score_30d": -0.15,
        },
        "回撤但长期趋势未破坏，用于观察强趋势中的短期回撤机会。",
    ),
]


def write_rebound_diagnostics_report(output_dir: str | Path, folds: int = 4) -> pd.DataFrame:
    output_dir = Path(output_dir)
    df = _read(output_dir / "epoch2" / "ground_truth.csv")
    if df.empty:
        result = pd.DataFrame()
        result.to_csv(output_dir / "rebound_diagnostics.csv", index=False, encoding="utf-8-sig")
        (output_dir / "rebound_diagnostics.md").write_text(_markdown(result), encoding="utf-8")
        return result
    result = _evaluate(df, folds)
    result.to_csv(output_dir / "rebound_diagnostics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "rebound_diagnostics.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _evaluate(df: pd.DataFrame, folds: int) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    dates = sorted(out["date"].dropna().unique())
    if len(dates) < 80:
        return pd.DataFrame()
    start_idx = max(80, int(len(dates) * 0.40))
    chunks = _split_dates(dates[start_idx:], folds)
    rows = []
    for fold_idx, chunk in enumerate(chunks, 1):
        if len(chunk) < 10:
            continue
        train_df = out[out["date"] < chunk[0]].copy()
        oos_df = out[out["date"].isin(chunk)].copy()
        if train_df.empty or oos_df.empty:
            continue
        for formula in REBOUND_FORMULAS:
            for top_n in TOP_NS:
                for gate in _rebound_gates(train_df):
                    selected = _select(oos_df, formula, top_n, gate)
                    metrics = _metrics(selected)
                    rows.append(
                        {
                            "fold": fold_idx,
                            "oos_start": pd.Timestamp(chunk[0]).date().isoformat(),
                            "oos_end": pd.Timestamp(chunk[-1]).date().isoformat(),
                            "formula_name": formula.name,
                            "formula_note": formula.note,
                            "top_n": top_n,
                            "date_gate": gate.name,
                            "date_gate_formula": gate.formula,
                            **{f"oos_{key}": value for key, value in metrics.items()},
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["oos_target_hit"] = (result["oos_avg_return_20d"] >= TARGET_AVG_20D) & (
        result["oos_positive_20d_rate"] >= TARGET_POSITIVE_20D
    )
    return result.sort_values(["fold", "oos_target_hit", "oos_avg_return_20d"], ascending=[True, False, False])


def _rebound_gates(train_df: pd.DataFrame) -> list[DateGate]:
    features = _date_features(train_df)
    gates = [DateGate("全时段", "all decision dates", lambda frame: pd.Series(True, index=_date_features(frame).index))]
    if features.empty:
        return gates

    prior_threshold = _quantile(features, "pool_avg_prior_return_20d", 0.25)
    breadth_threshold = _quantile(features, "pool_positive_breadth_20d", 0.35)
    news_threshold = _quantile(features, "pool_avg_news_risk", 0.35)
    dispersion_threshold = _quantile(features, "pool_return_dispersion_20d", 0.60)

    if prior_threshold is not None:
        gates.append(
            DateGate(
                "候选池深跌日",
                f"pool_avg_prior_return_20d <= {prior_threshold:.4f}",
                lambda frame, t=prior_threshold: _date_features(frame)["pool_avg_prior_return_20d"] <= t,
            )
        )
        if dispersion_threshold is not None:
            gates.append(
                DateGate(
                    "深跌且低分散",
                    f"pool_avg_prior_return_20d <= {prior_threshold:.4f} and pool_return_dispersion_20d <= {dispersion_threshold:.4f}",
                    lambda frame, t=prior_threshold, d=dispersion_threshold: (
                        _date_features(frame)["pool_avg_prior_return_20d"] <= t
                    )
                    & (_date_features(frame)["pool_return_dispersion_20d"] <= d),
                )
            )
    if breadth_threshold is not None:
        gates.append(
            DateGate(
                "低广度反弹日",
                f"pool_positive_breadth_20d <= {breadth_threshold:.4f}",
                lambda frame, t=breadth_threshold: _date_features(frame)["pool_positive_breadth_20d"] <= t,
            )
        )
        if news_threshold is not None:
            gates.append(
                DateGate(
                    "低广度且低新闻风险",
                    f"pool_positive_breadth_20d <= {breadth_threshold:.4f} and pool_avg_news_risk <= {news_threshold:.4f}",
                    lambda frame, t=breadth_threshold, n=news_threshold: (
                        _date_features(frame)["pool_positive_breadth_20d"] <= t
                    )
                    & (_date_features(frame)["pool_avg_news_risk"] <= n),
                )
            )
    return gates


def _quantile(features: pd.DataFrame, column: str, q: float) -> float | None:
    if column not in features:
        return None
    values = pd.to_numeric(features[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.quantile(q))


def _split_dates(dates: list[Any], folds: int) -> list[list[Any]]:
    if not dates:
        return []
    buckets = pd.Series(range(len(dates))) * folds // max(1, len(dates))
    return [list(chunk) for _, chunk in pd.Series(dates).groupby(buckets)]


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 反弹型候选池诊断",
        "",
        "本报告专门比较深跌/低广度反弹策略族在不同 OOS fold 的表现，只用于研究辅助，不构成任何交易指令。",
        "",
        "| Fold | 公式 | Gate | TopN | OOS期数 | OOS均值 | OOS正收益率 | OOS稳定性 | 达标 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    if result.empty:
        lines.append("| 0 | NA | NA | 0 | 0 | NA | NA | NA | 否 |")
        return "\n".join(lines)

    for _, row in result.groupby("fold", group_keys=False).head(5).iterrows():
        hit = "是" if bool(row.get("oos_target_hit")) else "否"
        lines.append(
            f"| {int(row['fold'])} | {row['formula_name']} | {row['date_gate']} | {int(row['top_n'])} | "
            f"{int(row['oos_decision_dates'])} | {_fmt(row.get('oos_avg_return_20d'))} | "
            f"{_fmt(row.get('oos_positive_20d_rate'))} | {_fmt(row.get('oos_stability_score'))} | {hit} |"
        )

    summary = (
        result.groupby(["formula_name", "date_gate", "top_n"])
        .agg(
            folds=("fold", "count"),
            hit_rate=("oos_target_hit", "mean"),
            avg_return=("oos_avg_return_20d", "mean"),
            positive_rate=("oos_positive_20d_rate", "mean"),
            stability=("oos_stability_score", "mean"),
        )
        .reset_index()
        .sort_values(["hit_rate", "avg_return"], ascending=False)
        .head(8)
    )
    lines += [
        "",
        "## 策略族汇总",
        "",
        "| 公式 | Gate | TopN | fold数 | 达标率 | 平均20日均值 | 平均正收益率 | 平均稳定性 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['formula_name']} | {row['date_gate']} | {int(row['top_n'])} | {int(row['folds'])} | "
            f"{_fmt(row['hit_rate'])} | {_fmt(row['avg_return'])} | {_fmt(row['positive_rate'])} | {_fmt(row['stability'])} |"
        )
    lines += [
        "",
        "## 复用限制",
        "",
        "- 只有在多个时间 fold 同时稳定达标时，才可升级为正式候选池规则。",
        "- 若仅某一 fold 达标，先记录为市场状态线索，不能直接用于后续判断。",
        "- 新闻风险特征必须继续保持时间安全：只使用决策日 15:00 前已经可见的信息。",
    ]
    return "\n".join(lines)
