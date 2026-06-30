from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_AVG_20D = 8.0
TARGET_POSITIVE_20D = 0.65
MIN_VALID_DATES = 20
MIN_TEST_DATES = 20
TOP_NS = (3, 5, 10, 20)


@dataclass(frozen=True)
class Formula:
    name: str
    weights: dict[str, float]
    note: str


@dataclass(frozen=True)
class DateGate:
    name: str
    formula: str
    apply: Any


FORMULAS = [
    Formula("质量趋势防守", {"total_score": 0.35, "relative_strength_rank": 0.25, "close_above_ma200": 0.20, "news_risk_event_score_30d": -0.20}, "总分+强弱+200DMA，扣新闻风险"),
    Formula("同组共振", {"relative_strength_rank": 0.25, "peer_group_positive_breadth_20d": 0.25, "peer_group_above_ma200_rate": 0.25, "news_risk_event_score_30d": -0.25}, "个股强弱和同组广度共振"),
    Formula("相对同组强势", {"peer_relative_to_group_20d": 0.40, "relative_strength_rank": 0.25, "total_score": 0.20, "news_risk_event_score_30d": -0.15}, "强于同组且新闻风险低"),
    Formula("低风险观察", {"counter_score": 0.35, "news_risk_event_score_30d": -0.35, "news_conflict_intensity_30d": -0.15, "atr20_pct": -0.15}, "反证少、新闻风险低、波动较低"),
    Formula("深跌反弹候选", {"drawdown60": -0.35, "prior_return_20d": -0.25, "counter_score": 0.20, "news_risk_event_score_30d": -0.20}, "60日深跌且20日弱势后的反弹候选"),
    Formula("深跌但同组回暖", {"drawdown60": -0.30, "prior_return_20d": -0.20, "peer_group_positive_breadth_20d": 0.25, "peer_group_above_ma200_rate": 0.15, "news_risk_event_score_30d": -0.10}, "个股深跌但同组广度改善"),
    Formula("公告风险回避", {"news_risk_event_score_30d": -0.45, "news_negative_materiality_30d": -0.25, "total_score": 0.20, "counter_score": 0.10}, "显式回避重大负面新闻/公告"),
    Formula("公告机会低风险", {"news_opportunity_event_score_30d": 0.25, "news_risk_event_score_30d": -0.35, "relative_strength_rank": 0.20, "close_above_ma200": 0.20}, "机会新闻必须配合低风险和趋势"),
    Formula("BookSkill保守", {"book_score": 0.30, "counter_score": 0.30, "trend_score": 0.20, "news_risk_event_score_30d": -0.20}, "BookSkill得分和反证得分共振"),
    Formula("低波动趋势", {"close_above_ma200": 0.30, "ma200_slope20": 0.25, "atr20_pct": -0.25, "news_risk_event_score_30d": -0.20}, "200DMA趋势但过滤高波动和新闻风险"),
]


def write_pool_optimizer_report(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train = _read(output_dir / "epoch2" / "ground_truth.csv")
    test = _read(output_dir / "test" / "ground_truth.csv")
    if train.empty or test.empty:
        result = pd.DataFrame()
        result.to_csv(output_dir / "pool_optimizer_report.csv", index=False, encoding="utf-8-sig")
        return result
    search, valid = _time_split(train)
    rows = []
    gates = _date_gates(search)
    for formula in FORMULAS:
        for top_n in TOP_NS:
            for gate in gates:
                rows.append(_row(formula, top_n, gate, search, valid, test))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["target_hit_on_valid"] = (result["valid_avg_return_20d"] >= TARGET_AVG_20D) & (
            result["valid_positive_20d_rate"] >= TARGET_POSITIVE_20D
        ) & (result["valid_decision_dates"] >= MIN_VALID_DATES)
        result["target_hit_on_test"] = result["target_hit_on_valid"] & (result["test_avg_return_20d"] >= TARGET_AVG_20D) & (
            result["test_positive_20d_rate"] >= TARGET_POSITIVE_20D
        ) & (result["test_decision_dates"] >= MIN_TEST_DATES)
        result["selection_score"] = (
            result["valid_avg_return_20d"].fillna(-999)
            + result["valid_positive_20d_rate"].fillna(0) * 8
            + result["valid_stability_score"].fillna(-999) * 0.3
            - result["valid_std_return_20d"].fillna(999) * 0.05
        )
        result.loc[result["valid_decision_dates"] < MIN_VALID_DATES, "selection_score"] = -9999
        result = result.sort_values(["target_hit_on_valid", "selection_score"], ascending=False)
    result.to_csv(output_dir / "pool_optimizer_report.csv", index=False, encoding="utf-8-sig")
    (output_dir / "pool_optimizer_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    dates = sorted(out["date"].dropna().unique())
    cutoff = dates[int(len(dates) * 0.65)] if len(dates) >= 4 else None
    if cutoff is None:
        return out, out.iloc[0:0].copy()
    return out[out["date"] <= cutoff].copy(), out[out["date"] > cutoff].copy()


def _date_gates(search: pd.DataFrame) -> list[DateGate]:
    features = _date_features(search)
    gates = [DateGate("全时段", "all decision dates", lambda frame: pd.Series(True, index=_date_features(frame).index))]
    specs = [
        ("候选池深跌", "pool_avg_prior_return_20d", "<=", 0.25),
        ("候选池强势", "pool_avg_prior_return_20d", ">=", 0.75),
        ("低新闻风险日", "pool_avg_news_risk", "<=", 0.35),
        ("高广度日", "pool_positive_breadth_20d", ">=", 0.65),
        ("低广度反弹日", "pool_positive_breadth_20d", "<=", 0.35),
        ("高分散度日", "pool_return_dispersion_20d", ">=", 0.70),
    ]
    for name, feature, op, quantile in specs:
        if feature not in features or features[feature].dropna().empty:
            continue
        threshold = float(features[feature].quantile(quantile))
        if op == "<=":
            gates.append(
                DateGate(name, f"{feature} <= {threshold:.4f}", lambda frame, f=feature, t=threshold: _date_features(frame)[f] <= t)
            )
        else:
            gates.append(
                DateGate(name, f"{feature} >= {threshold:.4f}", lambda frame, f=feature, t=threshold: _date_features(frame)[f] >= t)
            )
    return gates


def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="date"))
    out = df.copy()
    out["prior_return_20d_num"] = pd.to_numeric(out.get("prior_return_20d"), errors="coerce")
    out["news_risk_num"] = pd.to_numeric(out.get("news_risk_event_score_30d"), errors="coerce").fillna(0)
    grouped = out.groupby("date")
    features = grouped.agg(
        pool_avg_prior_return_20d=("prior_return_20d_num", "mean"),
        pool_positive_breadth_20d=("prior_return_20d_num", lambda value: float((value > 0).mean())),
        pool_return_dispersion_20d=("prior_return_20d_num", "std"),
        pool_avg_news_risk=("news_risk_num", "mean"),
    )
    return features.fillna(0)


def _row(formula: Formula, top_n: int, gate: DateGate, search: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    row = {
        "formula_name": formula.name,
        "top_n": top_n,
        "date_gate": gate.name,
        "date_gate_formula": gate.formula,
        "formula": _formula_text(formula),
        "note": formula.note,
    }
    for label, df in [("search", search), ("valid", valid), ("test", test)]:
        selected = _select(df, formula, top_n, gate)
        metrics = _metrics(selected)
        row.update({f"{label}_{key}": value for key, value in metrics.items()})
    return row


def _select(df: pd.DataFrame, formula: Formula, top_n: int, gate: DateGate) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    if out.empty:
        return out
    allowed_dates = set(gate.apply(out).loc[lambda value: value].index.astype(str))
    out = out[out["date"].astype(str).isin(allowed_dates)].copy()
    if out.empty:
        return out
    score = pd.Series(0.0, index=out.index)
    for feature, weight in formula.weights.items():
        score += _feature_score(out, feature) * weight
    out["optimized_pool_score"] = score
    ordered = out.sort_values(["date", "optimized_pool_score", "code"], ascending=[True, False, True])
    return ordered.groupby("date", group_keys=False).head(top_n).copy()


def _feature_score(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df:
        return pd.Series(0.0, index=df.index)
    if feature == "close_above_ma200":
        raw = df.get(feature, pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1"]).astype(float)
    else:
        raw = pd.to_numeric(df.get(feature), errors="coerce").fillna(0)
    ranked = raw.groupby(df["date"]).rank(pct=True, method="average")
    return ranked.fillna(0.5)


def _metrics(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"decision_dates": 0, "avg_selected_count": 0, "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    daily = selected.groupby("date").agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
    values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
    if values.empty:
        return {"decision_dates": int(len(daily)), "avg_selected_count": round(float(daily["selected_count"].mean()), 4), "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    avg = float(values.mean())
    return {
        "decision_dates": int(len(daily)),
        "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def _formula_text(formula: Formula) -> str:
    return " + ".join(f"{weight:+.2f}*rank({feature})" for feature, weight in formula.weights.items())


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# 候选池评分公式优化",
        "",
        "本报告只用于研究辅助。公式只在训练集前段搜索、训练集后段验证，test 只做锁定后检验。",
        "",
        "| 公式 | 环境Gate | TopN | valid期数 | valid20日均值 | valid正收益率 | valid稳定性 | test期数 | test20日均值 | test正收益率 | test稳定性 | 达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result.head(20).iterrows():
        hit = "是" if bool(row.get("target_hit_on_test")) else "否"
        lines.append(
            f"| {row['formula_name']} | {row['date_gate']} | {int(row['top_n'])} | {int(row['valid_decision_dates'])} | {_fmt(row['valid_avg_return_20d'])} | "
            f"{_fmt(row['valid_positive_20d_rate'])} | {_fmt(row['valid_stability_score'])} | {int(row['test_decision_dates'])} | "
            f"{_fmt(row['test_avg_return_20d'])} | {_fmt(row['test_positive_20d_rate'])} | {_fmt(row['test_stability_score'])} | {hit} |"
        )
    if not result.empty and not bool(result["target_hit_on_test"].any()):
        lines += [
            "",
            "## 结论",
            "",
            f"- 当前候选池公式没有在 valid 和 test 上同时达到 20日均值 {TARGET_AVG_20D}%、20日正收益率 {TARGET_POSITIVE_20D:.0%}。",
            "- 若 test 高但 valid 弱，只能作为反证复核线索，不能升级为正式策略。",
        ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
