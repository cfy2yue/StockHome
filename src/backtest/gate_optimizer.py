from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


TARGET_AVG_20D = 8.0
TARGET_POSITIVE_20D = 0.65
MIN_TRAIN_SAMPLES = 80
MIN_VALID_SAMPLES = 30
MIN_TEST_SAMPLES = 20


@dataclass(frozen=True)
class Gate:
    name: str
    formula: str
    func: Callable[[pd.DataFrame], pd.Series]


def write_gate_optimization_report(output_dir: str | Path, data_dir: str | Path | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train = _read(output_dir / "epoch2" / "ground_truth.csv")
    test = _read(output_dir / "test" / "ground_truth.csv")
    if train.empty or test.empty:
        result = pd.DataFrame()
        result.to_csv(output_dir / "gate_optimization.csv", index=False, encoding="utf-8-sig")
        return result

    search, valid = _time_split(train)
    gates = _candidate_gates()
    rows = []
    for gate in gates:
        search_metrics = _metrics("search", gate, search)
        valid_metrics = _metrics("valid", gate, valid)
        test_metrics = _metrics("test", gate, test)
        rows.append({**search_metrics, **_prefix(valid_metrics, "valid_"), **_prefix(test_metrics, "test_")})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["target_hit_on_valid"] = (
            (result["avg_return_20d"] >= TARGET_AVG_20D)
            & (result["positive_20d_rate"] >= TARGET_POSITIVE_20D)
            & (result["sample_count"] >= MIN_TRAIN_SAMPLES)
            & (result["valid_avg_return_20d"] >= TARGET_AVG_20D)
            & (result["valid_positive_20d_rate"] >= TARGET_POSITIVE_20D)
            & (result["valid_sample_count"] >= MIN_VALID_SAMPLES)
        )
        result["target_hit_on_test"] = result["target_hit_on_valid"] & (result["test_avg_return_20d"] >= TARGET_AVG_20D) & (
            result["test_positive_20d_rate"] >= TARGET_POSITIVE_20D
        ) & (result["test_sample_count"] >= MIN_TEST_SAMPLES)
        result["selection_score"] = result["valid_avg_return_20d"].fillna(-999) + result["valid_positive_20d_rate"].fillna(0) * 10 + result[
            "valid_stability_score"
        ].fillna(-999) * 0.2
        result = result.sort_values(["target_hit_on_valid", "selection_score"], ascending=False)
    result.to_csv(output_dir / "gate_optimization.csv", index=False, encoding="utf-8-sig")

    baseline = _baseline_rows(output_dir, data_dir)
    baseline.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "gate_optimization.md").write_text(_markdown(result, baseline), encoding="utf-8")
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


def _candidate_gates() -> list[Gate]:
    gates: list[Gate] = [Gate("原始放入观察以上", "rating in ['放入观察','继续深挖']", lambda df: df["rating"].isin(["放入观察", "继续深挖"]))]
    for threshold in [0.5, 0.6, 0.67, 0.75, 0.8, 0.9]:
        gates.append(
            Gate(
                f"相对强弱>={threshold}",
                f"relative_strength_rank >= {threshold}",
                lambda df, t=threshold: _num(df, "relative_strength_rank") >= t,
            )
        )
    for threshold in [0, 5, 10, 15, 20]:
        gates.append(
            Gate(
                f"20日动量>={threshold}",
                f"prior_return_20d >= {threshold}",
                lambda df, t=threshold: _num(df, "prior_return_20d") >= t,
            )
        )
    for threshold in [5.0, 5.2, 5.4]:
        gates.append(Gate(f"总分>={threshold}", f"total_score >= {threshold}", lambda df, t=threshold: _num(df, "total_score") >= t))
    for threshold in [7, 8]:
        gates.append(Gate(f"趋势分>={threshold}", f"trend_score >= {threshold}", lambda df, t=threshold: _num(df, "trend_score") >= t))
    for skill in ["PPS-Q-017", "PPS-Q-019", "DOW-B-017", "PPS-Q-009", "DOW-B-004", "CANDLE_MACRO_002"]:
        gates.append(Gate(f"触发{skill}", f"triggered_skills contains {skill}", lambda df, s=skill: _skills(df).str.contains(s, regex=False)))
    gates += [
        Gate("无30日新闻事件", "news_count_30d == 0", lambda df: _num(df, "news_count_30d").fillna(0) == 0),
        Gate("低新闻风险", "news_risk_event_score_30d <= 0", lambda df: _num(df, "news_risk_event_score_30d").fillna(0) <= 0),
        Gate("正新闻净重大性", "news_net_materiality_30d > 0", lambda df: _num(df, "news_net_materiality_30d").fillna(0) > 0),
        Gate("新闻冲突低", "news_conflict_intensity_30d <= 0", lambda df: _num(df, "news_conflict_intensity_30d").fillna(0) <= 0),
        Gate("强于同组20日", "peer_relative_to_group_20d > 0", lambda df: _num(df, "peer_relative_to_group_20d").fillna(0) > 0),
        Gate("同组广度>=60%", "peer_group_positive_breadth_20d >= 0.6", lambda df: _num(df, "peer_group_positive_breadth_20d").fillna(0) >= 0.6),
        Gate("同组站上200DMA>=60%", "peer_group_above_ma200_rate >= 0.6", lambda df: _num(df, "peer_group_above_ma200_rate").fillna(0) >= 0.6),
        Gate("同组新闻风险低", "peer_group_news_risk_avg <= 0", lambda df: _num(df, "peer_group_news_risk_avg").fillna(0) <= 0),
        Gate("站上200DMA", "close_above_ma200 == true", lambda df: df["close_above_ma200"].astype(str).str.lower().isin(["true", "1"])),
        Gate(
            "共振:200DMA+强弱前1/3+DOW",
            "close_above_ma200 and relative_strength_rank>=0.67 and PPS-Q-017/PPS-Q-019/DOW-B-017",
            lambda df: (_num(df, "relative_strength_rank") >= 0.67)
            & df["close_above_ma200"].astype(str).str.lower().isin(["true", "1"])
            & _skills(df).str.contains("PPS-Q-017", regex=False)
            & _skills(df).str.contains("PPS-Q-019", regex=False)
            & _skills(df).str.contains("DOW-B-017", regex=False),
        ),
        Gate(
            "强动量低反证",
            "prior_return_20d>=10 and relative_strength_rank>=0.67 and counter_score>=7",
            lambda df: (_num(df, "prior_return_20d") >= 10) & (_num(df, "relative_strength_rank") >= 0.67) & (_num(df, "counter_score") >= 7),
        ),
        Gate(
            "强趋势排除回撤观察区",
            "trend_score>=8 and not DOW-B-004",
            lambda df: (_num(df, "trend_score") >= 8) & (~_skills(df).str.contains("DOW-B-004", regex=False)),
        ),
        Gate(
            "科创强势门控",
            "sector_group==star_technology and relative_strength_rank>=0.67 and close_above_ma200",
            lambda df: (df["sector_group"].astype(str) == "star_technology")
            & (_num(df, "relative_strength_rank") >= 0.67)
            & df["close_above_ma200"].astype(str).str.lower().isin(["true", "1"]),
        ),
        Gate(
            "个股强于同组且新闻风险低",
            "peer_relative_to_group_20d>0 and news_risk_event_score_30d<=0",
            lambda df: (_num(df, "peer_relative_to_group_20d").fillna(0) > 0) & (_num(df, "news_risk_event_score_30d").fillna(0) <= 0),
        ),
    ]
    return gates


def _metrics(split: str, gate: Gate, df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        sub = df
    else:
        mask = gate.func(df).fillna(False)
        sub = df[mask].copy()
    return {
        "split": split,
        "gate_name": gate.name,
        "formula": gate.formula,
        "sample_count": int(len(sub)),
        "avg_return_20d": _mean(sub, "return_20d"),
        "positive_20d_rate": _positive(sub, "return_20d"),
        "std_return_20d": _std(sub, "return_20d"),
        "loss_20d_over_5_rate": _loss_rate(sub, "return_20d", -5),
        "stability_score": _stability(sub),
    }


def _baseline_rows(output_dir: Path, data_dir: str | Path | None) -> pd.DataFrame:
    rows = []
    codes = set()
    for split in ["epoch2", "test"]:
        df = _read(output_dir / split / "ground_truth.csv")
        if df.empty:
            continue
        if "code" in df:
            codes.update(df["code"].astype(str).str.zfill(6).unique())
        all_gate = Gate("20日滚动长期持有基线", "all decision windows", lambda frame: pd.Series(True, index=frame.index))
        original_gate = Gate("原始放入观察以上", "rating in ['放入观察','继续深挖']", lambda frame: frame["rating"].isin(["放入观察", "继续深挖"]))
        rows.append(_metrics(split, all_gate, df))
        rows.append(_metrics(split, original_gate, df))
    if data_dir:
        rows.extend(_period_hold_rows(data_dir, codes))
    return pd.DataFrame(rows)


def _period_hold_rows(data_dir: str | Path, codes: set[str] | None = None) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    returns = []
    for path in data_dir.glob("*/daily.csv"):
        if codes and path.parent.name not in codes:
            continue
        df = pd.read_csv(path)
        if len(df) < 2:
            continue
        start = float(df["close"].iloc[0])
        end = float(df["close"].iloc[-1])
        if start:
            returns.append((end / start - 1) * 100)
    values = pd.Series(returns, dtype=float)
    if values.empty:
        return []
    return [
        {
            "split": "full_period",
            "gate_name": "整段持有基线",
            "formula": "last_close / first_close - 1",
            "sample_count": int(len(values)),
            "avg_return_20d": round(float(values.mean()), 4),
            "positive_20d_rate": round(float((values > 0).mean()), 4),
            "std_return_20d": round(float(values.std(ddof=0)), 4),
            "loss_20d_over_5_rate": round(float((values <= -5).mean()), 4),
            "stability_score": round(float(values.mean() - 0.5 * values.std(ddof=0) - 10 * (values <= -5).mean()), 4),
        }
    ]


def _prefix(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in data.items() if k not in {"gate_name", "formula", "split"}}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df.get(col), errors="coerce")


def _skills(df: pd.DataFrame) -> pd.Series:
    return df["triggered_skills"].fillna("").astype(str)


def _values(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


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
    return round(float(values.mean() - 0.5 * values.std(ddof=0) - 10 * (values <= -5).mean()), 4)


def _markdown(result: pd.DataFrame, baseline: pd.DataFrame) -> str:
    lines = [
        "# Gate 优化与长期持有基线",
        "",
        "本报告只用于研究辅助。Gate 使用训练集早期搜索、训练集后期验证，test 只做锁定后检验。",
        "",
        "## 长期持有/原始口径基线",
        "",
        "| 数据集 | 口径 | 样本数 | 20日均值 | 20日正收益率 | 20日波动 | 稳定性分 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in baseline.iterrows():
        lines.append(
            f"| {row['split']} | {row['gate_name']} | {int(row['sample_count'])} | {_fmt(row['avg_return_20d'])} | "
            f"{_fmt(row['positive_20d_rate'])} | {_fmt(row['std_return_20d'])} | {_fmt(row['stability_score'])} |"
        )
    lines += [
        "",
        "## 验证集排序靠前 Gate",
        "",
        "| Gate | 公式 | valid样本 | valid20日均值 | valid正收益率 | test样本 | test20日均值 | test正收益率 | test稳定性 | 达标 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    top = result.head(12) if not result.empty else pd.DataFrame()
    for _, row in top.iterrows():
        hit = "是" if bool(row.get("target_hit_on_test")) else "否"
        lines.append(
            f"| {row['gate_name']} | `{row['formula']}` | {int(row['valid_sample_count'])} | {_fmt(row['valid_avg_return_20d'])} | "
            f"{_fmt(row['valid_positive_20d_rate'])} | {int(row['test_sample_count'])} | {_fmt(row['test_avg_return_20d'])} | "
            f"{_fmt(row['test_positive_20d_rate'])} | {_fmt(row['test_stability_score'])} | {hit} |"
        )
    if not result.empty:
        lines += [
            "",
            "## 候选公式池",
            "",
            "| Gate | 公式 |",
            "|---|---|",
        ]
        for _, row in result[["gate_name", "formula"]].drop_duplicates().sort_values("gate_name").iterrows():
            lines.append(f"| {row['gate_name']} | `{row['formula']}` |")
    if not result.empty and not bool(result["target_hit_on_test"].any()):
        lines += [
            "",
            "## 结论",
            "",
            f"- 当前样本中，没有 gate 在锁定 test 上同时达到 20日均值 {TARGET_AVG_20D}% 和 20日正收益率 {TARGET_POSITIVE_20D:.0%}。",
            "- 这说明目标需要更大样本、更长历史、更丰富输入通道或更细分的分群模型验证。",
            "- 在未通过 test 前，不应把任一 gate 升级为正式实操规则。",
        ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"
