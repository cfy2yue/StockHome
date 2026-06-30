from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
BANK_RETURN_20D = ((1 + 0.03) ** (20 / 252) - 1) * 100
BASELINE_GATE_PORTFOLIO_POSITIVE_RATE = 0.4043
BASELINE_GATE_PORTFOLIO_AVG_RETURN = -0.2435
BASELINE_GATE_PORTFOLIO_CASH_AVG = 0.1805


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio positive-rate oriented experiments.")
    parser.add_argument("--sample-code-count", type=int, default=100)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--panel-seed", default="date-generalization-panel-v1")
    parser.add_argument("--topn", nargs="+", type=int, default=[1, 2, 3, 5, 10, 15])
    args = parser.parse_args()

    frame = load_ground_truth(GT_SOURCES)
    rows = []
    block_order = list(TIME_BLOCKS)
    for panel in range(args.panels):
        panel_frame, codes = _sample_panel(frame, sample_code_count=args.sample_code_count, panel_index=panel, panel_seed=args.panel_seed)
        for valid_block in block_order[1:]:
            train_blocks = block_order[: block_order.index(valid_block)]
            train_df = _blocks(panel_frame, train_blocks)
            valid_df = _blocks(panel_frame, [valid_block])
            thresholds = _thresholds(train_df)
            for preset in PRESETS:
                scored = _score(valid_df, preset)
                for date_gate in DATE_GATES:
                    dated = _apply_date_gate(scored, train_df, thresholds, date_gate)
                    for row_gate in ROW_GATES:
                        gated = _apply_row_gate(dated, row_gate)
                        for frequency in FREQUENCIES:
                            freq_df = _apply_frequency(gated, frequency)
                            expected_decision_dates = _scheduled_date_count(scored, frequency)
                            for top_n in args.topn:
                                selected = _select_daily_top(freq_df, top_n)
                                metrics = _metrics(selected, expected_decision_dates=expected_decision_dates)
                                rows.append(
                                    {
                                        "panel": panel,
                                        "panel_code_count": len(codes),
                                        "train_blocks": "+".join(train_blocks),
                                        "valid_block": valid_block,
                                        "score_preset": preset,
                                        "date_gate": date_gate,
                                        "row_gate": row_gate,
                                        "decision_frequency": frequency,
                                        "top_n": top_n,
                                        **metrics,
                                    }
                                )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "portfolio_positive_rate_experiments.csv", index=False, encoding="utf-8-sig")
    aggregate = _aggregate(result)
    aggregate.to_csv(OUTPUT / "portfolio_positive_rate_experiments_aggregate.csv", index=False, encoding="utf-8-sig")
    diagnostics = _diagnostics(result, aggregate)
    diagnostics.to_csv(OUTPUT / "portfolio_positive_rate_experiments_diagnostics.csv", index=False, encoding="utf-8-sig")
    _write_report(aggregate, diagnostics, OUTPUT / "portfolio_positive_rate_experiments.md")
    print("A股研究Agent")
    print(f"rows: {len(result)}")
    print(f"wrote: {OUTPUT / 'portfolio_positive_rate_experiments_aggregate.csv'}")
    if not aggregate.empty:
        print(f"best: {aggregate.iloc[0].to_dict()}")


PRESETS = [
    "pullback_recovery",
    "peer_confirmed_pullback",
    "peer_breadth_quality",
    "risk_defensive",
    "news_peer_supported",
]

DATE_GATES = [
    "all_dates",
    "pool_pullback",
    "pool_not_hot",
    "peer_breadth_ok",
    "peer_breadth_strong",
    "low_news_risk",
]

ROW_GATES = [
    "none",
    "peer_relative_positive",
    "peer_breadth_above_half",
    "no_major_data_gap",
    "news_risk_low",
    "peer_and_gap_safe",
]

FREQUENCIES = [
    "twice_weekly",
    "weekly_friday",
    "weekly_tuesday",
    "every_2_weeks",
]


def _sample_panel(frame: pd.DataFrame, *, sample_code_count: int, panel_index: int, panel_seed: str) -> tuple[pd.DataFrame, list[str]]:
    codes = sorted(frame["code"].astype(str).str.zfill(6).dropna().unique())
    shuffled = sorted(codes, key=lambda code: hashlib.sha256(f"{panel_seed}:{code}".encode("utf-8")).hexdigest())
    start = panel_index * sample_code_count
    panel_codes = shuffled[start : start + sample_code_count]
    if len(panel_codes) < sample_code_count:
        raise ValueError(f"not enough codes for panel_index={panel_index}, requested={sample_code_count}, got={len(panel_codes)}")
    return frame[frame["code"].isin(set(panel_codes))].copy(), panel_codes


def _blocks(frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    if not blocks:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = pd.Series(False, index=frame.index)
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    data = frame[mask].copy()
    if "gt_status" in data:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()
    return data


def _thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    features = _date_features(train_df)
    if features.empty:
        return {}
    return {
        "pool_prior_q40": float(features["pool_avg_prior_return_20d"].quantile(0.40)),
        "pool_prior_q70": float(features["pool_avg_prior_return_20d"].quantile(0.70)),
        "peer_breadth_q55": float(features["pool_peer_breadth"].quantile(0.55)),
        "peer_breadth_q70": float(features["pool_peer_breadth"].quantile(0.70)),
        "news_risk_q50": float(features["pool_news_risk"].quantile(0.50)),
    }


def _date_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["prior_num"] = _num(data, "prior_return_20d")
    data["peer_breadth_num"] = _num(data, "peer_group_positive_breadth_20d")
    data["news_risk_num"] = _news_risk_signal(data)
    grouped = data.groupby(data["date"].astype(str))
    return grouped.agg(
        pool_avg_prior_return_20d=("prior_num", "mean"),
        pool_peer_breadth=("peer_breadth_num", "mean"),
        pool_news_risk=("news_risk_num", "mean"),
    ).fillna(0.0)


def _score(frame: pd.DataFrame, preset: str) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        data["_candidate_score"] = []
        return data
    rel = _num(data, "relative_strength_rank")
    counter = _num(data, "counter_score") / 10
    above = data.get("close_above_ma200", pd.Series(False, index=data.index)).astype(str).str.lower().isin(["true", "1"]).astype(float)
    prior = _num(data, "prior_return_20d")
    rsi = _num(data, "rsi14")
    atr = _num(data, "atr20_pct")
    peer_rel = _num(data, "peer_relative_to_group_20d")
    peer_breadth = _num(data, "peer_group_positive_breadth_20d")
    peer_above = _num(data, "peer_group_above_ma200_rate")
    news_count = _news_count_signal(data)
    news_opp = _news_opportunity_signal(data)
    news_risk = _news_risk_signal(data)
    safe_pullback = ((prior >= -15) & (prior <= 25)).astype(float)
    overheat = ((prior >= 60) | (rsi >= 80)).astype(float)

    if preset == "pullback_recovery":
        score = 0.55 * rel + 0.20 * counter + 0.15 * above + 0.45 * safe_pullback - 0.80 * overheat
    elif preset == "peer_confirmed_pullback":
        score = 0.42 * rel + 0.20 * counter + 0.15 * above + 0.32 * safe_pullback
        score += 0.18 * (peer_rel > 0).astype(float) + 0.18 * (peer_breadth >= 0.55).astype(float)
        score -= 0.75 * overheat + 0.20 * (news_risk > 0).astype(float)
    elif preset == "peer_breadth_quality":
        score = 0.35 * rel + 0.18 * counter + 0.18 * above + 0.24 * (peer_breadth >= 0.60).astype(float)
        score += 0.16 * (peer_above >= 0.55).astype(float) + 0.10 * (peer_rel > -3).astype(float)
        score -= 0.50 * overheat + 0.30 * (atr > 8).astype(float)
    elif preset == "risk_defensive":
        score = 0.34 * rel + 0.22 * counter + 0.20 * above + 0.20 * safe_pullback + 0.12 * (peer_breadth >= 0.50).astype(float)
        score -= 0.80 * (atr > 8).astype(float) + 0.80 * overheat + 0.50 * (news_risk > 0).astype(float)
    elif preset == "news_peer_supported":
        score = 0.38 * rel + 0.16 * counter + 0.12 * above + 0.18 * (news_count > 0).astype(float)
        score += 0.10 * news_opp + 0.18 * (peer_breadth >= 0.55).astype(float) + 0.12 * (peer_rel > 0).astype(float)
        score -= 0.55 * (news_risk > 0).astype(float) + 0.60 * overheat
    else:
        raise ValueError(f"unknown score preset: {preset}")
    data["_candidate_score"] = score
    return data.sort_values(["date", "_candidate_score", "code"], ascending=[True, False, True])


def _apply_date_gate(frame: pd.DataFrame, train_df: pd.DataFrame, thresholds: dict[str, float], gate: str) -> pd.DataFrame:
    if frame.empty or gate == "all_dates" or not thresholds:
        return frame.copy()
    features = _date_features(frame)
    if features.empty:
        return frame.iloc[0:0].copy()
    if gate == "pool_pullback":
        allowed = features[features["pool_avg_prior_return_20d"] <= thresholds["pool_prior_q40"]].index
    elif gate == "pool_not_hot":
        allowed = features[features["pool_avg_prior_return_20d"] <= thresholds["pool_prior_q70"]].index
    elif gate == "peer_breadth_ok":
        allowed = features[features["pool_peer_breadth"] >= thresholds["peer_breadth_q55"]].index
    elif gate == "peer_breadth_strong":
        allowed = features[features["pool_peer_breadth"] >= thresholds["peer_breadth_q70"]].index
    elif gate == "low_news_risk":
        allowed = features[features["pool_news_risk"] <= thresholds["news_risk_q50"]].index
    else:
        raise ValueError(f"unknown date gate: {gate}")
    return frame[frame["date"].astype(str).isin(set(allowed.astype(str)))].copy()


def _apply_row_gate(frame: pd.DataFrame, gate: str) -> pd.DataFrame:
    if frame.empty or gate == "none":
        return frame.copy()
    data = frame.copy()
    selector = pd.Series(True, index=data.index)
    if gate == "peer_relative_positive":
        selector &= _num(data, "peer_relative_to_group_20d") > 0
    elif gate == "peer_breadth_above_half":
        selector &= _num(data, "peer_group_positive_breadth_20d") >= 0.50
    elif gate == "no_major_data_gap":
        gaps = data["data_gaps"].fillna("").astype(str) if "data_gaps" in data else pd.Series("", index=data.index)
        selector &= ~gaps.str.contains("financial_publish_date_missing", regex=False)
    elif gate == "news_risk_low":
        selector &= _news_risk_signal(data) <= 0
    elif gate == "peer_and_gap_safe":
        gaps = data["data_gaps"].fillna("").astype(str) if "data_gaps" in data else pd.Series("", index=data.index)
        selector &= _num(data, "peer_group_positive_breadth_20d") >= 0.50
        selector &= _num(data, "peer_relative_to_group_20d") > -3
        selector &= ~gaps.str.contains("financial_publish_date_missing", regex=False)
    else:
        raise ValueError(f"unknown row gate: {gate}")
    return data[selector].copy()


def _apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency == "twice_weekly":
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(f"unknown frequency: {frequency}")


def _select_daily_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["date", "_candidate_score", "code"], ascending=[True, False, True])
    return ordered.groupby("date", group_keys=False).head(top_n).copy()


def _metrics(selected: pd.DataFrame, *, expected_decision_dates: int) -> dict[str, Any]:
    if selected.empty:
        cash_values = pd.Series([BANK_RETURN_20D] * max(expected_decision_dates, 0), dtype=float)
        return {
            "decision_dates": 0,
            "expected_decision_dates": int(expected_decision_dates),
            "decision_coverage": 0.0,
            "avg_selected_count": 0.0,
            "avg_return_20d": None,
            "raw_positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "cash_blended_avg_return_20d": round(float(cash_values.mean()), 4) if not cash_values.empty else None,
            "cash_blended_positive_20d_rate": round(float((cash_values > 0).mean()), 4) if not cash_values.empty else None,
            "stability_score": None,
        }
    daily = selected.groupby("date").agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
    values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
    decision_dates = int(len(daily))
    expected_dates = max(int(expected_decision_dates), decision_dates)
    coverage = decision_dates / expected_dates if expected_dates else 0.0
    if values.empty:
        cash_values = pd.Series([BANK_RETURN_20D] * expected_dates, dtype=float)
        return {
            "decision_dates": decision_dates,
            "expected_decision_dates": expected_dates,
            "decision_coverage": round(float(coverage), 4),
            "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
            "avg_return_20d": None,
            "raw_positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "cash_blended_avg_return_20d": round(float(cash_values.mean()), 4) if not cash_values.empty else None,
            "cash_blended_positive_20d_rate": round(float((cash_values > 0).mean()), 4) if not cash_values.empty else None,
            "stability_score": None,
        }
    skipped_dates = max(expected_dates - decision_dates, 0)
    cash_blended = pd.concat(
        [
            values.clip(lower=-100),
            pd.Series([BANK_RETURN_20D] * skipped_dates, dtype=float),
        ],
        ignore_index=True,
    )
    avg = float(values.mean())
    std = float(values.std(ddof=0))
    loss = float((values <= -5).mean())
    return {
        "decision_dates": decision_dates,
        "expected_decision_dates": expected_dates,
        "decision_coverage": round(float(coverage), 4),
        "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
        "avg_return_20d": round(avg, 4),
        "raw_positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4),
        "cash_blended_positive_20d_rate": round(float((cash_blended > 0).mean()), 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def _aggregate(result: pd.DataFrame) -> pd.DataFrame:
    grouped = result.groupby(["score_preset", "date_gate", "row_gate", "decision_frequency", "top_n"])
    aggregate = grouped.agg(
        panel_blocks=("valid_block", "count"),
        avg_decision_dates=("decision_dates", "mean"),
        avg_expected_decision_dates=("expected_decision_dates", "mean"),
        decision_coverage=("decision_coverage", "mean"),
        avg_selected_count=("avg_selected_count", "mean"),
        avg_return_mean=("avg_return_20d", "mean"),
        avg_return_std=("avg_return_20d", "std"),
        raw_positive_20d_rate_mean=("raw_positive_20d_rate", "mean"),
        raw_positive_20d_rate_std=("raw_positive_20d_rate", "std"),
        cash_blended_avg_return_20d_mean=("cash_blended_avg_return_20d", "mean"),
        cash_blended_positive_20d_rate_mean=("cash_blended_positive_20d_rate", "mean"),
        loss_over_5_mean=("loss_20d_over_5_rate", "mean"),
        stability_mean=("stability_score", "mean"),
        raw_positive_hit_blocks=("raw_positive_20d_rate", lambda value: int((pd.to_numeric(value, errors="coerce") >= 0.60).sum())),
    ).reset_index()
    aggregate["coverage_ok"] = aggregate["decision_coverage"] >= 0.25
    aggregate["raw_positive_lift_vs_gate"] = aggregate["raw_positive_20d_rate_mean"] - BASELINE_GATE_PORTFOLIO_POSITIVE_RATE
    aggregate["avg_return_lift_vs_gate"] = aggregate["avg_return_mean"] - BASELINE_GATE_PORTFOLIO_AVG_RETURN
    aggregate["cash_blended_avg_lift_vs_gate"] = aggregate["cash_blended_avg_return_20d_mean"] - BASELINE_GATE_PORTFOLIO_CASH_AVG
    aggregate["rank_score"] = (
        aggregate["raw_positive_20d_rate_mean"].fillna(0)
        - 0.20 * aggregate["raw_positive_20d_rate_std"].fillna(0)
        - 0.15 * aggregate["loss_over_5_mean"].fillna(0)
        + 0.03 * aggregate["cash_blended_avg_return_20d_mean"].fillna(0)
        + 0.10 * aggregate["decision_coverage"].clip(upper=1).fillna(0)
    )
    aggregate = aggregate.sort_values(
        ["coverage_ok", "rank_score", "raw_positive_20d_rate_mean", "loss_over_5_mean", "avg_return_mean"],
        ascending=[False, False, False, True, False],
    )
    return aggregate


def _diagnostics(result: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    best = aggregate.iloc[0]
    keys = ["score_preset", "date_gate", "row_gate", "decision_frequency", "top_n"]
    selector = pd.Series(True, index=result.index)
    for key in keys:
        selector &= result[key].astype(str).eq(str(best[key]))
    subset = result[selector].copy()
    return (
        subset.groupby("valid_block")
        .agg(
            panel_blocks=("panel", "count"),
            avg_decision_dates=("decision_dates", "mean"),
            decision_coverage=("decision_coverage", "mean"),
            avg_return_mean=("avg_return_20d", "mean"),
            raw_positive_20d_rate_mean=("raw_positive_20d_rate", "mean"),
            cash_blended_avg_return_20d_mean=("cash_blended_avg_return_20d", "mean"),
            cash_blended_positive_20d_rate_mean=("cash_blended_positive_20d_rate", "mean"),
            loss_over_5_mean=("loss_20d_over_5_rate", "mean"),
        )
        .reset_index()
    )


def _write_report(aggregate: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Portfolio Positive Rate Experiments",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 目的",
        "",
        "当前组合模式正收益率低于目标。本实验用现有时间安全特征，在 DeepSeek 调用前系统搜索 TopN、决策频率、日期 gate、同行 gate、新闻/缺口 gate，寻找下一轮 DS round 候选。",
        "",
        "## 指标口径",
        "",
        "- `raw_positive_20d_rate` / `raw_positive_20d_rate_mean`：实际选中组合后的原始 20 日正收益率，优先用于证明选股能力。",
        "- `cash_blended_avg_return_20d`：未触发策略时按 3% 年化现金替代的体验口径，不能单独证明选股能力。",
        "- `decision_coverage`：触发决策日期占可决策日期比例；覆盖率过低时，不得把少数漂亮样本当作稳定策略。",
        "",
        "## Top Candidates",
        "",
        _table(aggregate.head(30)),
        "",
        "## Best Candidate By Time Block",
        "",
        _table(diagnostics),
        "",
        "## 使用边界",
        "",
        "- 本实验使用未来 20 日收益做后验评价，不进入 DeepSeek 决策输入。",
        "- 结果只能作为下一轮策略候选，不能直接作为最终验收。",
        "- 若候选通过率高但决策日期过少，必须标记为覆盖不足，不能替代全流程验证。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _scheduled_date_count(frame: pd.DataFrame, frequency: str) -> int:
    if frame.empty:
        return 0
    scheduled = _apply_frequency(frame, frequency)
    return int(scheduled["date"].astype(str).nunique())


def _num(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def _news_count_signal(frame: pd.DataFrame) -> pd.Series:
    return _num(frame, "news_count_30d").combine(_num(frame, "event_count"), max)


def _news_risk_signal(frame: pd.DataFrame) -> pd.Series:
    legacy = _num(frame, "news_risk_event_score_30d") + _num(frame, "news_warning_score_30d")
    return legacy.combine(_num(frame, "news_warning_score"), max)


def _news_opportunity_signal(frame: pd.DataFrame) -> pd.Series:
    legacy = _num(frame, "news_opportunity_event_score_30d") + _num(frame, "news_opportunity_alert_score_30d")
    return legacy.combine(_num(frame, "news_opportunity_score"), max)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")


if __name__ == "__main__":
    main()
