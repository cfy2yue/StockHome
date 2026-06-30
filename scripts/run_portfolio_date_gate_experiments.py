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

from scripts.run_portfolio_candidate_experiments import _sample_panel, _score_preset, _table
from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward date-gate experiments for portfolio mode.")
    parser.add_argument("--sample-code-count", type=int, default=100)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--panel-seed", default="date-generalization-panel-v1")
    parser.add_argument("--portfolio-preset", default="pullback_recovery")
    parser.add_argument("--topn", nargs="+", type=int, default=[3, 5, 10])
    args = parser.parse_args()

    frame = load_ground_truth(GT_SOURCES)
    rows = []
    block_order = list(TIME_BLOCKS)
    for panel in range(args.panels):
        panel_frame, codes = _sample_panel(frame, sample_code_count=args.sample_code_count, panel_index=panel, panel_seed=args.panel_seed)
        for block in block_order[1:]:
            train_blocks = block_order[: block_order.index(block)]
            train_df = _blocks(panel_frame, train_blocks)
            valid_df = _blocks(panel_frame, [block])
            date_gates = _build_date_gates(train_df)
            scored_valid = _score_preset(valid_df, args.portfolio_preset)
            valid_features = _date_features(valid_df)
            for gate in date_gates:
                gate_name, gate_formula = gate["name"], gate["formula"]
                allowed_dates = _apply_date_gate(valid_features, gate)
                gated = scored_valid[scored_valid["date"].astype(str).isin(allowed_dates)].copy()
                for frequency in ["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"]:
                    freq_df = _apply_frequency(gated, frequency)
                    for top_n in args.topn:
                        selected = _select_daily_top(freq_df, top_n)
                        metrics = _metrics(selected)
                        rows.append(
                            {
                                "panel": panel,
                                "panel_code_count": len(codes),
                                "train_blocks": "+".join(train_blocks),
                                "valid_block": block,
                                "portfolio_preset": args.portfolio_preset,
                                "date_gate": gate_name,
                                "date_gate_formula": gate_formula,
                                "decision_frequency": frequency,
                                "top_n": top_n,
                                **metrics,
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "portfolio_date_gate_experiments.csv", index=False, encoding="utf-8-sig")
    aggregate = _aggregate(result)
    aggregate.to_csv(OUTPUT / "portfolio_date_gate_experiments_aggregate.csv", index=False, encoding="utf-8-sig")
    _write_report(aggregate, OUTPUT / "portfolio_date_gate_experiments.md")
    print("A股研究Agent")
    print(f"wrote: {OUTPUT / 'portfolio_date_gate_experiments_aggregate.csv'}")
    print(f"best: {aggregate.iloc[0].to_dict() if not aggregate.empty else 'NA'}")


def _blocks(frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    if not blocks:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = pd.Series(False, index=frame.index)
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame[mask].copy()


def _build_date_gates(train_df: pd.DataFrame) -> list[dict[str, Any]]:
    features = _date_features(train_df)
    if features.empty:
        return [{"name": "all_dates", "formula": "all decision dates", "feature": "", "op": "all", "threshold": None}]
    gates: list[dict[str, Any]] = [{"name": "all_dates", "formula": "all decision dates", "feature": "", "op": "all", "threshold": None}]
    specs = [
        ("pool_not_hot", "pool_avg_prior_return_20d", "<=", 0.70),
        ("pool_pullback", "pool_avg_prior_return_20d", "<=", 0.40),
        ("breadth_recovering", "pool_positive_breadth_20d", ">=", 0.55),
        ("low_overheat_ratio", "pool_overheat_ratio", "<=", 0.35),
        ("low_dispersion", "pool_return_dispersion_20d", "<=", 0.55),
        ("low_news_risk", "pool_avg_news_risk", "<=", 0.50),
    ]
    for name, feature, op, q in specs:
        if feature not in features or features[feature].dropna().empty:
            continue
        threshold = float(features[feature].quantile(q))
        gates.append({"name": name, "formula": f"{feature} {op} train_quantile({q})={threshold:.4f}", "feature": feature, "op": op, "threshold": threshold})
    return gates


def _apply_date_gate(features: pd.DataFrame, gate: dict[str, Any]) -> set[str]:
    if features.empty:
        return set()
    if gate["op"] == "all":
        return set(features.index.astype(str))
    feature = str(gate["feature"])
    if feature not in features:
        return set()
    threshold = float(gate["threshold"])
    if gate["op"] == "<=":
        return set(features[features[feature] <= threshold].index.astype(str))
    return set(features[features[feature] >= threshold].index.astype(str))


def _date_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name="date"))
    data = frame.copy()
    data["prior_num"] = pd.to_numeric(data.get("prior_return_20d"), errors="coerce")
    data["rsi_num"] = pd.to_numeric(data.get("rsi14"), errors="coerce")
    data["news_risk_num"] = pd.to_numeric(data.get("news_risk_event_score_30d"), errors="coerce").fillna(0)
    data["overheat"] = (data["prior_num"] >= 80) | (data["rsi_num"] >= 85)
    grouped = data.groupby(data["date"].astype(str))
    return grouped.agg(
        pool_avg_prior_return_20d=("prior_num", "mean"),
        pool_positive_breadth_20d=("prior_num", lambda value: float((value > 0).mean())),
        pool_return_dispersion_20d=("prior_num", "std"),
        pool_overheat_ratio=("overheat", "mean"),
        pool_avg_news_risk=("news_risk_num", "mean"),
    ).fillna(0)


def _apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    dates = pd.to_datetime(data["date"], errors="coerce")
    if frequency == "weekly_friday":
        data = data[dates.dt.dayofweek == 4].copy()
    elif frequency == "weekly_tuesday":
        data = data[dates.dt.dayofweek == 1].copy()
    elif frequency == "every_2_weeks":
        week = dates.dt.isocalendar().week.astype(int)
        data = data[week % 2 == 0].copy()
    return data


def _select_daily_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(["date", "_candidate_score", "code"], ascending=[True, False, True])
    return ordered.groupby("date", group_keys=False).head(top_n).copy()


def _metrics(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"decision_dates": 0, "avg_selected_count": 0, "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    daily = selected.groupby("date").agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
    values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
    if values.empty:
        return {"decision_dates": int(len(daily)), "avg_selected_count": round(float(daily["selected_count"].mean()), 4), "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    avg = float(values.mean())
    std = float(values.std(ddof=0))
    loss = float((values <= -5).mean())
    return {
        "decision_dates": int(len(daily)),
        "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def _aggregate(result: pd.DataFrame) -> pd.DataFrame:
    return (
        result.groupby(["portfolio_preset", "date_gate", "decision_frequency", "top_n"])
        .agg(
            panel_blocks=("valid_block", "count"),
            avg_decision_dates=("decision_dates", "mean"),
            avg_return_mean=("avg_return_20d", "mean"),
            avg_return_std=("avg_return_20d", "std"),
            positive_rate_mean=("positive_20d_rate", "mean"),
            positive_rate_std=("positive_20d_rate", "std"),
            stability_mean=("stability_score", "mean"),
            hit_blocks=("positive_20d_rate", lambda value: int((pd.to_numeric(value, errors="coerce") >= 0.60).sum())),
        )
        .reset_index()
        .sort_values(["positive_rate_mean", "avg_return_mean", "stability_mean"], ascending=False)
    )


def _write_report(aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Portfolio Date Gate Experiments",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Aggregate",
        "",
        _table(aggregate.head(40)),
        "",
        "## Notes",
        "",
        "- Gate 阈值只用历史 train_blocks 估计，再验证当前 valid_block。",
        "- 这是 DS 前的系统性实验，用于选择下一轮组合模式候选日期和决策频率。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

