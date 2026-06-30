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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic portfolio candidate preset experiments before costly DS rounds.")
    parser.add_argument("--sample-code-count", type=int, default=100)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--panel-seed", default="date-generalization-panel-v1")
    parser.add_argument("--topn", nargs="+", type=int, default=[3, 5, 10, 20])
    args = parser.parse_args()

    frame = load_ground_truth(GT_SOURCES)
    rows = []
    for panel in range(args.panels):
        panel_frame, codes = _sample_panel(frame, sample_code_count=args.sample_code_count, panel_index=panel, panel_seed=args.panel_seed)
        for block in list(TIME_BLOCKS)[1:]:
            block_frame = _window(panel_frame, block)
            for preset in PRESETS:
                scored = _score_preset(block_frame, preset)
                for top_n in args.topn:
                    selected = _diverse_top(scored, top_n)
                    metrics = _metrics(selected)
                    rows.append(
                        {
                            "panel": panel,
                            "panel_code_count": len(codes),
                            "valid_block": block,
                            "portfolio_preset": preset,
                            "top_n": top_n,
                            **metrics,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "portfolio_candidate_experiments.csv", index=False, encoding="utf-8-sig")
    aggregate = (
        result.groupby(["portfolio_preset", "top_n"])
        .agg(
            blocks=("valid_block", "nunique"),
            panel_blocks=("valid_block", "count"),
            avg_return_mean=("avg_return_20d", "mean"),
            avg_return_std=("avg_return_20d", "std"),
            positive_rate_mean=("positive_20d_rate", "mean"),
            positive_rate_std=("positive_20d_rate", "std"),
            stability_mean=("stability_score", "mean"),
        )
        .reset_index()
        .sort_values(["positive_rate_mean", "avg_return_mean", "stability_mean"], ascending=False)
    )
    aggregate.to_csv(OUTPUT / "portfolio_candidate_experiments_aggregate.csv", index=False, encoding="utf-8-sig")
    _write_report(result, aggregate, OUTPUT / "portfolio_candidate_experiments.md")
    print("A股研究Agent")
    print(f"wrote: {OUTPUT / 'portfolio_candidate_experiments_aggregate.csv'}")
    print(f"best: {aggregate.iloc[0].to_dict() if not aggregate.empty else 'NA'}")


PRESETS = [
    "raw_relative_strength",
    "no_overheat_no_evidence",
    "balanced_momentum",
    "pullback_recovery",
    "low_vol_quality",
    "news_supported",
]


def _score_preset(frame: pd.DataFrame, preset: str) -> pd.DataFrame:
    data = frame.copy()
    if "gt_status" in data:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()
    if data.empty:
        data["_candidate_score"] = []
        return data
    rel = _num(data, "relative_strength_rank")
    counter = _num(data, "counter_score") / 10
    above = data.get("close_above_ma200", pd.Series(False, index=data.index)).astype(str).str.lower().isin(["true", "1"]).astype(float)
    prior = _num(data, "prior_return_20d")
    rsi = _num(data, "rsi14")
    atr = _num(data, "atr20_pct")
    news_count = _num(data, "news_count_30d")
    news_opp = _num(data, "news_opportunity_event_score_30d") + _num(data, "news_opportunity_alert_score_30d")
    news_risk = _num(data, "news_risk_event_score_30d") + _num(data, "news_warning_score_30d")
    overheat = ((prior >= 80) | (rsi >= 85)).astype(float)
    score = rel + 0.18 * counter + 0.12 * above - 0.08 * news_risk
    if preset == "no_overheat_no_evidence":
        score -= _overheat_no_evidence(data).astype(float) * 2.5
    elif preset == "balanced_momentum":
        score += 0.10 * above + 0.05 * news_opp
        score -= ((prior > 60) | (rsi > 80)).astype(float) * 0.8
        score -= (atr > 12).astype(float) * 0.35
    elif preset == "pullback_recovery":
        score = 0.55 * rel + 0.20 * counter + 0.15 * above
        score += ((prior >= -15) & (prior <= 25)).astype(float) * 0.45
        score -= ((prior > 60) | (rsi > 80)).astype(float) * 0.8
    elif preset == "low_vol_quality":
        score = 0.45 * rel + 0.25 * counter + 0.15 * above
        score -= (atr > 8).astype(float) * 0.60
        score -= overheat * 0.75
    elif preset == "news_supported":
        score = 0.50 * rel + 0.20 * counter + 0.15 * above + (news_count > 0).astype(float) * 0.35 + news_opp * 0.08
        score -= (news_risk > 0).astype(float) * 0.5
        score -= _overheat_no_evidence(data).astype(float) * 1.0
    data["_candidate_score"] = score
    return data.sort_values(["_candidate_score", "date", "code"], ascending=[False, True, True])


def _diverse_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = []
    seen_codes = set()
    seen_dates = set()
    for index, row in frame.iterrows():
        code = str(row.get("code")).zfill(6)
        date = str(row.get("date"))
        if code in seen_codes or date in seen_dates:
            continue
        selected.append(index)
        seen_codes.add(code)
        seen_dates.add(date)
        if len(selected) >= top_n:
            break
    return frame.loc[selected].copy()


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_20d"), errors="coerce").dropna()
    if values.empty:
        return {"selected_count": 0, "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    avg = float(values.mean())
    std = float(values.std(ddof=0))
    loss = float((values <= -5).mean())
    return {
        "selected_count": int(len(values)),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def _window(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _sample_panel(frame: pd.DataFrame, *, sample_code_count: int, panel_index: int, panel_seed: str) -> tuple[pd.DataFrame, list[str]]:
    codes = sorted(frame["code"].astype(str).str.zfill(6).dropna().unique())
    shuffled = sorted(codes, key=lambda code: hashlib.sha256(f"{panel_seed}:{code}".encode("utf-8")).hexdigest())
    start = panel_index * sample_code_count
    panel_codes = shuffled[start : start + sample_code_count]
    if len(panel_codes) < sample_code_count:
        raise ValueError("not enough codes for requested panel")
    return frame[frame["code"].isin(set(panel_codes))].copy(), panel_codes


def _num(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def _overheat_no_evidence(frame: pd.DataFrame) -> pd.Series:
    prior = _num(frame, "prior_return_20d")
    rsi = _num(frame, "rsi14")
    news_count = _num(frame, "news_count_30d")
    data_gaps = frame["data_gaps"].fillna("").astype(str) if "data_gaps" in frame else pd.Series("", index=frame.index)
    return ((prior >= 80) | (rsi >= 85)) & (news_count <= 0) & data_gaps.str.contains("financial_publish_date_missing", regex=False)


def _write_report(result: pd.DataFrame, aggregate: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Portfolio Candidate Experiments",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Aggregate",
        "",
        _table(aggregate.head(30)),
        "",
        "## Notes",
        "",
        "- 这是确定性候选池实验，用未来20日收益做后验评估，不输入 DeepSeek 决策。",
        "- 它用于决定下一轮 DS 组合模式 evidence pack 的候选预设和 TopN，而不是最终验收。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    return str(value)


if __name__ == "__main__":
    main()
