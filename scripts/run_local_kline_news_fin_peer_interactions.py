from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kline_channel_exploration import DEFAULT_DAILY_DIR, GT_SOURCES, prepare_frame  # noqa: E402
from src.agent_training.dual_mode_round import load_ground_truth  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
TRAIN_BLOCKS = ["H2023_1", "H2023_2", "H2024_1"]
VALID_BLOCKS = ["H2024_2", "H2025_1"]
TEST_BLOCKS = ["H2025_2", "H2026_1"]
KLINE_20D_PULLBACK_THRESHOLD = -10.1231
KLINE_60D_DEEP_DRAWDOWN_THRESHOLD = -16.9912


@dataclass(frozen=True)
class LocalRule:
    rule_id: str
    rule_kind: str
    description: str
    selector: Callable[[pd.DataFrame], pd.Series]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local news/financial/peer/K-line interaction screen before spending DeepSeek tokens.")
    parser.add_argument("--output-prefix", default="local_kline_news_fin_peer_interactions_v1")
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--max-daily-files", type=int, default=0)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(load_ground_truth(GT_SOURCES), daily_dir=Path(args.daily_dir), max_daily_files=args.max_daily_files)
    frame = add_interaction_flags(frame)
    result, block_result = evaluate_rules(frame)

    csv_path = REPORT_DIR / f"{args.output_prefix}.csv"
    block_csv_path = REPORT_DIR / f"{args.output_prefix}_blocks.csv"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    block_result.to_csv(block_csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(frame, result, block_result, csv_path, block_csv_path), encoding="utf-8")

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"rules={len(result)}")
    print(f"csv={csv_path}")
    print(f"block_csv={block_csv_path}")
    print(f"report={report_path}")


def add_interaction_flags(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["kline_20d_pullback_flag"] = _num(data, "kline_return_20d").le(KLINE_20D_PULLBACK_THRESHOLD)
    data["kline_60d_deep_drawdown_flag"] = _num(data, "kline_return_60d").le(KLINE_60D_DEEP_DRAWDOWN_THRESHOLD)
    data["kline_cycle_vol_rebound_flag"] = _num(data, "kline_volatility_ratio_20_60").ge(1.2560) & _num(data, "kline_trend_consistency_20d").le(0.35)

    peer_breadth = _coalesce(data, ["peer_kline_group_positive_breadth_20d", "peer_group_positive_breadth_20d"])
    data["peer_confirmed_flag"] = peer_breadth.ge(0.45)
    data["peer_weak_flag"] = peer_breadth.lt(0.40) | peer_breadth.isna()

    news_missing = _num(data, "news_missing_rate").fillna(1.0)
    news_risk = _coalesce(data, ["ds_news_risk_score", "news_warning_score", "news_risk_event_score_30d", "news_warning_score_30d"]).fillna(0.0)
    news_opportunity = _coalesce(data, ["ds_news_opportunity_score", "news_opportunity_score", "news_opportunity_event_score_30d"]).fillna(0.0)
    data["news_available_flag"] = news_missing.lt(0.8)
    data["news_low_risk_flag"] = news_risk.le(0.5)
    data["news_high_opportunity_flag"] = news_opportunity.ge(0.6)

    financial_missing = _num(data, "financial_report_missing_rate").fillna(1.0)
    financial_risk = _num(data, "financial_quality_risk_score").fillna(0.0)
    data["financial_available_flag"] = financial_missing.lt(0.8)
    data["financial_low_risk_flag"] = financial_risk.lt(0.6)
    data["financial_high_risk_flag"] = financial_risk.ge(0.6)

    triggered = data.get("triggered_skills", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    data["book_skill_present_flag"] = triggered.ne("") & ~triggered.str.lower().isin(["nan", "none", "[]"])
    data["major_confirmation_gap_flag"] = (
        ~data["news_available_flag"]
        | ~data["financial_available_flag"]
        | data["peer_weak_flag"]
        | ~data["book_skill_present_flag"]
    )
    data["clean_cross_channel_context_flag"] = (
        data["peer_confirmed_flag"]
        & data["news_available_flag"]
        & data["news_low_risk_flag"]
        & data["financial_available_flag"]
        & data["financial_low_risk_flag"]
        & data["book_skill_present_flag"]
    )
    data["opportunity_but_confirmation_weak_flag"] = data["news_high_opportunity_flag"] & (data["peer_weak_flag"] | ~data["financial_available_flag"])
    return data


def evaluate_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = _rules()
    train = frame[frame["time_block"].isin(TRAIN_BLOCKS)]
    valid = frame[frame["time_block"].isin(VALID_BLOCKS)]
    test = frame[frame["time_block"].isin(TEST_BLOCKS)]
    baseline = _metrics(test)
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for rule in rules:
        row = {"rule_id": rule.rule_id, "rule_kind": rule.rule_kind, "description": rule.description}
        masks = {
            "train": _safe_selector(rule, train),
            "valid": _safe_selector(rule, valid),
            "test": _safe_selector(rule, test),
        }
        for split, data in [("train", train), ("valid", valid), ("test", test)]:
            row.update({f"{split}_{key}": value for key, value in _metrics(data[masks[split]]).items()})
        row["test_minus_baseline_pos"] = _delta(row["test_positive_20d_rate"], baseline["positive_20d_rate"])
        row["test_minus_baseline_avg"] = _delta(row["test_avg_return_20d"], baseline["avg_return_20d"])
        row["test_loss_delta"] = _delta(row["test_loss_gt5_rate"], baseline["loss_gt5_rate"])
        row["gate_status"] = _gate_status(row)
        rows.append(row)

        for block in TRAIN_BLOCKS + VALID_BLOCKS + TEST_BLOCKS:
            data = frame[frame["time_block"].eq(block)]
            metrics = _metrics(data[_safe_selector(rule, data)])
            block_rows.append({"rule_id": rule.rule_id, "time_block": block, **metrics})
    return pd.DataFrame(rows), pd.DataFrame(block_rows)


def render_report(frame: pd.DataFrame, result: pd.DataFrame, block_result: pd.DataFrame, csv_path: Path, block_csv_path: Path) -> str:
    shortlisted = result[result["gate_status"].astype(str).eq("observe_candidate")]
    lines = [
        "# Local K-Line x News x Financial x Peer Interaction Screen V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。本实验不调用 DeepSeek，不读取 API key/token。",
        "",
        "## Purpose",
        "",
        "在继续花 DeepSeek token 前，先用本地后验数据筛查多尺度 K 线、新闻/公告、财报和同组相对量价的交互规则，避免把明显不泛化的规则送入 Agent round。",
        "",
        "## Split",
        "",
        f"- rows: `{len(frame)}`",
        f"- train_blocks: `{','.join(TRAIN_BLOCKS)}`",
        f"- valid_blocks: `{','.join(VALID_BLOCKS)}`",
        f"- test_blocks: `{','.join(TEST_BLOCKS)}`",
        f"- output_csv: `{csv_path}`",
        f"- block_csv: `{block_csv_path}`",
        "",
        "## Gate Rules",
        "",
        "- 只用 train/valid 设计规则，最终只看 test 作为是否进入 DS 小 shard 的依据。",
        "- 候选进入 DS 的最低条件：test 正收益率较 baseline 提升 `>=0.04`，test loss_gt5 不升，test 样本不少于 120，且单一股票集中度不过高。",
        "- 若 test 低于 baseline、只靠单一 block/单一股票、或规则属于已反证路径，则停止进入 DS。",
        "",
        "## Summary",
        "",
        _table(result),
        "",
        "## Shortlist",
        "",
        _table(shortlisted) if not shortlisted.empty else "本轮没有规则达到 observe_candidate。后续不应直接放大 DS，应先改数据图谱或规则定义。",
        "",
        "## Block Metrics",
        "",
        _table(block_result),
        "",
        "## Interpretation",
        "",
        "- `kline_20d_pullback` 若仍是唯一候选，也只能作为弱量价提示进入 evidence pack，不能单独升级。",
        "- 新闻机会强但同行弱或财报缺失、财报高风险、peer 弱、Book Skill 缺口，应优先作为反证/降权条件。",
        "- 当前 peer K 线仍是候选池横截面；接入行业/概念/地域/相关股票图谱前，不宣称是真正同行规则。",
    ]
    return "\n".join(lines) + "\n"


def _rules() -> list[LocalRule]:
    return [
        LocalRule("baseline_all", "baseline", "全体 test baseline", lambda df: pd.Series(True, index=df.index)),
        LocalRule("kline_20d_pullback_all", "candidate", "20日回撤弱量价提示", lambda df: df["kline_20d_pullback_flag"]),
        LocalRule("kline_pullback_peer_confirmed", "candidate", "20日回撤且同组20日正收益广度>=0.45", lambda df: df["kline_20d_pullback_flag"] & df["peer_confirmed_flag"]),
        LocalRule("kline_pullback_peer_news_lowrisk", "candidate", "20日回撤+同组确认+新闻风险不高", lambda df: df["kline_20d_pullback_flag"] & df["peer_confirmed_flag"] & df["news_low_risk_flag"]),
        LocalRule("kline_pullback_clean_cross_channel", "candidate", "20日回撤且新闻/财报/同行/BookSkill无主要缺口", lambda df: df["kline_20d_pullback_flag"] & df["clean_cross_channel_context_flag"]),
        LocalRule("kline_pullback_major_gap_counter", "counterevidence", "20日回撤但新闻/财报/同行/BookSkill存在主要缺口", lambda df: df["kline_20d_pullback_flag"] & df["major_confirmation_gap_flag"]),
        LocalRule("kline_pullback_peer_weak_counter", "counterevidence", "20日回撤但同组广度弱或缺失", lambda df: df["kline_20d_pullback_flag"] & df["peer_weak_flag"]),
        LocalRule("kline_pullback_financial_high_risk_counter", "counterevidence", "20日回撤但财务质量风险高", lambda df: df["kline_20d_pullback_flag"] & df["financial_high_risk_flag"]),
        LocalRule("kline_pullback_news_opportunity_weak_confirm_counter", "counterevidence", "20日回撤且机会新闻高但同行弱或财报缺失", lambda df: df["kline_20d_pullback_flag"] & df["opportunity_but_confirmation_weak_flag"]),
        LocalRule("kline_pullback_high_atr_control", "control", "20日回撤且ATR高波动", lambda df: df["kline_20d_pullback_flag"] & _num(df, "kline_atr20_pct").ge(6.7167)),
        LocalRule("kline_cycle_vol_rebound", "candidate", "震荡/波动扩张回弹提示", lambda df: df["kline_cycle_vol_rebound_flag"]),
        LocalRule("kline_cycle_vol_rebound_peer_confirmed", "candidate", "震荡/波动扩张且同组确认", lambda df: df["kline_cycle_vol_rebound_flag"] & df["peer_confirmed_flag"]),
        LocalRule("kline_60d_deep_drawdown_control", "rejected_control", "60日深跌正向规则反证", lambda df: df["kline_60d_deep_drawdown_flag"]),
    ]


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_20d"), errors="coerce").dropna()
    if values.empty:
        return {
            "sample_count": 0,
            "unique_stocks": 0,
            "top_stock_share": pd.NA,
            "avg_return_20d": pd.NA,
            "positive_20d_rate": pd.NA,
            "loss_gt5_rate": pd.NA,
            "stability_score": pd.NA,
        }
    counts = frame.loc[values.index, "code"].astype(str).value_counts(normalize=True)
    avg = float(values.mean())
    pos = float((values > 0).mean())
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    return {
        "sample_count": int(len(values)),
        "unique_stocks": int(frame.loc[values.index, "code"].astype(str).nunique()),
        "top_stock_share": round(float(counts.iloc[0]), 4) if not counts.empty else pd.NA,
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(pos, 4),
        "loss_gt5_rate": round(loss, 4),
        "stability_score": round(avg - 0.35 * std - 8 * loss, 4),
    }


def _gate_status(row: dict[str, Any]) -> str:
    if row["rule_id"] == "baseline_all":
        return "baseline"
    if row["test_sample_count"] < 120:
        return "reject_too_few_test_samples"
    if pd.notna(row["test_top_stock_share"]) and float(row["test_top_stock_share"]) > 0.15:
        return "reject_stock_concentration"
    if row["rule_kind"] in {"counterevidence", "rejected_control"}:
        return "counter_or_control"
    if (
        pd.notna(row["test_minus_baseline_pos"])
        and float(row["test_minus_baseline_pos"]) >= 0.04
        and pd.notna(row["test_loss_delta"])
        and float(row["test_loss_delta"]) <= 0.0
        and pd.notna(row["valid_positive_20d_rate"])
    ):
        return "observe_candidate"
    return "reject_or_control"


def _safe_selector(rule: LocalRule, frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    mask = rule.selector(frame)
    return mask.reindex(frame.index).fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in columns:
        if column in frame:
            out = out.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return out


def _delta(value: Any, baseline: Any) -> Any:
    if pd.isna(value) or pd.isna(baseline):
        return pd.NA
    return round(float(value) - float(baseline), 4)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
