from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kline_channel_exploration import (  # noqa: E402
    DEFAULT_DAILY_DIR,
    DEFAULT_KLINE_FEATURE_CACHE_PATH,
    GT_SOURCES,
    prepare_frame,
)
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    DEFAULT_CORR_PEER_CACHE_PATH,
    DEFAULT_TUSHARE_PEER_CACHE_PATH,
    add_regime_features,
    merge_correlation_peer_features,
    merge_tushare_peer_features,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE_DIR = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_OUTPUT = MARKET_CACHE_DIR / "decision_point_table_v1.csv"
DEFAULT_SUMMARY = REPORT_DIR / "decision_point_table_v1_summary.md"

FUTURE_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "rating",
}

OUTPUT_COLUMNS = [
    "decision_point_id",
    "date",
    "code",
    "name",
    "time_block",
    "decision_frequency",
    "policy_profile",
    "decision_point_type",
    "decision_priority",
    "normal_or_key_point",
    "trigger_channel",
    "trigger_reason",
    "trigger_strength",
    "available_at",
    "source_ref_ids",
    "cooldown_group",
    "sampling_weight",
    "research_only",
    "not_investment_instruction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build time-safe decision-point table without DS/API calls.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--daily-feature-cache", default=str(DEFAULT_KLINE_FEATURE_CACHE_PATH))
    parser.add_argument("--corr-peer-cache", default=str(DEFAULT_CORR_PEER_CACHE_PATH))
    parser.add_argument("--tushare-peer-cache", default=str(DEFAULT_TUSHARE_PEER_CACHE_PATH))
    parser.add_argument("--skip-kline", action="store_true")
    parser.add_argument("--skip-corr-peer", action="store_true")
    parser.add_argument("--skip-tushare-peer", action="store_true")
    args = parser.parse_args()

    frame = load_feature_frame(
        daily_dir=Path(args.daily_dir),
        daily_feature_cache=None if args.skip_kline else Path(args.daily_feature_cache),
        corr_peer_cache=None if args.skip_corr_peer else Path(args.corr_peer_cache),
        tushare_peer_cache=None if args.skip_tushare_peer else Path(args.tushare_peer_cache),
    )
    table = build_decision_point_table(frame)
    assert_no_future_fields(table)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    write_summary(table, Path(args.summary), feature_frame=frame, output=output)

    print("A股研究Agent")
    print(f"rows={len(table)}")
    print(f"stocks={table['code'].nunique() if not table.empty else 0}")
    print(f"dates={table['date'].nunique() if not table.empty else 0}")
    print(f"output={output}")
    print(f"summary={args.summary}")


def load_feature_frame(
    *,
    daily_dir: Path,
    daily_feature_cache: Path | None,
    corr_peer_cache: Path | None,
    tushare_peer_cache: Path | None,
) -> pd.DataFrame:
    raw = load_ground_truth(GT_SOURCES)
    frame = prepare_frame(
        raw,
        daily_dir=daily_dir,
        daily_feature_cache=daily_feature_cache,
        rebuild_daily_feature_cache=False,
    )
    if corr_peer_cache is not None and corr_peer_cache.exists():
        corr = pd.read_csv(corr_peer_cache, dtype={"code": str}, low_memory=False)
        frame = merge_correlation_peer_features(frame, corr)
    if tushare_peer_cache is not None and tushare_peer_cache.exists():
        peer = pd.read_csv(tushare_peer_cache, dtype={"code": str}, low_memory=False)
        frame = merge_tushare_peer_features(frame, peer)
    frame = add_regime_features(frame)
    return _normalize_base_frame(frame)


def build_decision_point_table(frame: pd.DataFrame) -> pd.DataFrame:
    base = _normalize_base_frame(frame)
    pieces = [
        _scheduled_points(base, "twice_weekly"),
        _scheduled_points(base, "weekly_tuesday"),
        _scheduled_points(base, "weekly_friday"),
        _scheduled_points(base, "every_2_weeks"),
    ]
    key_points = _key_points(base)
    if not key_points.empty:
        pieces.append(key_points)
        pieces.append(_scheduled_plus_key(base, key_points))
    table = pd.concat([part for part in pieces if not part.empty], ignore_index=True)
    table = table[OUTPUT_COLUMNS].copy()
    table = table.drop_duplicates(["decision_point_id"]).sort_values(["date", "code", "decision_frequency", "decision_point_type"])
    return table.reset_index(drop=True)


def _normalize_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data = data[data["code"].str.fullmatch(r"\d{6}")].copy()
    if "time_block" not in data:
        data["time_block"] = data["date"].map(_time_block)
    data = data[data["time_block"].notna()].copy()
    if "name" not in data:
        data["name"] = ""
    data = data.sort_values(["code", "date"]).drop_duplicates(["date", "code"]).reset_index(drop=True)
    data["_week"] = data["date"].dt.strftime("%G-W%V")
    return data


def _time_block(value: Any) -> str | None:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return block
    return None


def _scheduled_points(base: pd.DataFrame, frequency: str) -> pd.DataFrame:
    data = base.copy()
    if frequency == "weekly_tuesday":
        data = data[data["date"].dt.weekday == 1].copy()
    elif frequency == "weekly_friday":
        data = data[data["date"].dt.weekday == 4].copy()
    elif frequency == "every_2_weeks":
        dates = sorted(data["date"].dropna().unique())
        selected = {date for idx, date in enumerate(dates) if idx % 4 == 0}
        data = data[data["date"].isin(selected)].copy()
    elif frequency != "twice_weekly":
        raise ValueError(f"unsupported scheduled frequency: {frequency}")
    if data.empty:
        return _empty_table()
    return _rows_from_base(
        data,
        decision_frequency=frequency,
        decision_point_type="scheduled",
        priority=0,
        trigger_channel="calendar",
        trigger_reason=f"scheduled_{frequency}",
        trigger_strength=0.0,
        sampling_weight=1.0,
    )


def _scheduled_plus_key(base: pd.DataFrame, key_points: pd.DataFrame) -> pd.DataFrame:
    scheduled = _scheduled_points(base, "twice_weekly")
    if scheduled.empty:
        return _empty_table()
    key_lookup = set(zip(key_points["date"], key_points["code"]))
    scheduled["_key_pair"] = list(zip(scheduled["date"], scheduled["code"]))
    scheduled.loc[scheduled["_key_pair"].isin(key_lookup), "normal_or_key_point"] = "key"
    scheduled.loc[scheduled["_key_pair"].isin(key_lookup), "sampling_weight"] = 2.0
    scheduled["decision_frequency"] = "scheduled_plus_key"
    scheduled["decision_point_id"] = scheduled.apply(
        lambda row: _decision_point_id(row["date"], row["code"], "scheduled_plus_key", row["decision_point_type"]),
        axis=1,
    )
    return scheduled.drop(columns=["_key_pair"])


def _key_points(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in base.iterrows():
        triggers = _trigger_reasons(row)
        if not triggers:
            continue
        priority = max(int(item["priority"]) for item in triggers)
        strength = max(float(item["strength"]) for item in triggers)
        channels = sorted({str(item["channel"]) for item in triggers})
        reasons = [str(item["reason"]) for item in triggers]
        rows.append(
            {
                "_source_index": idx,
                "priority": priority,
                "trigger_channel": "+".join(channels),
                "trigger_reason": ";".join(reasons[:8]),
                "trigger_strength": round(strength, 4),
            }
        )
    if not rows:
        return _empty_table()
    markers = pd.DataFrame(rows)
    data = base.loc[markers["_source_index"].tolist()].reset_index(drop=True)
    markers = markers.reset_index(drop=True)
    data["_priority"] = markers["priority"]
    data["_trigger_channel"] = markers["trigger_channel"]
    data["_trigger_reason"] = markers["trigger_reason"]
    data["_trigger_strength"] = markers["trigger_strength"]
    data = _apply_cooldown(data)
    return _rows_from_base(
        data,
        decision_frequency="key_points_only",
        decision_point_type="event_triggered",
        priority=None,
        trigger_channel=None,
        trigger_reason=None,
        trigger_strength=None,
        sampling_weight=None,
    )


def _apply_cooldown(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    major = data[pd.to_numeric(data["_priority"], errors="coerce").ge(3)].copy()
    ordinary = data[pd.to_numeric(data["_priority"], errors="coerce").lt(3)].copy()
    if not ordinary.empty:
        ordinary = ordinary.sort_values(["code", "_week", "_trigger_strength"], ascending=[True, True, False])
        ordinary = ordinary.drop_duplicates(["code", "_week"], keep="first")
    return pd.concat([major, ordinary], ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def _trigger_reasons(row: pd.Series) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    _add_threshold_trigger(triggers, row, "financial_report", "financial_report_materiality_score", ">=", 0.6, 3)
    _add_threshold_trigger(triggers, row, "financial_report", "financial_quality_risk_score", ">=", 0.6, 3)
    _add_threshold_trigger(triggers, row, "financial_report", "financial_surprise_score", "<=", -0.4, 3)
    if _num(row.get("financial_report_event_count")) > 0:
        _add_threshold_trigger(triggers, row, "financial_report", "financial_disclosure_quality_score", "<=", 0.3, 3)

    _add_threshold_trigger(triggers, row, "news_announcement", "news_warning_score", ">=", 0.6, 2)
    _add_threshold_trigger(triggers, row, "news_announcement", "news_conflict_intensity_30d", ">=", 1.0, 2)
    _add_threshold_trigger(triggers, row, "news_announcement", "announcement_materiality_score", ">=", 0.6, 2)
    _add_threshold_trigger(triggers, row, "source_quality", "news_missing_rate", ">=", 0.8, 1)
    _add_threshold_trigger(triggers, row, "source_quality", "news_timestamp_quality", "<=", 0.3, 1)
    _add_threshold_trigger(triggers, row, "source_quality", "news_evidence_quality", "<=", 0.3, 1)
    _add_threshold_trigger(triggers, row, "source_quality", "financial_report_missing_rate", ">=", 0.8, 1)

    _add_threshold_trigger(triggers, row, "price_structure", "prior_return_20d", "<=", -10.1231, 2)
    _add_threshold_trigger(triggers, row, "price_structure", "prior_return_20d", ">=", 60.0, 2)
    _add_threshold_trigger(triggers, row, "price_structure", "rsi14", ">=", 80.0, 2)
    _add_threshold_trigger(triggers, row, "price_structure", "rsi14", "<=", 25.0, 2)
    _add_threshold_trigger(triggers, row, "price_structure", "atr20_pct", ">=", 4.0, 2)
    _add_threshold_trigger(triggers, row, "price_structure", "drawdown60", "<=", -16.9912, 2)

    if _num(row.get("tushare_industry_positive_breadth_20d")) <= 0.4 and _num(row.get("tushare_industry_relative_return_20d")) < 0:
        triggers.append(_trigger("peer_context", "industry_breadth_weak_and_target_lagging", 2, 0.8))
    if _num(row.get("peer_group_positive_breadth_20d")) <= 0.4 and _num(row.get("peer_relative_to_group_20d")) < 0:
        triggers.append(_trigger("peer_context", "original_peer_breadth_weak_and_target_lagging", 2, 0.7))
    _add_threshold_trigger(triggers, row, "peer_context", "tushare_industry_news_attention_gap", "<=", -0.3, 2)
    return triggers


def _add_threshold_trigger(
    triggers: list[dict[str, Any]],
    row: pd.Series,
    channel: str,
    field: str,
    op: str,
    threshold: float,
    priority: int,
) -> None:
    value = _num(row.get(field))
    if pd.isna(value):
        return
    hit = value >= threshold if op == ">=" else value <= threshold
    if not hit:
        return
    distance = abs(float(value) - threshold)
    scale = max(abs(threshold), 1.0)
    strength = min(1.0, 0.5 + distance / scale)
    triggers.append(_trigger(channel, f"{field}{op}{threshold:g}", priority, strength))


def _trigger(channel: str, reason: str, priority: int, strength: float) -> dict[str, Any]:
    return {"channel": channel, "reason": reason, "priority": priority, "strength": strength}


def _rows_from_base(
    data: pd.DataFrame,
    *,
    decision_frequency: str,
    decision_point_type: str,
    priority: int | None,
    trigger_channel: str | None,
    trigger_reason: str | None,
    trigger_strength: float | None,
    sampling_weight: float | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": data["date"].dt.date.astype(str),
            "code": data["code"].astype(str).str.zfill(6),
            "name": data["name"].fillna("").astype(str),
            "time_block": data["time_block"].astype(str),
            "decision_frequency": decision_frequency,
            "policy_profile": "mid_horizon_research",
            "decision_point_type": decision_point_type,
            "decision_priority": data.get("_priority", priority),
            "normal_or_key_point": "key" if decision_point_type != "scheduled" else "normal",
            "trigger_channel": data.get("_trigger_channel", trigger_channel),
            "trigger_reason": data.get("_trigger_reason", trigger_reason),
            "trigger_strength": data.get("_trigger_strength", trigger_strength),
            "available_at": data["date"].dt.date.astype(str) + " 15:00",
            "source_ref_ids": data.apply(_source_refs, axis=1),
            "cooldown_group": data["code"].astype(str).str.zfill(6) + ":" + data["_week"].astype(str),
            "sampling_weight": data.get("_sampling_weight", sampling_weight),
            "research_only": True,
            "not_investment_instruction": True,
        }
    )
    out["decision_priority"] = pd.to_numeric(out["decision_priority"], errors="coerce").fillna(0).astype(int)
    out["trigger_strength"] = pd.to_numeric(out["trigger_strength"], errors="coerce").fillna(0.0).round(4)
    out["sampling_weight"] = pd.to_numeric(out["sampling_weight"], errors="coerce").fillna(
        out["decision_priority"].map(lambda value: 3.0 if int(value) >= 3 else (2.0 if int(value) > 0 else 1.0))
    )
    out["decision_point_id"] = out.apply(
        lambda row: _decision_point_id(row["date"], row["code"], row["decision_frequency"], row["decision_point_type"]),
        axis=1,
    )
    return out[OUTPUT_COLUMNS].copy()


def _source_refs(row: pd.Series) -> str:
    refs = ["local_gt_cache"]
    if _num(row.get("news_missing_rate")) < 1 or _num(row.get("news_count_30d")) > 0:
        refs.append("local_news_event_features")
    if _num(row.get("financial_report_event_count")) > 0 or _num(row.get("financial_report_missing_rate")) < 1:
        refs.append("local_financial_report_features")
    if any(col in row.index and pd.notna(row.get(col)) for col in ["kline_return_20d", "prior_return_20d", "rsi14"]):
        refs.append("local_daily_kline_features")
    if any(col in row.index and pd.notna(row.get(col)) for col in ["tushare_industry", "tushare_area", "peer_group_positive_breadth_20d"]):
        refs.append("local_peer_context_features")
    return ";".join(dict.fromkeys(refs))


def _decision_point_id(date_value: Any, code: Any, frequency: str, point_type: str) -> str:
    date_text = str(pd.to_datetime(date_value, errors="coerce").date())
    code_text = str(code).zfill(6)
    return f"{date_text}:{code_text}:{frequency}:{point_type}"


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def assert_no_future_fields(frame: pd.DataFrame) -> None:
    leaked = sorted(FUTURE_FIELDS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"decision point table contains future/result fields: {leaked}")


def write_summary(table: pd.DataFrame, path: Path, *, feature_frame: pd.DataFrame, output: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decision Point Table V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "把日线缓存中的 stock-date 行整理为 Agent 可消费的决策点表。该表不含未来收益、GT 或离线 label，只用于决定哪些样本进入后续量化工具训练、抽样和 DeepSeek evidence pack。",
        "",
        "## Outputs",
        "",
        f"- decision_point_table: `{output}`",
        f"- rows: `{len(table)}`",
        f"- unique_stocks: `{table['code'].nunique() if not table.empty else 0}`",
        f"- unique_dates: `{table['date'].nunique() if not table.empty else 0}`",
        f"- feature_source_rows: `{len(feature_frame)}`",
        "",
        "## Counts By Frequency",
        "",
        _table(table.groupby(["decision_frequency", "decision_point_type"]).size().reset_index(name="rows") if not table.empty else table),
        "",
        "## Counts By Time Block",
        "",
        _table(table.groupby(["time_block", "decision_frequency"]).size().reset_index(name="rows") if not table.empty else table),
        "",
        "## Key Trigger Channels",
        "",
        _table(
            table[table["normal_or_key_point"] == "key"]
            .groupby(["trigger_channel", "decision_priority"])
            .size()
            .reset_index(name="rows")
            .sort_values(["decision_priority", "rows"], ascending=[False, False])
            if not table.empty
            else table
        ),
        "",
        "## Boundary",
        "",
        "- 决策点表只使用当日及以前可见的本地缓存字段。",
        "- `return_5d/10d/20d`、`gt_status`、`gt_pass` 和离线 label 不进入该表。",
        "- `key_points_only` 是训练/抽样视角，不代表一定提高研究分级；后续必须用滚动时间块和消融验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()
