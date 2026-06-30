from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
TUSHARE_CACHE = ROOT / "data" / "date_generalization_cache" / "tushare_pro"
REPORT_DIR = ROOT / "reports" / "date_generalization"
MASTER_UNIVERSE_PATH = ROOT / "data" / "backtest_scale" / "a_share_codes.csv"
LOCAL_CACHE_DIR = ROOT / "data" / "backtest_scale_500"
GT_PATHS = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local market_5000 cache entrypoint from free/local sources.")
    parser.add_argument("--max-corr-stocks", type=int, default=500, help="Maximum local cached stocks used for historical-correlation TopK.")
    parser.add_argument("--corr-topk", type=int, default=10)
    args = parser.parse_args()

    MARKET_CACHE.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    master = _load_master_universe()
    local = _load_local_daily_coverage()
    gt = _load_ground_truth()

    stock_master = _merge_master_and_local(master, local, gt)
    related = _build_related_stock_graph(stock_master, args.max_corr_stocks, args.corr_topk)
    news_schema = _news_world_model_schema()
    news_event_schema = _news_event_schema()
    coverage = _coverage_rows(stock_master, related, news_schema)

    stock_master.to_csv(MARKET_CACHE / "stock_master_universe.csv", index=False, encoding="utf-8-sig")
    local.to_csv(MARKET_CACHE / "local_daily_coverage.csv", index=False, encoding="utf-8-sig")
    related.to_csv(MARKET_CACHE / "related_stock_graph.csv", index=False, encoding="utf-8-sig")
    news_schema.to_csv(MARKET_CACHE / "news_world_model_schema.csv", index=False, encoding="utf-8-sig")
    news_event_schema.to_csv(MARKET_CACHE / "news_event_schema.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(REPORT_DIR / "data_cache_5000_coverage.csv", index=False, encoding="utf-8-sig")
    _write_cache_manifest(stock_master, local, related, news_schema)
    _write_coverage_report(stock_master, local, related, coverage)

    print("A股研究Agent")
    print(f"market_5000 universe rows: {len(stock_master)}")
    print(f"local daily stocks: {int(stock_master['has_local_daily'].sum()) if 'has_local_daily' in stock_master else 0}")
    print(f"related graph rows: {len(related)}")


def _load_master_universe() -> pd.DataFrame:
    if not MASTER_UNIVERSE_PATH.exists():
        return pd.DataFrame(columns=["code", "name"])
    master = pd.read_csv(MASTER_UNIVERSE_PATH, dtype={"code": str}, low_memory=False)
    if "code" not in master:
        return pd.DataFrame(columns=["code", "name"])
    if "name" not in master:
        master["name"] = ""
    master["code"] = master["code"].astype(str).str.zfill(6)
    master["name"] = master["name"].astype(str)
    master = master.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    master["board"] = master["code"].map(_board)
    master["market"] = master["code"].map(_market)
    master["supported_a_share_code"] = master["code"].map(_is_supported_a_share_code)
    master["scaling_eligible"] = master["supported_a_share_code"] & ~master["name"].str.contains("ST|退|B", regex=True, na=False)
    master["universe_source_tier"] = "public_aggregator_cache"
    master["universe_source_name"] = "AKShare stock_info_a_code_name cached csv"
    return master


def _load_local_daily_coverage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not LOCAL_CACHE_DIR.exists():
        return pd.DataFrame(columns=["code", "local_name", "daily_rows", "daily_start", "daily_end", "local_cache_source"])
    for stock_dir in sorted(path for path in LOCAL_CACHE_DIR.iterdir() if path.is_dir()):
        code = stock_dir.name.zfill(6)
        metadata = _read_json_or_yaml(stock_dir / "metadata.json")
        daily_path = stock_dir / "daily.csv"
        row = {
            "code": code,
            "local_name": metadata.get("name") or "",
            "daily_rows": 0,
            "daily_start": "",
            "daily_end": "",
            "local_cache_source": metadata.get("source") or "local_backtest_cache",
            "financial_status": metadata.get("financial_note") or "unknown",
            "has_news_json": (stock_dir / "news.json").exists(),
        }
        if daily_path.exists():
            try:
                daily = pd.read_csv(daily_path, usecols=["date"], low_memory=False)
                dates = pd.to_datetime(daily["date"], errors="coerce").dropna()
                row["daily_rows"] = int(len(dates))
                row["daily_start"] = dates.min().date().isoformat() if not dates.empty else ""
                row["daily_end"] = dates.max().date().isoformat() if not dates.empty else ""
            except Exception as exc:  # pragma: no cover - corrupt cache should be reported, not fatal.
                row["local_cache_source"] = f"read_error:{type(exc).__name__}"
        rows.append(row)
    return pd.DataFrame(rows)


def _load_ground_truth() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False, dtype={"code": str}) for path in GT_PATHS if path.exists()]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def _merge_master_and_local(master: pd.DataFrame, local: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        master = local[["code", "local_name"]].rename(columns={"local_name": "name"}).copy()
        master["board"] = master["code"].map(_board)
        master["market"] = master["code"].map(_market)
        master["supported_a_share_code"] = master["code"].map(_is_supported_a_share_code)
        master["scaling_eligible"] = master["supported_a_share_code"]
        master["universe_source_tier"] = "local_cache"
        master["universe_source_name"] = "data/backtest_scale_500"
    sector = _sector_summary(gt)
    news = _news_summary(gt)
    merged = master.merge(local, on="code", how="left")
    merged = merged.merge(sector, on="code", how="left")
    merged = merged.merge(news, on="code", how="left")
    if "local_name" in merged:
        merged["name"] = merged["name"].where(merged["name"].astype(str).str.len() > 0, merged["local_name"].fillna(""))
    merged["daily_rows"] = merged["daily_rows"].fillna(0).astype(int)
    merged["has_local_daily"] = merged["daily_rows"] > 0
    merged["has_news_json"] = merged.get("has_news_json", False).fillna(False).astype(bool)
    merged["sector_group"] = merged["sector_group"].fillna("unknown")
    merged["region"] = "unknown"
    merged["concept_tags"] = "unknown"
    merged["local_data_status"] = merged["has_local_daily"].map({True: "daily_cache_ready", False: "metadata_only"})
    return merged


def _sector_summary(gt: pd.DataFrame) -> pd.DataFrame:
    if gt.empty or "sector_group" not in gt:
        return pd.DataFrame(columns=["code", "sector_group", "gt_rows", "gt_evaluated_rows"])
    grouped = gt.groupby("code")
    return grouped.agg(
        sector_group=("sector_group", "first"),
        gt_rows=("date", "count"),
        gt_evaluated_rows=("gt_status", lambda value: int(value.astype(str).eq("evaluated").sum())),
    ).reset_index()


def _news_summary(gt: pd.DataFrame) -> pd.DataFrame:
    if gt.empty:
        return pd.DataFrame(columns=["code", "news_rows", "peer_feature_rows"])
    grouped = gt.groupby("code")
    return grouped.agg(
        news_rows=("news_count_30d", lambda value: int((pd.to_numeric(value, errors="coerce").fillna(0) > 0).sum()) if "news_count_30d" in gt else 0),
        peer_feature_rows=("peer_group_size", lambda value: int(pd.to_numeric(value, errors="coerce").notna().sum()) if "peer_group_size" in gt else 0),
    ).reset_index()


def _build_related_stock_graph(stock_master: pd.DataFrame, max_corr_stocks: int, corr_topk: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    local = stock_master[stock_master["has_local_daily"]].copy() if "has_local_daily" in stock_master else pd.DataFrame()
    for relation, field in [("same_sector_group", "sector_group"), ("same_board", "board")]:
        if field not in local:
            continue
        for _, row in local.iterrows():
            code = row["code"]
            value = row.get(field)
            peers = local[(local[field] == value) & (local["code"] != code)]["code"].astype(str).head(20).tolist()
            rows.append(
                {
                    "code": code,
                    "relation_type": relation,
                    "relation_value": value,
                    "related_codes": ";".join(peers),
                    "related_count": len(peers),
                    "source_tier": "local_cache",
                    "status": "ready" if peers else "insufficient_peers",
                }
            )
    corr_rows = _historical_corr_topk(local["code"].astype(str).head(max_corr_stocks).tolist(), corr_topk)
    rows.extend(corr_rows)
    if not rows:
        return pd.DataFrame(columns=["code", "relation_type", "relation_value", "related_codes", "related_count", "source_tier", "status"])
    return pd.DataFrame(rows)


def _historical_corr_topk(codes: list[str], topk: int) -> list[dict[str, Any]]:
    prices = []
    for code in codes:
        daily_path = LOCAL_CACHE_DIR / code / "daily.csv"
        if not daily_path.exists():
            continue
        try:
            daily = pd.read_csv(daily_path, usecols=["date", "close"], low_memory=False)
        except Exception:
            continue
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
        series = daily.dropna(subset=["date", "close"]).sort_values("date").set_index("date")["close"].pct_change().rename(code)
        prices.append(series)
    if len(prices) < 2:
        return []
    returns = pd.concat(prices, axis=1).dropna(how="all").tail(252)
    corr = returns.corr(min_periods=60)
    rows = []
    for code in corr.columns:
        peers = corr[code].drop(labels=[code], errors="ignore").dropna().sort_values(ascending=False).head(topk)
        rows.append(
            {
                "code": code,
                "relation_type": "historical_corr_topk",
                "relation_value": "last_252_trade_days",
                "related_codes": ";".join(peers.index.astype(str).tolist()),
                "related_count": int(len(peers)),
                "source_tier": "local_cache",
                "status": "ready" if len(peers) else "insufficient_overlap",
            }
        )
    return rows


def _news_world_model_schema() -> pd.DataFrame:
    fields = [
        ("self_news_intensity", "股票自身新闻/公告强度", "numeric", "self_count_30d normalized by local history"),
        ("peer_news_intensity", "同行新闻/公告强度", "numeric", "peer mean count/materiality"),
        ("policy_background_score", "政策背景", "numeric", "policy event score from official/public sources"),
        ("region_background_score", "地域背景", "numeric", "region policy/event score; capped unless self event exists"),
        ("self_vs_peer_attention_gap", "自身相对同行关注差", "numeric", "self_news_intensity - peer_news_intensity"),
        ("peer_active_self_silent_flag", "同行活跃但自身沉默", "bool", "peer high and self low"),
        ("news_warning_score", "风险预警", "numeric", "regulatory/legal/financing/holding-change risk"),
        ("news_opportunity_score", "机会信号", "numeric", "orders/capacity/technology/product/policy opportunity"),
        ("news_evidence_quality", "证据质量", "numeric", "official source and timestamp quality weighted"),
        ("news_missing_rate", "新闻缺失率", "numeric", "missing source slots / expected source slots"),
        ("news_timestamp_quality", "时间戳质量", "numeric", "available_at quality; must be <= decision time"),
        ("news_peer_diffusion_score", "新闻从同行扩散到目标股的可能性", "numeric", "peer risk/opportunity co-occurrence"),
        ("official_confirmation_score", "官方确认度", "numeric", "official_count/(official_count+public_count)"),
        ("community_attention_score", "社区关注度", "numeric", "public/community count zscore capped"),
        ("community_crowding_risk", "社区拥挤反证", "numeric", "high positive crowding + overheat"),
        ("announcement_materiality_score", "公告重要性", "numeric", "max official announcement materiality"),
    ]
    return pd.DataFrame(
        [
            {
                "feature_name": name,
                "description": desc,
                "type": typ,
                "calculation_hint": hint,
                "leakage_guard": "only use records with available_at <= decision_time",
            }
            for name, desc, typ, hint in fields
        ]
    )


def _news_event_schema() -> pd.DataFrame:
    fields = [
        ("event_id", "事件唯一 ID", "string", "hash(interface, ts_code, available_at, title, url)"),
        ("ts_code", "Tushare 股票代码；市场级新闻可为空", "string", "from announcement/news source"),
        ("event_date", "事件日期", "date", "derived from available_at"),
        ("event_time", "原始事件时间", "datetime", "exact time if available"),
        ("available_at", "可用于决策的时间戳", "datetime", "must be <= decision_time before entering evidence pack"),
        ("available_at_guard_status", "时间戳质量", "enum", "exact_time/date_only_close_assumed/missing_time"),
        ("interface", "来源接口", "string", "anns_d/news/major_news/etc."),
        ("event_type", "事件类型", "enum", "official_announcement/market_news/community_seed/etc."),
        ("title", "标题", "string", "raw title"),
        ("risk_score", "风险事件分", "0-1", "keyword/materiality classifier"),
        ("opportunity_score", "机会事件分", "0-1", "keyword/materiality classifier"),
        ("policy_score", "政策背景分", "-1..1", "policy keyword classifier"),
        ("official_confirmation_score", "官方确认度", "0-1", "source type and timestamp quality"),
        ("announcement_materiality_score", "公告重要性", "0-1", "official announcement materiality"),
        ("news_evidence_quality", "证据质量", "0-1", "official source and timestamp quality"),
    ]
    return pd.DataFrame(
        [
            {
                "field_name": name,
                "description": desc,
                "type": typ,
                "calculation_hint": hint,
                "leakage_guard": "only use records with available_at <= decision_time",
            }
            for name, desc, typ, hint in fields
        ]
    )


def _coverage_rows(stock_master: pd.DataFrame, related: pd.DataFrame, news_schema: pd.DataFrame) -> pd.DataFrame:
    eligible = int(stock_master["scaling_eligible"].sum()) if "scaling_eligible" in stock_master else len(stock_master)
    local_daily = int(stock_master["has_local_daily"].sum()) if "has_local_daily" in stock_master else 0
    tushare = _tushare_cache_status()
    disclosure_count = _financial_disclosure_count()
    news_events = _news_event_count()
    news_features = _news_event_feature_count()
    ann_cap_risk = int(tushare.get("possible_row_cap_requests", 0) or 0)
    rows = [
        {
            "component": "raw_a_share_metadata",
            "target": "full A-share metadata entrypoint",
            "current_count": int(len(stock_master)),
            "status": "metadata_ready" if len(stock_master) >= 5000 else "partial_metadata",
            "source": "data/backtest_scale/a_share_codes.csv",
            "next_action": "keep raw list as candidate source and apply eligibility filters per experiment",
        },
        {
            "component": "scaling_eligible_universe",
            "target": "supported active-like candidates after ST/delist/B filtering",
            "current_count": eligible,
            "status": "near_5000_after_quality_filter" if eligible >= 4500 else "partial_after_quality_filter",
            "source": "data/backtest_scale/a_share_codes.csv",
            "next_action": "use this as candidate universe, then expand local daily/financial/news cache in shards",
        },
        {
            "component": "daily_price_features",
            "target": "daily bars for train/test/scaling stocks",
            "current_count": local_daily,
            "status": "partial_local_cache",
            "source": "data/backtest_scale_500/*/daily.csv",
            "next_action": "cache more shards before 1000+ stock strategy search",
        },
        {
            "component": "related_stock_graph",
            "target": "same sector, same board, same region/concept, historical corr TopK, news co-occurrence",
            "current_count": int(len(related)),
            "status": "partial_sector_board_corr_ready",
            "source": "local metadata and local daily cache",
            "next_action": "add region/concept/news co-occurrence after paid_standardized cache adapter is ready",
        },
        {
            "component": "news_world_model_schema",
            "target": "self, peer, policy, region, risk, opportunity, quality, missingness",
            "current_count": int(len(news_schema)),
            "status": "schema_ready_data_sparse",
            "source": "local schema",
            "next_action": "backfill announcement/news event tables with timestamp guards",
        },
        {
            "component": "news_announcement_events",
            "target": "available_at-safe raw news/announcement event table",
            "current_count": news_events,
            "status": "event_cache_ready_with_cap_risk" if ann_cap_risk and news_events > 0 else "event_cache_ready" if news_events > 0 else "schema_ready_cache_missing",
            "source": "data/date_generalization_cache/market_5000/combined_news_event_table.csv",
            "next_action": "keep expanding local/official announcement sources; split dense anns_d dates and keep empty_ok/possible row-cap coverage visible",
        },
        {
            "component": "news_world_model_event_features",
            "target": "stock-date news features derived only from available_at-safe events",
            "current_count": news_features,
            "status": "feature_cache_ready" if news_features > 0 else "event_features_missing",
            "source": "data/date_generalization_cache/market_5000/combined_news_world_model_event_features.csv",
            "next_action": "join into evidence pack only after filtering available_at <= decision_time",
        },
        {
            "component": "joined_gt_combined_news_cache",
            "target": "cached ground-truth table with combined news as-of join for repeated strategy rounds",
            "current_count": _joined_gt_cache_count(),
            "status": "cache_ready" if _joined_gt_cache_count() > 0 else "cache_missing",
            "source": "data/date_generalization_cache/market_5000/joined_ground_truth_combined_news.csv",
            "next_action": "rebuild automatically when GT sources or combined news feature table changes",
        },
        {
            "component": "financial_disclosure_dates",
            "target": "report period and actual disclosure date",
            "current_count": disclosure_count,
            "status": "ready" if disclosure_count > 0 else "adapter_ready_cache_missing",
            "source": "data/date_generalization_cache/tushare_pro/derived/financial_disclosure_calendar.csv",
            "next_action": "run scripts/build_tushare_cache.py --execute --interfaces stock_basic fina_indicator --max-stocks <N> to populate disclosure-safe financial cache",
        },
        {
            "component": "paid_standardized_tushare",
            "target": "authorized offline standardized source",
            "current_count": tushare["rows"],
            "status": tushare["status"],
            "source": "data/date_generalization_cache/tushare_pro/",
            "next_action": tushare["next_action"],
        },
    ]
    return pd.DataFrame(rows)


def _write_cache_manifest(stock_master: pd.DataFrame, local: pd.DataFrame, related: pd.DataFrame, news_schema: pd.DataFrame) -> None:
    manifest = {
        "research_only": True,
        "no_broker": True,
        "no_auto_trade": True,
        "paid_standardized_allowed": True,
        "paid_standardized_cache_available": _tushare_cache_status()["rows"] > 0,
        "bandwidth_note": "No external download is performed by default; future downloads must stay <= 4MB/s.",
        "files": {
            "stock_master_universe.csv": len(stock_master),
            "local_daily_coverage.csv": len(local),
            "related_stock_graph.csv": len(related),
            "news_world_model_schema.csv": len(news_schema),
            "news_event_schema.csv": _news_event_schema_count(),
            "combined_news_event_table.csv": _news_event_count(),
            "combined_news_world_model_event_features.csv": _news_event_feature_count(),
            "joined_ground_truth_combined_news.csv": _joined_gt_cache_count(),
        },
        "joined_ground_truth_cache_size_mb": _joined_gt_cache_size_mb(),
    }
    (MARKET_CACHE / "cache_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_coverage_report(stock_master: pd.DataFrame, local: pd.DataFrame, related: pd.DataFrame, coverage: pd.DataFrame) -> None:
    eligible = int(stock_master["scaling_eligible"].sum()) if "scaling_eligible" in stock_master else len(stock_master)
    local_daily = int(stock_master["has_local_daily"].sum()) if "has_local_daily" in stock_master else 0
    lines = [
        "# 5000 股数据底座覆盖报告",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        f"- 全 A 元数据层原始股票数：{len(stock_master)}。",
        f"- 质量过滤后 eligible 股票数：{eligible}。",
        f"- 当前已有本地日线/GT 缓存股票数：{local_daily}。",
        f"- 当前相关股票图谱行数：{len(related)}。",
        f"- 当前 joined GT + combined news 缓存行数：{_joined_gt_cache_count()}，大小约 {_joined_gt_cache_size_mb()} MB。",
        _tushare_summary_line(),
        "- 回测决策点只能读本地缓存，不能临时请求未来或未披露数据。",
        "",
        "## 覆盖表",
        "",
        _markdown_table(coverage),
        "",
        "## 已生成文件",
        "",
        "- `data/date_generalization_cache/market_5000/stock_master_universe.csv`",
        "- `data/date_generalization_cache/market_5000/local_daily_coverage.csv`",
        "- `data/date_generalization_cache/market_5000/related_stock_graph.csv`",
        "- `data/date_generalization_cache/market_5000/news_world_model_schema.csv`",
        "- `data/date_generalization_cache/market_5000/news_event_schema.csv`",
        "- `data/date_generalization_cache/market_5000/cache_manifest.json`",
        "",
        "## 执行入口",
        "",
        "```bash",
        ".conda/stock-agent/bin/python scripts/build_market_5000_cache.py",
        "```",
        "",
        "## 缺口",
        "",
        "- 日线、财务披露日、新闻/公告事件表尚未覆盖 5000 股。",
        "- 地域、概念、新闻共现 TopK 需要下一轮数据源补齐。",
        "- 缺披露日的财务字段不得进入 walk-forward 决策。",
    ]
    (REPORT_DIR / "data_cache_5000_coverage.md").write_text("\n".join(lines), encoding="utf-8")


def _tushare_cache_status() -> dict[str, Any]:
    coverage_path = REPORT_DIR / "tushare_data_coverage.csv"
    manifest_path = TUSHARE_CACHE / "cache_manifest.json"
    rows = 0
    ok_requests = 0
    empty_ok_requests = 0
    possible_row_cap_requests = 0
    partial_empty_interfaces: list[str] = []
    if coverage_path.exists():
        try:
            coverage = pd.read_csv(coverage_path)
            rows = int(pd.to_numeric(coverage.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            ok_requests = int(pd.to_numeric(coverage.get("ok_requests", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            empty_ok_requests = int(pd.to_numeric(coverage.get("empty_ok_requests", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            possible_row_cap_requests = int(pd.to_numeric(coverage.get("possible_row_cap_requests", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            if "status" in coverage and "interface" in coverage:
                partial_empty_interfaces = (
                    coverage[coverage["status"].astype(str).str.contains("empty", case=False, na=False)]["interface"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
        except Exception:
            rows = 0
            ok_requests = 0
            empty_ok_requests = 0
            possible_row_cap_requests = 0
            partial_empty_interfaces = []
    dry_run = True
    if manifest_path.exists():
        try:
            dry_run = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("dry_run", True))
        except Exception:
            dry_run = True
    if rows > 0 and ok_requests > 0 and not dry_run:
        return {
            "rows": rows,
            "status": "smoke_cache_ready",
            "empty_ok_requests": empty_ok_requests,
            "possible_row_cap_requests": possible_row_cap_requests,
            "partial_empty_interfaces": partial_empty_interfaces,
            "next_action": "extend paid_standardized cache in bounded shards: trade_cal, daily, adj_factor, fina_indicator, then derive disclosure calendar",
        }
    if manifest_path.exists():
        return {
            "rows": rows,
            "status": "adapter_ready_dry_run_or_empty",
            "empty_ok_requests": empty_ok_requests,
            "possible_row_cap_requests": possible_row_cap_requests,
            "partial_empty_interfaces": partial_empty_interfaces,
            "next_action": "run a bounded --execute smoke or shard; keep token out of outputs",
        }
    return {
        "rows": 0,
        "status": "adapter_not_run",
        "empty_ok_requests": 0,
        "possible_row_cap_requests": 0,
        "partial_empty_interfaces": [],
        "next_action": "run scripts/build_tushare_cache.py dry-run first, then bounded --execute shard",
    }


def _financial_disclosure_count() -> int:
    path = TUSHARE_CACHE / "derived" / "financial_disclosure_calendar.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, usecols=["ts_code", "report_period", "disclosure_date"])))
    except Exception:
        return 0


def _news_event_schema_count() -> int:
    path = MARKET_CACHE / "news_event_schema.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _news_event_count() -> int:
    path = MARKET_CACHE / "combined_news_event_table.csv"
    if not path.exists():
        path = TUSHARE_CACHE / "derived" / "news_event_table.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _news_event_feature_count() -> int:
    path = MARKET_CACHE / "combined_news_world_model_event_features.csv"
    if not path.exists():
        path = TUSHARE_CACHE / "derived" / "news_world_model_event_features.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return 0


def _joined_gt_cache_count() -> int:
    path = MARKET_CACHE / "joined_ground_truth_combined_news.csv"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, usecols=[0], low_memory=False)))
    except Exception:
        return 0


def _joined_gt_cache_size_mb() -> float:
    path = MARKET_CACHE / "joined_ground_truth_combined_news.csv"
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / 1024 / 1024, 2)


def _tushare_summary_line() -> str:
    status = _tushare_cache_status()
    if status["rows"] > 0:
        suffix = ""
        if status.get("empty_ok_requests"):
            interfaces = ",".join(status.get("partial_empty_interfaces") or [])
            suffix = f"；另有 {status['empty_ok_requests']} 次空返回需继续换窗口/来源验证"
            if interfaces:
                suffix += f"（{interfaces}）"
        if status.get("possible_row_cap_requests"):
            suffix += f"；{status['possible_row_cap_requests']} 个公告分片可能触及接口行数上限，不能直接宣称完整覆盖"
        return f"- Tushare Pro paid_standardized smoke cache 已存在，当前缓存行数：{status['rows']}；新闻/公告事件行数：{_news_event_count()}{suffix}。"
    return "- 当前 Tushare Pro paid_standardized 缓存尚未写入真实行；已有 adapter/dry-run 入口。"


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _is_supported_a_share_code(code: str) -> bool:
    return str(code).startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688"))


def _board(code: str) -> str:
    code = str(code)
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("000", "001", "002", "003")):
        return "深市主板"
    if code.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    return "其他"


def _market(code: str) -> str:
    code = str(code)
    if code.startswith(("600", "601", "603", "605", "688")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    return "OTHER"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
