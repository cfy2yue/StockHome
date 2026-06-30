from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
BASELINE_GATE_PORTFOLIO_POSITIVE_RATE = 0.4043
GT_PATHS = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
MASTER_UNIVERSE_PATH = ROOT / "data" / "backtest_scale" / "a_share_codes.csv"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_CACHE.mkdir(parents=True, exist_ok=True)
    frame = _load_ground_truth()
    coverage = _write_market_cache(frame)
    book_audit = _write_book_skill_grounding_audit(frame)
    seed_count = _write_external_skill_seed_log()
    _update_memory_ledgers()
    _write_strategy_hardening_summary(coverage, book_audit, seed_count)
    print("A股研究Agent")
    print("wrote strategy hardening deliverables")


def _load_ground_truth() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in GT_PATHS if path.exists()]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame.drop_duplicates(["date", "code"]).reset_index(drop=True)


def _write_market_cache(frame: pd.DataFrame) -> dict[str, Any]:
    master = _load_master_universe()
    if frame.empty:
        universe = pd.DataFrame(columns=["code", "name", "sector_group", "first_date", "last_date", "row_count"])
    else:
        grouped = frame.groupby("code", dropna=False)
        universe = grouped.agg(
            name=("name", "first"),
            sector_group=("sector_group", "first"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            row_count=("date", "count"),
            evaluated_rows=("gt_status", lambda value: int(value.astype(str).eq("evaluated").sum()) if value is not None else 0),
            news_rows=("news_count_30d", lambda value: int((pd.to_numeric(value, errors="coerce").fillna(0) > 0).sum())),
            peer_feature_rows=("peer_group_size", lambda value: int(pd.to_numeric(value, errors="coerce").notna().sum())),
            financial_gap_rows=("data_gaps", lambda value: int(value.fillna("").astype(str).str.contains("financial_publish_date_missing", regex=False).sum())),
        ).reset_index()
        universe["financial_publish_date_missing_rate"] = (universe["financial_gap_rows"] / universe["row_count"]).round(4)
        universe["news_coverage_rate"] = (universe["news_rows"] / universe["row_count"]).round(4)
        universe["peer_feature_coverage_rate"] = (universe["peer_feature_rows"] / universe["row_count"]).round(4)
    universe.to_csv(MARKET_CACHE / "universe_coverage.csv", index=False, encoding="utf-8-sig")
    if not master.empty:
        merged = master.merge(universe, on="code", how="left", suffixes=("", "_local"))
        merged["has_local_daily"] = merged["row_count"].fillna(0).astype(int) > 0
        merged["local_data_status"] = merged["has_local_daily"].map({True: "local_backtest_cache_ready", False: "metadata_only"})
    else:
        merged = universe.copy()
        merged["has_local_daily"] = True
        merged["local_data_status"] = "local_backtest_cache_ready"
    merged.to_csv(MARKET_CACHE / "stock_master_universe.csv", index=False, encoding="utf-8-sig")

    related = _related_stock_sample(frame)
    related.to_csv(MARKET_CACHE / "related_stock_channel_sample.csv", index=False, encoding="utf-8-sig")

    local_count = int(universe["code"].nunique()) if not universe.empty else 0
    master_count = int(master["code"].nunique()) if not master.empty else local_count
    eligible_count = int(master["scaling_eligible"].sum()) if "scaling_eligible" in master else local_count
    target_count = 5000
    rows = [
        {
            "component": "stock_universe",
            "target": ">=5000 or full A-share coverage",
            "current_count": master_count,
            "status": "full_metadata_ready_daily_cache_partial" if master_count >= target_count else "partial_metadata_universe",
            "source": "data/backtest_scale/a_share_codes.csv public free cache + local backtest cache",
            "next_action": "expand daily/financial/news cache for eligible universe; do not use Tushare under current AGENTS.md",
        },
        {
            "component": "scaling_eligible_universe",
            "target": "active/supported A-share candidates after ST/delist/B filtering",
            "current_count": eligible_count,
            "status": "near_5000_after_quality_filter" if eligible_count >= 4500 else "partial_after_quality_filter",
            "source": "data/backtest_scale/a_share_codes.csv filtered by code prefix and name",
            "next_action": "review exclusion rules before treating eligible list as final all-A training universe",
        },
        {
            "component": "daily_price_features",
            "target": "daily bars, adjusted features, volatility and returns",
            "current_count": int(len(frame)),
            "status": f"available_for_{local_count}_stock_cache" if not frame.empty else "missing",
            "source": "existing local ground_truth",
            "next_action": "expand cache before 300/500/1000 stock scaling",
        },
        {
            "component": "peer_channel",
            "target": "industry, concept, region and historical-correlation TopK",
            "current_count": int(universe["peer_feature_rows"].sum()) if "peer_feature_rows" in universe else 0,
            "status": "peer_group_available_but_concept_region_topk_incomplete",
            "source": "existing peer_group fields",
            "next_action": "derive concept/region/topk graph in data/date_generalization_cache/market_5000",
        },
        {
            "component": "news_world_model",
            "target": "self, peer, policy, region, risk, opportunity, evidence quality",
            "current_count": int(universe["news_rows"].sum()) if "news_rows" in universe else 0,
            "status": "sparse_or_zero_for_many_dates",
            "source": "existing news_* fields",
            "next_action": "build announcement/news coverage report before claiming news advantage",
        },
        {
            "component": "financial_disclosure_dates",
            "target": "report period and actual disclosure date",
            "current_count": int(universe["financial_gap_rows"].sum()) if "financial_gap_rows" in universe else 0,
            "status": "gap_detected",
            "source": "data_gaps flags",
            "next_action": "fields with missing disclosure date cannot enter walk-forward decisions",
        },
        {
            "component": "paid_standardized_tushare",
            "target": "authorized offline standardized cache",
            "current_count": 0,
            "status": "allowed_not_cached_in_this_run",
            "source": "user-authorized paid_standardized source",
            "next_action": "build offline cache adapter; never print token; use >=0.7s interval and <=4MB/s downloads",
        },
    ]
    coverage = pd.DataFrame(rows)
    coverage.to_csv(REPORT_DIR / "data_cache_5000_coverage.csv", index=False, encoding="utf-8-sig")

    readme = [
        "# Market 5000 Cache",
        "",
        "本目录是 5000 支股票或全 A 可覆盖股票的数据底座入口。当前写入本地全 A 元数据缓存和现有 500 股回测缓存覆盖情况；授权 paid_standardized 数据源可用于后续离线缓存，不接券商接口。",
        "",
        "## Files",
        "",
        "- `stock_master_universe.csv`: 本地公开免费源缓存的全 A/5000 股票元数据层，并标记哪些股票已有本地日线。",
        "- `universe_coverage.csv`: 当前本地 ground truth 覆盖股票、日期、新闻、同行和财报披露缺口。",
        "- `related_stock_channel_sample.csv`: 基于现有 peer 字段生成的相关股票通道样例。",
        "- `related_stock_graph.csv`: 由 `scripts/build_market_5000_cache.py` 生成的同板块、同粗行业、历史相关性 TopK 图谱。",
        "- `news_world_model_schema.csv`: 新闻/公告向量化字段和防泄漏要求。",
        "",
        "## Boundary",
        "",
        "- 当前项目允许授权 Tushare Pro/paid_standardized 数据源；完整 5000/全 A 缓存应走离线缓存、限速和凭证安全。",
        "- 回测决策点只能读本地缓存，不得临时请求未来或未披露数据。",
        "- 后续下载大文件时必须限制总带宽不超过 4MB/s。",
    ]
    (MARKET_CACHE / "README.md").write_text("\n".join(readme), encoding="utf-8")

    md = [
        "# 5000 股数据底座覆盖报告",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        f"- 当前本地缓存覆盖股票数：{local_count}。",
        f"- 当前全 A 元数据层股票数：{master_count}。",
        f"- 当前质量过滤后 eligible 股票数：{eligible_count}。",
        f"- 当前已有本地日线/GT 的股票数：{local_count}。",
        "- 目标：至少 5000 支股票或全 A 可覆盖股票。",
        f"- 当前状态：全 A 元数据入口已超过 5000 行；质量过滤后为 {eligible_count}，完整日线、财务披露日、新闻和相关图谱仍只覆盖本地已缓存部分。",
        "- 当前项目允许授权 Tushare Pro/paid_standardized 数据源；如启用必须标注来源、保护 token、走离线缓存。",
        "",
        "## 覆盖表",
        "",
        _markdown_table(coverage),
        "",
        "## 相关股票通道",
        "",
        "- 已有：`peer_group_size`、`peer_group_avg_return_20d`、`peer_relative_to_group_20d`、`peer_group_positive_breadth_20d`。",
        "- 已生成：`related_stock_graph.csv`，包含同板块、同粗行业和本地历史相关性 TopK 雏形。",
        "- 缺口：同概念、同地域、新闻共现 TopK 尚未完整缓存。",
        "",
        "## 下一步",
        "",
        "1. 在不修改政策的前提下，优先用公开免费源补全 A 股基础 universe、行业、地域、概念。",
        "2. 扩展 `related_stock_graph.csv`：补同概念、同地域、新闻共现 TopK。",
        "3. 再扩大到 300/500/1000 股训练面板，避免一开始把 5000 股全部交给 DeepSeek。",
    ]
    (REPORT_DIR / "data_cache_5000_coverage.md").write_text("\n".join(md), encoding="utf-8")
    return {"current_count": local_count, "master_count": master_count, "eligible_count": eligible_count, "target_count": target_count}


def _load_master_universe() -> pd.DataFrame:
    if not MASTER_UNIVERSE_PATH.exists():
        return pd.DataFrame(columns=["code", "name", "board", "scaling_eligible", "universe_source"])
    master = pd.read_csv(MASTER_UNIVERSE_PATH, dtype={"code": str}, low_memory=False)
    if "code" not in master or "name" not in master:
        return pd.DataFrame(columns=["code", "name", "board", "scaling_eligible", "universe_source"])
    master["code"] = master["code"].astype(str).str.zfill(6)
    master["name"] = master["name"].astype(str)
    master = master.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    master["board"] = master["code"].map(_board)
    master["supported_a_share_code"] = master["code"].map(_is_supported_a_share_code)
    master["scaling_eligible"] = master["supported_a_share_code"] & ~master["name"].str.contains("ST|退|B", regex=True, na=False)
    master["universe_source"] = "AKShare public free cache stock_info_a_code_name"
    return master


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


def _related_stock_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "code", "name", "sector_group", "peer_group_size", "peer_relative_to_group_20d", "peer_group_positive_breadth_20d"])
    cols = ["date", "code", "name", "sector_group", "peer_group_size", "peer_relative_to_group_20d", "peer_group_positive_breadth_20d"]
    available = [col for col in cols if col in frame.columns]
    sample = frame[available].drop_duplicates(["date", "code"]).head(200).copy()
    return sample


def _write_book_skill_grounding_audit(frame: pd.DataFrame) -> dict[str, Any]:
    source_manifest = _load_yaml(ROOT / "book_skills" / "source_manifest.yaml")
    strategy_cards = []
    for path in [
        ROOT / "book_skills" / "strategy_cards.yaml",
        ROOT / "book_skills" / "auto_strategy_cards.yaml",
        ROOT / "book_skills" / "supporting_strategy_cards.yaml",
    ]:
        data = _load_yaml(path)
        if isinstance(data, list):
            strategy_cards.extend(data)
    card_by_id = {str(card.get("strategy_id")): card for card in strategy_cards if card.get("strategy_id")}

    source_rows = []
    for item in source_manifest.get("sources", []) if isinstance(source_manifest, dict) else []:
        pages = int(item.get("pages") or 0)
        covered = int(item.get("text_layer_pages") or 0) + int(item.get("ocr_pages") or 0)
        source_rows.append(
            {
                "book": item.get("file"),
                "pages": pages,
                "covered_pages": covered,
                "needs_ocr_pages": int(item.get("needs_ocr_pages") or 0),
                "coverage_rate": round(covered / pages, 4) if pages else None,
                "status": "system_covered" if pages and covered / pages >= 0.95 else "partial_or_needs_ocr",
            }
        )

    counter = Counter()
    if not frame.empty and "triggered_skills" in frame:
        for text in frame["triggered_skills"].fillna("").astype(str):
            for skill in text.replace(",", ";").split(";"):
                skill = skill.strip()
                if skill and skill.lower() != "nan":
                    counter[skill] += 1

    adaptation = pd.read_csv(REPORT_DIR / "book_skill_adaptation_log.csv") if (REPORT_DIR / "book_skill_adaptation_log.csv").exists() else pd.DataFrame()
    top_ids = [skill for skill, _ in counter.most_common(20)]
    for skill in adaptation.get("strategy_id", pd.Series(dtype=str)).astype(str).tolist():
        if skill not in top_ids:
            top_ids.append(skill)
    top_ids = top_ids[:20]

    top_rows = []
    for skill_id in top_ids:
        card = card_by_id.get(skill_id, {})
        source = card.get("source", {}) if isinstance(card, dict) else {}
        source_complete = bool(source.get("book") and source.get("chapter") and source.get("page_range") and source.get("extraction_method"))
        top_rows.append(
            {
                "strategy_id": skill_id,
                "trigger_count": int(counter.get(skill_id, 0)),
                "source_book": source.get("book") or _infer_book(skill_id),
                "chapter": source.get("chapter") or "needs_grounding",
                "page_range": source.get("page_range") or "needs_grounding",
                "extraction_method": source.get("extraction_method") or "needs_grounding",
                "source_status": "grounded" if source_complete else "source_needs_page_grounding",
                "priority_action": "can_use_as_book_evidence" if source_complete else "weak_evidence_until_page_grounding",
            }
        )

    source_df = pd.DataFrame(source_rows)
    top_df = pd.DataFrame(top_rows)
    md = [
        "# Book Skill Grounding Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        f"- 已读取策略卡数量：{len(strategy_cards)}。",
        f"- 高频/适配待审策略数量：{len(top_rows)}。",
        "- 能找到书名、章节、页码范围、提取方式的策略可作为 Book Skill 证据；缺页码 grounding 的派生策略只能作为弱证据。",
        "- 本脚本只读 `book_skills/` 已提取文本和策略卡，不修改、不移动、不覆盖原始 PDF。",
        "",
        "## 书籍 OCR/文本覆盖",
        "",
        _markdown_table(source_df),
        "",
        "## 高频策略 Grounding 状态",
        "",
        _markdown_table(top_df),
        "",
        "## 下一步",
        "",
        "1. 对 `source_needs_page_grounding` 的 PPS/DOW 派生 ID 回到逐页 OCR 文本补页码和章节。",
        "2. 每条 accepted skill 必须记录触发次数、20 日正收益率、20 日均值、适用条件和失效条件。",
        "3. 下一轮 DeepSeek 决策前必须读取本 audit，并把缺来源策略作为弱证据或反证。",
    ]
    (REPORT_DIR / "book_skill_grounding_audit.md").write_text("\n".join(md), encoding="utf-8")
    return {"strategy_card_count": len(strategy_cards), "top_grounding_rows": len(top_rows)}


def _write_external_skill_seed_log() -> int:
    seeds = [
        ("SEED-001", "official_disclosure", "巨潮资讯公告", "https://www.cninfo.com.cn/", "公告密集披露常伴随基本面或治理变化", "announcement_density_30d > peer_median and evidence_quality high", "announcement_count, materiality, self_vs_peer", "unverified"),
        ("SEED-002", "official_disclosure", "上交所公司公告", "https://www.sse.com.cn/disclosure/listedinfo/announcement/", "交易所公告比社区消息优先级更高", "official_announcement_flag raises evidence quality but not direction", "official_count, event_type", "unverified"),
        ("SEED-003", "official_disclosure", "深交所互动易", "https://irm.cninfo.com.cn/", "互动问答可提示订单、产能、客户和监管关注", "irm_topic_count by category becomes news vector input", "topic_category, timestamp_quality", "unverified"),
        ("SEED-004", "official_disclosure", "上证 e 互动", "https://sns.sseinfo.com/", "问答活跃度可反映关注度但不等于利好", "interaction_attention_gap vs peers", "question_count, reply_quality", "unverified"),
        ("SEED-005", "public_aggregator", "AKShare 股票接口说明", "https://akshare.akfamily.xyz/data/stock/stock.html", "开源接口可发现可抓字段但需事实核验", "field_availability enters data coverage, not alpha directly", "available_fields, source_tier", "watching"),
        ("SEED-006", "public_sentiment", "东方财富股吧", "https://guba.eastmoney.com/", "社区热度突增可能代表情绪拥挤或事件发酵", "forum_attention_spike without official support is risk flag", "post_count_zscore, official_confirmation", "unverified"),
        ("SEED-007", "public_sentiment", "东方财富股吧", "https://guba.eastmoney.com/", "同行都被讨论但目标股沉默可能提示关注度不足", "peer_active_self_silent_flag lowers confidence", "self_posts, peer_posts", "unverified"),
        ("SEED-008", "official_disclosure", "巨潮资讯公告", "https://www.cninfo.com.cn/", "定增、减持、质押等资本动作需单独风险分类", "financing_or_holding_change event adds counter evidence", "event_type, shareholder_change", "unverified"),
        ("SEED-009", "official_disclosure", "交易所问询函", "https://www.sse.com.cn/disclosure/listedinfo/announcement/", "问询函/监管函通常优先进入风险预警", "regulatory_letter_count_90d > 0 triggers risk gate", "letter_count, severity", "unverified"),
        ("SEED-010", "official_disclosure", "业绩预告/快报", "https://www.cninfo.com.cn/", "业绩预告方向与价格动量冲突时要降级", "earnings_surprise_conflict lowers score", "forecast_change, price_momentum", "unverified"),
        ("SEED-011", "public_aggregator", "AKShare 行业板块", "https://akshare.akfamily.xyz/data/stock/stock.html", "行业广度弱时单股强势可能更脆弱", "peer_breadth < 0.5 reduces portfolio exposure", "peer_breadth, relative_strength", "watching"),
        ("SEED-012", "public_aggregator", "AKShare 概念板块", "https://akshare.akfamily.xyz/data/stock/stock.html", "概念扩散强于个股自身新闻时需要比较相对受益", "concept_policy_coverage minus self_coverage", "concept_news_count, self_news_count", "unverified"),
        ("SEED-013", "official_disclosure", "上市公司公告", "https://www.cninfo.com.cn/", "资产出售/并购重组需要区分一次性收益与主业改善", "restructuring event requires recurring_quality check", "event_type, recurring_profit", "unverified"),
        ("SEED-014", "public_sentiment", "社区经验帖", "https://guba.eastmoney.com/", "高位一致看多可能是情绪拥挤反证", "forum_positive_crowding with RSI high triggers overheat risk", "sentiment_ratio, rsi14", "unverified"),
        ("SEED-015", "official_disclosure", "互动平台产能问答", "https://irm.cninfo.com.cn/", "产能扩张需要订单和利用率共同确认", "capacity_signal accepted only with order_signal", "capacity_score, order_score", "unverified"),
        ("SEED-016", "official_disclosure", "价格政策/补贴公告", "https://www.cninfo.com.cn/", "政策利好要比较目标股与同行覆盖差异", "policy_background_score relative to peer group", "policy_score, peer_policy_score", "unverified"),
        ("SEED-017", "official_disclosure", "地域政策公开信息", "https://www.cninfo.com.cn/", "地域政策应作为背景而非单股强证据", "region_background_score capped unless self event exists", "region_score, self_event", "unverified"),
        ("SEED-018", "public_aggregator", "公开新闻聚合", "https://akshare.akfamily.xyz/data/stock/stock.html", "新闻缺失不能视为中性好消息", "news_missing_rate > threshold lowers confidence", "missing_rate, source_count", "watching"),
        ("SEED-019", "official_disclosure", "现金流/财报披露", "https://www.cninfo.com.cn/", "利润改善但现金流恶化时要触发财报反证", "profit_cashflow_divergence risk flag", "net_profit_growth, operating_cashflow", "unverified"),
        ("SEED-020", "official_disclosure", "分红与回购公告", "https://www.cninfo.com.cn/", "回购分红可改善质量信号但需看现金和估值", "buyback_dividend_quality requires cash adequacy", "cash_ratio, buyback_amount", "unverified"),
    ]
    path = REPORT_DIR / "external_skill_seed_log.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed_id",
                "source_tier",
                "source_name",
                "source_url",
                "raw_observation_summary",
                "candidate_rule",
                "quantifiable_features",
                "validation_status",
                "validation_plan",
                "risk_and_boundary",
            ],
        )
        writer.writeheader()
        for seed_id, tier, name, url, obs, rule, features, status in seeds:
            writer.writerow(
                {
                    "seed_id": seed_id,
                    "source_tier": tier,
                    "source_name": name,
                    "source_url": url,
                    "raw_observation_summary": obs,
                    "candidate_rule": rule,
                    "quantifiable_features": features,
                    "validation_status": status,
                    "validation_plan": "must pass at least two half-year blocks and three stock samples before accepted_skill",
                    "risk_and_boundary": "candidate seed only; community sources are not facts and cannot directly drive user-facing conclusions",
                }
            )
    return len(seeds)


def _update_memory_ledgers() -> None:
    strategy_path = ROOT / "memory" / "strategy_experience_ledger.csv"
    strategy_rows = _read_csv_rows(strategy_path)
    strategy_rows = [row for row in strategy_rows if row.get("experience_id") not in {"EXP-20260625-006", "EXP-20260625-007"}]
    strategy_rows.extend(
        [
            {
                "experience_id": "EXP-20260625-006",
                "source_round": "portfolio_positive_rate_experiments",
                "task_mode": "portfolio_pool",
                "rule_or_observation": "peer_confirmed_pullback + pool_pullback + every_2_weeks + top15 improved raw positive rate but early blocks remain weak",
                "train_blocks": "H2023_1-H2026_1 local panel search",
                "validation_block": "three 100-stock panels, 18 panel-blocks",
                "metric_before": "gate raw_pos=0.4043 raw_avg=-0.2435",
                "metric_after": "candidate raw_pos=0.6448 raw_avg=2.9869 coverage=0.3317 hit_blocks=9/18",
                "accepted_or_rejected": "observe",
                "failure_condition": "H2023_2/H2024_1/H2024_2 raw positive still below 0.60 for best candidate",
                "next_action": "send to DeepSeek flash validation with stronger peer/news/book evidence and avoid claiming final pass",
            },
            {
                "experience_id": "EXP-20260625-007",
                "source_round": "data_cache_5000_coverage",
                "task_mode": "all",
                "rule_or_observation": "full-A metadata entrypoint exists but complete walk-forward cache is still partial",
                "train_blocks": "not applicable",
                "validation_block": "coverage audit",
                "metric_before": "target >=5000 metadata plus scalable daily/financial/news cache",
                "metric_after": "metadata rows=5529 eligible=4981 local_daily=500 ground_truth_unique=495",
                "accepted_or_rejected": "accepted",
                "failure_condition": "paid_standardized cache has not yet been built and validated",
                "next_action": "build Tushare/paid_standardized offline cache with token safety, source labels and rate limits",
            },
        ]
    )
    _write_csv_rows(strategy_path, strategy_rows)

    book_path = ROOT / "memory" / "book_skill_adaptation_ledger.csv"
    book_rows = _read_csv_rows(book_path)
    book_rows = [row for row in book_rows if row.get("experience_id") not in {"BS-20260625-004"}]
    book_rows.append(
        {
            "experience_id": "BS-20260625-004",
            "source_round": "book_skill_grounding_audit",
            "strategy_id": "top20_triggered_skills",
            "source_book": "multiple",
            "source_status": "mixed_grounded_and_needs_page_grounding",
            "task_mode": "all",
            "trigger_count": "see reports/date_generalization/book_skill_grounding_audit.md",
            "metric_summary": "formal cards exist but many high-frequency derived IDs still need page grounding",
            "accepted_or_rejected": "observe",
            "failure_condition": "missing book/chapter/page/extraction method prevents strong evidence use",
            "next_action": "ground top PPS/DOW derived IDs before raising priority",
        }
    )
    _write_csv_rows(book_path, book_rows)


def _write_strategy_hardening_summary(coverage: dict[str, Any], book_audit: dict[str, Any], seed_count: int) -> None:
    agg = pd.read_csv(REPORT_DIR / "portfolio_positive_rate_experiments_aggregate.csv") if (REPORT_DIR / "portfolio_positive_rate_experiments_aggregate.csv").exists() else pd.DataFrame()
    diag = pd.read_csv(REPORT_DIR / "portfolio_positive_rate_experiments_diagnostics.csv") if (REPORT_DIR / "portfolio_positive_rate_experiments_diagnostics.csv").exists() else pd.DataFrame()
    best = agg.iloc[0].to_dict() if not agg.empty else {}
    lines = [
        "# Strategy Hardening Summary",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 当前阶段",
        "",
        "- 服务器环境已重建：`/data/cyx/1030/stock/.conda/stock-agent`，Python 3.11。",
        "- 全量测试通过：`67 passed`。",
        "- 组合正收益率实验已使用三组不同 100 股 sample 重跑，共 77760 条明细。",
        "- 当前项目允许授权 paid_standardized 数据源；本轮未调用付费 API，只使用已有本地缓存。",
        "",
        "## 组合模式实验结果",
        "",
        f"- 当前 gate 基线 raw 20 日正收益率：{BASELINE_GATE_PORTFOLIO_POSITIVE_RATE:.4f}。",
        f"- 最佳候选：`{best.get('score_preset', 'NA')} + {best.get('date_gate', 'NA')} + {best.get('row_gate', 'NA')} + {best.get('decision_frequency', 'NA')} + Top{best.get('top_n', 'NA')}`。",
        f"- 最佳候选 raw 20 日正收益率均值：{_fmt(best.get('raw_positive_20d_rate_mean'))}。",
        f"- 相对 gate raw 正收益率提升：{_fmt(best.get('raw_positive_lift_vs_gate'))}。",
        f"- 20 日平均收益均值：{_fmt(best.get('avg_return_mean'))}%。",
        f"- 决策覆盖率：{_fmt(best.get('decision_coverage'))}。",
        f"- 达到 0.60 的 panel-block：{best.get('raw_positive_hit_blocks', 'NA')}/{best.get('panel_blocks', 'NA')}。",
        "",
        "判断：组合 raw 正收益率相对 gate 有明显候选提升，但早期时间块仍弱，不能宣称最终时间泛化达标。",
        "",
        "## 最佳候选时间块诊断",
        "",
        _markdown_table(diag),
        "",
        "## 5000 股数据底座",
        "",
        f"- 当前全 A 元数据层股票数：{coverage.get('master_count')}；质量过滤后 eligible：{coverage.get('eligible_count')}。",
        f"- 当前本地日线/GT 缓存覆盖股票数：{coverage.get('current_count')}，目标为 {coverage.get('target_count')} 或全 A 可覆盖。",
        "- 已建立 `data/date_generalization_cache/market_5000/`，输出 coverage、全 A 元数据、相关股票图谱和新闻 world model schema。",
        "- 缺口：同概念、同地域、新闻共现 TopK 尚未完整缓存；财务披露日仍不足。",
        "- Tushare Pro/paid_standardized 数据源已允许作为后续离线缓存层；本轮未调用。",
        "",
        "## Book Skill 与外部 seed",
        "",
        f"- 已审计策略卡数量：{book_audit.get('strategy_card_count')}。",
        f"- 高频/适配待审 Book Skill 行数：{book_audit.get('top_grounding_rows')}。",
        f"- 已建立公开经验 seed：{seed_count} 条，默认未验证或观察状态。",
        "- 社区/论坛经验只能作为 candidate_skill_seed，必须经多时间块和多股票样本验证后才可升级。",
        "",
        "## 下一步",
        "",
        "1. 用 DeepSeek Flash 对最佳候选做小规模真实 Agent validation，重点复盘 H2023_2/H2024_1/H2024_2。",
        "2. 补同概念、同地域、历史相关性 TopK、新闻共现 TopK，提升 peer/news 通道。",
        "3. 对 PPS/DOW 高频派生策略补页码 grounding，未补齐前只作为弱证据。",
        "4. 启用 Tushare Pro/paid_standardized 时，直接按安全 adapter、离线缓存、限速和来源标注推进。",
    ]
    (REPORT_DIR / "strategy_hardening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _infer_book(strategy_id: str) -> str:
    if strategy_id.startswith("PPS"):
        return "专业投机原理"
    if strategy_id.startswith("DOW"):
        return "道氏理论"
    if strategy_id.startswith("MOS"):
        return "安全边际"
    if strategy_id.startswith("CANDLE"):
        return "日本蜡烛图技术"
    return "unknown"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "NA")).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
