from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DATA_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
BOOK_DIR = ROOT / "book_skills"
DOCS_DIR = ROOT / "docs"
CONFIG_DIR = ROOT / "config"
MEMORY_DIR = ROOT / "memory"
GT_PATHS = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

TIME_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1_YTD": ("2026-01-01", "2026-06-25"),
}
PANELS = {"sample_A": "systematic-A", "sample_B": "systematic-B", "sample_C": "systematic-C"}
PANEL_SIZE = 100
BANK_RETURN_20D = ((1 + 0.03) ** (20 / 252) - 1) * 100


NEWS_FEATURES = [
    ("self_news_intensity", "股票自身新闻/公告强度", "0-1", "min(1, log1p(self_count_30d)/log1p(20))", "missing -> 0 and raise news_missing_rate", True),
    ("peer_news_intensity", "同行新闻/公告强度", "0-1", "min(1, peer_group_news_count_avg/10)", "missing -> peer_unknown", True),
    ("policy_background_score", "政策背景", "-1..1", "industry_policy materiality normalized", "missing -> 0 but lower evidence quality", True),
    ("region_background_score", "地域背景", "-1..1", "region event score, capped unless self event exists", "missing -> 0", True),
    ("self_vs_peer_attention_gap", "自身相对同行关注差", "-1..1", "self_news_intensity - peer_news_intensity", "missing peer -> 0 and flag", True),
    ("peer_active_self_silent_flag", "同行活跃但自身沉默", "0/1", "peer_news_intensity high and self_news_intensity low", "missing -> 0 with warning", True),
    ("news_warning_score", "风险预警", "0-1", "risk/regulatory/financing/holding-change materiality", "missing -> 0 with missing-rate penalty", True),
    ("news_opportunity_score", "机会信号", "0-1", "order/capacity/product/policy opportunity materiality", "missing -> 0", True),
    ("news_evidence_quality", "证据质量", "0-1", "official source ratio + timestamp quality", "missing -> low quality", True),
    ("news_missing_rate", "新闻缺失率", "0-1", "missing expected source slots / expected slots", "missing -> 1", True),
    ("news_timestamp_quality", "时间戳质量", "0-1", "available_at completeness and <= decision_time", "missing -> 0", True),
    ("news_peer_diffusion_score", "同行新闻扩散", "-1..1", "peer risk/opportunity co-occurrence around target", "missing -> 0", True),
    ("official_confirmation_score", "官方确认度", "0-1", "official_count/(official_count+public_count)", "missing -> 0", True),
    ("community_attention_score", "社区关注度", "0-1", "public/community count zscore capped", "missing -> 0", False),
    ("community_crowding_risk", "社区拥挤反证", "0-1", "high positive crowding + high RSI/overheat", "missing -> 0", True),
    ("announcement_materiality_score", "公告重要性", "0-1", "max official announcement materiality", "missing -> 0", True),
]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    frame = _load_ground_truth()
    cards = _load_strategy_cards()
    top_skills = _top_skill_stats(frame, cards)

    _write_data_upgrade_decision()
    _write_news_world_model_docs(frame)
    _write_book_skill_outputs(top_skills)
    experiments = _run_systematic_experiments(frame)
    _write_systematic_reports(experiments, frame)
    _write_memory_ledgers(experiments, top_skills)
    _write_strategy_search_reports(experiments)
    _write_user_guides(experiments)
    _write_deepseek_reports()
    print("A股研究Agent")
    print("wrote systematic goal deliverables")


def _load_ground_truth() -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"code": str}, low_memory=False) for path in GT_PATHS if path.exists()]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "code"]).drop_duplicates(["date", "code"]).copy()
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    for col in ["return_5d", "return_10d", "return_20d", "total_score", "book_score"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.reset_index(drop=True)


def _load_strategy_cards() -> dict[str, dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for path in [BOOK_DIR / "strategy_cards.yaml", BOOK_DIR / "auto_strategy_cards.yaml", BOOK_DIR / "supporting_strategy_cards.yaml"]:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or []
        if isinstance(data, list):
            cards.extend(data)
    return {str(card.get("strategy_id")): card for card in cards if card.get("strategy_id")}


def _write_data_upgrade_decision() -> None:
    lines = [
        "# Data Upgrade Decision",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        "- 若目标是当前成熟 test 20 日 raw 正收益率 >= 0.65、往期半年块 >= 0.60、未来 zeroshot >= 0.60，仅依赖当前开源散源和 500 股缓存风险较高。",
        "- 当前最关键的数据缺口不是股票元数据，而是财务实际披露日、复权和停复牌/涨跌停、行业/地域/概念标签、公告/新闻时间戳和相关股票图谱。",
        "- 建议把用户购买的标准化 API 作为下一阶段 `optional paid_standardized offline cache source` 启用；当前项目规则已允许授权付费/会员/标准化数据源。",
        "- 本轮未读取 token、未调用 Tushare Pro、未下载付费数据；下一步可直接实现安全离线缓存 adapter。",
        "",
        "## 当前开源/本地缓存的限制",
        "",
        "- 时间线主要从 2023 起，难以覆盖完整牛熊切换和更早 zeroshot 检验。",
        "- 财务披露日不足，财务字段不能安全进入 walk-forward 决策。",
        "- 新闻/公告历史覆盖稀疏，当前不能把新闻通道稳定写成已验证优势。",
        "- 地域、概念、指数成分、新闻共现 TopK 不完整，限制单支和组合模式的同行/相关股票判断。",
        "- 开源接口字段和稳定性不统一，回测可复现性弱于标准化离线缓存。",
        "",
        "## 标准化 API 最优先补齐字段",
        "",
        "1. 交易日历、股票上市/退市状态、停复牌、涨跌停。",
        "2. 日线、复权因子、成交额、换手、波动率。",
        "3. 财务报告期和实际披露日，财务指标、利润表、资产负债表、现金流量表。",
        "4. 行业、地域、概念、指数成分。",
        "5. 公告/新闻事件表，必须保留 `published_at`/`available_at`。",
        "6. 可选特色字段：筹码、券商金股、每日胜率等，只能作为候选通道，需 ablation 验证。",
        "",
        "## 限速和安全方案",
        "",
        "- API 只用于离线缓存；回测和 DeepSeek 决策只读本地缓存。",
        "- 请求间隔默认 >= 0.7 秒，总频率 <= 100 次/分钟。",
        "- 下载带宽 <= 4MB/s。",
        "- token 只能从本地文件或环境变量读取，不写入代码、报告、prompt、ledger、缓存元数据或 Git。",
        "- 接口失败写入 coverage，不让流程崩溃。",
        "",
        "## 决策",
        "",
        "当前建议：需要标准化数据升级，但本轮不直接调用。下一步可直接实现安全离线缓存 adapter，并在报告中标注 paid_standardized 来源。",
    ]
    (REPORT_DIR / "data_upgrade_decision.md").write_text("\n".join(lines), encoding="utf-8")


def _write_news_world_model_docs(frame: pd.DataFrame) -> None:
    schema_rows = [
        {
            "feature_name": name,
            "description": desc,
            "range": rng,
            "calculation": calc,
            "missing_policy": missing,
            "enter_evidence_pack": enters,
            "leakage_guard": "available_at <= decision_time",
        }
        for name, desc, rng, calc, missing, enters in NEWS_FEATURES
    ]
    schema = pd.DataFrame(schema_rows)
    schema.to_csv(DATA_CACHE / "news_world_model_schema.csv", index=False, encoding="utf-8-sig")
    schema.to_csv(DATA_CACHE / "news_event_schema.csv", index=False, encoding="utf-8-sig")
    yaml_data = {
        "news_feature_schema_version": "news_world_model_v2",
        "updated_at": "2026-06-25",
        "leakage_guard": "Only use news/announcement records with available_at <= decision_time.",
        "features": schema_rows,
    }
    (CONFIG_DIR / "news_feature_schema.yaml").write_text(yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    news_cols = [c for c in frame.columns if c.startswith("news_") or c.startswith("peer_group_news")]
    coverage_rows = []
    for col in news_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        coverage_rows.append(
            {
                "feature": col,
                "non_null_rate": round(float(values.notna().mean()), 4),
                "non_zero_rate": round(float((values.fillna(0) != 0).mean()), 4),
                "mean_value": round(float(values.fillna(0).mean()), 4),
            }
        )
    coverage = pd.DataFrame(coverage_rows).sort_values(["non_zero_rate", "feature"], ascending=[False, True])
    coverage.to_csv(REPORT_DIR / "news_world_model_coverage.csv", index=False, encoding="utf-8-sig")
    coverage_lines = [
        "# News World Model Coverage",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        f"- 当前 ground truth 行数：{len(frame)}。",
        f"- 当前新闻/公告相关原始字段：{len(news_cols)} 个。",
        f"- V2 schema 字段：{len(schema)} 个。",
        "- 新闻事件表尚未完整回填，新闻通道仍需按 `available_at <= decision_time` 做历史化缓存。",
        "- 当前不能只靠新闻通道宣称能力提升，必须结合 ablation。",
        "",
        "## 原始字段覆盖",
        "",
        _markdown_table(coverage),
    ]
    (REPORT_DIR / "news_world_model_coverage.md").write_text("\n".join(coverage_lines), encoding="utf-8")

    docs = [
        "# News World Model V2",
        "",
        "本文件定义新闻/公告通道如何被量化为 Agent evidence，不构成投资建议。",
        "",
        "## 核心原则",
        "",
        "- 所有事件必须满足 `available_at <= decision_time`。",
        "- 新闻缺失不是中性好消息；缺失率进入 `news_missing_rate` 并降低置信度。",
        "- 单支模式使用更细粒度的自身、同行、地域、政策、社区信号。",
        "- 组合模式强调相对值：自身 vs 同行、同行活跃自身沉默、政策覆盖差、风险扩散。",
        "",
        "## 字段定义",
        "",
        _markdown_table(schema),
        "",
        "## 当前覆盖判断",
        "",
        f"- 当前 ground truth 行数：{len(frame)}。",
        f"- 当前已有新闻相关原始字段：{len(news_cols)} 个。",
        "- 历史新闻/公告仍偏稀疏，不能仅凭新闻通道宣称模型优势；必须用 ablation 验证。",
    ]
    (DOCS_DIR / "NEWS_WORLD_MODEL_V2.md").write_text("\n".join(docs), encoding="utf-8")


def _top_skill_stats(frame: pd.DataFrame, cards: dict[str, dict[str, Any]]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    if "triggered_skills" in frame:
        for _, row in frame.iterrows():
            text = str(row.get("triggered_skills") or "")
            for skill in text.replace(",", ";").split(";"):
                skill = skill.strip()
                if not skill or skill.lower() == "nan":
                    continue
                counter[skill] += 1
                rows.append({"strategy_id": skill, "return_20d": row.get("return_20d"), "date": row.get("date")})
    data = pd.DataFrame(rows)
    output = []
    for strategy_id, count in counter.most_common(20):
        subset = data[data["strategy_id"].eq(strategy_id)]
        returns = pd.to_numeric(subset["return_20d"], errors="coerce").dropna()
        card = cards.get(strategy_id, {})
        source = card.get("source", {}) if isinstance(card, dict) else {}
        source_status = "grounded" if source.get("book") and source.get("chapter") and source.get("page_range") and source.get("extraction_method") else "needs_grounding"
        confidence = source.get("confidence") or ("medium" if source_status == "grounded" else "low")
        output.append(
            {
                "strategy_id": strategy_id,
                "trigger_count": count,
                "sample_count": int(len(returns)),
                "raw_positive_20d_rate": round(float((returns > 0).mean()), 4) if len(returns) else None,
                "raw_avg_return_20d": round(float(returns.mean()), 4) if len(returns) else None,
                "source_book": source.get("book") or _infer_book(strategy_id),
                "chapter": source.get("chapter") or "needs_grounding",
                "page_range": source.get("page_range") or "needs_grounding",
                "extraction_method": source.get("extraction_method") or "needs_grounding",
                "confidence": confidence,
                "source_status": source_status,
                "validation_status": _skill_validation_status(returns, source_status),
            }
        )
    return pd.DataFrame(output)


def _write_book_skill_outputs(top_skills: pd.DataFrame) -> None:
    rows = []
    for _, row in top_skills.iterrows():
        rows.append(
            {
                "strategy_id": row["strategy_id"],
                "source_book": row["source_book"],
                "chapter": row["chapter"],
                "page_range": row["page_range"],
                "extraction_method": row["extraction_method"],
                "confidence": row["confidence"],
                "source_status": row["source_status"],
                "validation_status": row["validation_status"],
                "trigger_count": int(row["trigger_count"]),
                "sample_count": int(row["sample_count"]),
                "raw_positive_20d_rate": _none_to_float(row["raw_positive_20d_rate"]),
                "raw_avg_return_20d": _none_to_float(row["raw_avg_return_20d"]),
                "applicable_condition": "仅在 evidence pack 同时存在量价、同行、新闻或财务披露证据时作为辅助证据。",
                "failure_condition": "下一时间块失效、缺页码 grounding、或与强反证冲突时降权。",
                "user_output_boundary": "只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。",
            }
        )
    (BOOK_DIR / "grounded_skill_cards.yaml").write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8")
    lines = [
        "# Book Skill Validation Report",
        "",
        "本报告用于研究辅助型操作建议，不自动交易，不接券商接口，不承诺收益。",
        "",
        "## 高频 Book Skill 后验表现",
        "",
        _markdown_table(top_skills),
        "",
        "## 判断",
        "",
        "- 只有来源完整且跨时间块表现稳定的策略才能升级为 accepted skill。",
        "- 缺页码、缺章节或下一时间块失效的策略只能作为弱证据或反证。",
        "- 当前 Book Skill 有多个高频触发，但仍需在系统化实验中证明边际贡献。",
    ]
    (REPORT_DIR / "book_skill_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_systematic_experiments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    codes = sorted(frame["code"].dropna().astype(str).unique())
    strategies = _strategy_grid()
    for panel_name, seed in PANELS.items():
        panel_codes = _sample_codes(codes, seed, PANEL_SIZE)
        panel = frame[frame["code"].isin(panel_codes)].copy()
        for block, (start, end) in TIME_BLOCKS.items():
            block_df = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))].copy()
            if block_df.empty:
                continue
            prior_blocks = [b for b in TIME_BLOCKS if list(TIME_BLOCKS).index(b) < list(TIME_BLOCKS).index(block)]
            train_df = pd.concat(
                [
                    panel[(panel["date"] >= pd.Timestamp(TIME_BLOCKS[b][0])) & (panel["date"] <= pd.Timestamp(TIME_BLOCKS[b][1]))]
                    for b in prior_blocks
                ],
                ignore_index=True,
            ) if prior_blocks else panel.iloc[0:0].copy()
            thresholds = _thresholds(train_df if not train_df.empty else block_df)
            for spec in strategies:
                for frequency in spec["frequencies"]:
                    scheduled = _apply_frequency(block_df, frequency)
                    expected_dates = int(scheduled["date"].dt.strftime("%Y-%m-%d").nunique())
                    gated = _apply_strategy_gates(scheduled, spec, thresholds)
                    scored = _score_frame(gated, spec, panel_name, block)
                    selected = _select_top(scored, spec)
                    for task_mode in ["portfolio_pool_optimize", "single_stock_watch"]:
                        metric = _metrics(selected, expected_dates, task_mode)
                        rows.append(
                            {
                                "panel": panel_name,
                                "panel_size": len(panel_codes),
                                "time_block": block,
                                "is_provisional": block.endswith("YTD"),
                                "train_blocks": "+".join(prior_blocks) if prior_blocks else "none_prior_block",
                                "task_mode": task_mode,
                                "strategy_name": spec["name"],
                                "strategy_family": spec["family"],
                                "ablation_group": spec["ablation"],
                                "is_baseline": spec["is_baseline"],
                                "decision_frequency": frequency,
                                "top_n": spec["top_n"],
                                **metric,
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "systematic_experiment_matrix.csv", index=False, encoding="utf-8-sig")
    return result


def _strategy_grid() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_original_top3", "family": "baseline", "ablation": "baseline_original", "is_baseline": True, "score": "total", "top_n": 3, "gate": "none", "frequencies": ["twice_weekly"]},
        {"name": "baseline_original_top5", "family": "baseline", "ablation": "baseline_original", "is_baseline": True, "score": "total", "top_n": 5, "gate": "none", "frequencies": ["twice_weekly"]},
        {"name": "baseline_original_top10", "family": "baseline", "ablation": "baseline_original", "is_baseline": True, "score": "total", "top_n": 10, "gate": "none", "frequencies": ["twice_weekly"]},
        {"name": "baseline_equal_weight_all", "family": "baseline", "ablation": "all_candidates", "is_baseline": True, "score": "total", "top_n": 9999, "gate": "none", "frequencies": ["weekly_friday"]},
        {"name": "baseline_random_top10", "family": "baseline", "ablation": "random", "is_baseline": True, "score": "random", "top_n": 10, "gate": "none", "frequencies": ["weekly_friday"]},
        {"name": "baseline_cash_3pct", "family": "baseline", "ablation": "cash", "is_baseline": True, "score": "cash", "top_n": 0, "gate": "cash", "frequencies": ["weekly_friday"]},
        {"name": "candidate_full_agent_proxy_top15", "family": "agent_proxy", "ablation": "full_agent", "is_baseline": False, "score": "full", "top_n": 15, "gate": "pool_pullback_news_safe", "frequencies": ["twice_weekly", "weekly_friday", "every_2_weeks", "date_gate_only"]},
        {"name": "candidate_no_news_top15", "family": "ablation", "ablation": "no_news", "is_baseline": False, "score": "no_news", "top_n": 15, "gate": "pool_pullback", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_no_peer_top15", "family": "ablation", "ablation": "no_peer", "is_baseline": False, "score": "no_peer", "top_n": 15, "gate": "pool_pullback_news_safe", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_no_bookskill_top15", "family": "ablation", "ablation": "no_bookskill", "is_baseline": False, "score": "no_book", "top_n": 15, "gate": "pool_pullback_news_safe", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_no_memory_top15", "family": "ablation", "ablation": "no_memory", "is_baseline": False, "score": "full", "top_n": 15, "gate": "pool_pullback_news_safe", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_no_python_gate_top15", "family": "ablation", "ablation": "no_python_gate", "is_baseline": False, "score": "full", "top_n": 15, "gate": "none", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_python_only_top15", "family": "deterministic", "ablation": "python_only", "is_baseline": False, "score": "python_only", "top_n": 15, "gate": "pool_pullback_news_safe", "frequencies": ["every_2_weeks"]},
        {"name": "candidate_industry_top5", "family": "peer_pool", "ablation": "peer_pool", "is_baseline": False, "score": "full", "top_n": 5, "gate": "peer_breadth_ok", "frequencies": ["weekly_friday", "every_2_weeks"]},
    ]


def _sample_codes(codes: list[str], seed: str, size: int) -> list[str]:
    shuffled = sorted(codes, key=lambda code: hashlib.sha256(f"{seed}:{code}".encode()).hexdigest())
    return shuffled[: min(size, len(shuffled))]


def _apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    dates = frame["date"]
    if frequency == "twice_weekly":
        return frame[dates.dt.weekday.isin([1, 4])].copy()
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    if frequency == "date_gate_only":
        return frame.copy()
    raise ValueError(f"unknown frequency: {frequency}")


def _thresholds(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"prior_q40": 0.0, "peer_breadth_q55": 0.5, "news_risk_q50": 0.0}
    return {
        "prior_q40": float(_num(frame, "prior_return_20d").quantile(0.40)),
        "peer_breadth_q55": float(_num(frame, "peer_group_positive_breadth_20d").quantile(0.55)),
        "news_risk_q50": float((_num(frame, "news_risk_event_score_30d") + _num(frame, "news_legal_regulatory_score_30d")).quantile(0.50)),
    }


def _apply_strategy_gates(frame: pd.DataFrame, spec: dict[str, Any], thresholds: dict[str, float]) -> pd.DataFrame:
    if frame.empty or spec["gate"] in {"none", "cash"}:
        return frame.copy()
    data = frame.copy()
    selector = pd.Series(True, index=data.index)
    if "pool_pullback" in spec["gate"]:
        date_prior = data.assign(prior_num=_num(data, "prior_return_20d")).groupby(data["date"].dt.strftime("%Y-%m-%d"))["prior_num"].mean()
        allowed = set(date_prior[date_prior <= thresholds["prior_q40"]].index)
        selector &= data["date"].dt.strftime("%Y-%m-%d").isin(allowed)
    if "news_safe" in spec["gate"]:
        selector &= (_num(data, "news_risk_event_score_30d") + _num(data, "news_legal_regulatory_score_30d")) <= max(0.0, thresholds["news_risk_q50"])
    if spec["gate"] == "peer_breadth_ok":
        selector &= _num(data, "peer_group_positive_breadth_20d") >= thresholds["peer_breadth_q55"]
    return data[selector].copy()


def _score_frame(frame: pd.DataFrame, spec: dict[str, Any], panel_name: str, block: str) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        data["_score"] = []
        return data
    score_kind = spec["score"]
    if score_kind == "cash":
        data["_score"] = 0.0
        return data.iloc[0:0].copy()
    if score_kind == "random":
        data["_score"] = data["code"].map(lambda code: int(hashlib.sha256(f"{panel_name}:{block}:{code}".encode()).hexdigest()[:8], 16) / 1e9)
        return data
    total = _num(data, "total_score")
    rel = _num(data, "relative_strength_rank")
    peer_rel = _num(data, "peer_relative_to_group_20d")
    peer_breadth = _num(data, "peer_group_positive_breadth_20d")
    news_risk = _num(data, "news_risk_event_score_30d") + _num(data, "news_legal_regulatory_score_30d")
    news_opp = _num(data, "news_opportunity_event_score_30d")
    book = _num(data, "book_score")
    prior = _num(data, "prior_return_20d")
    overheat = ((prior > 60) | (_num(data, "rsi14") > 80)).astype(float)
    base = 0.45 * total + 0.20 * rel + 0.15 * (peer_rel > 0).astype(float) + 0.15 * (peer_breadth >= 0.55).astype(float) + 0.08 * news_opp + 0.08 * book - 0.20 * news_risk - 0.60 * overheat
    if score_kind == "total":
        data["_score"] = total
    elif score_kind == "full":
        data["_score"] = base
    elif score_kind == "no_news":
        data["_score"] = base - 0.08 * news_opp + 0.20 * news_risk
    elif score_kind == "no_peer":
        data["_score"] = base - 0.15 * (peer_rel > 0).astype(float) - 0.15 * (peer_breadth >= 0.55).astype(float)
    elif score_kind == "no_book":
        data["_score"] = base - 0.08 * book
    elif score_kind == "python_only":
        data["_score"] = 0.50 * rel + 0.25 * (prior.between(-15, 25)).astype(float) - 0.70 * overheat + 0.15 * (peer_breadth >= 0.5).astype(float)
    else:
        data["_score"] = total
    return data


def _select_top(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or spec["top_n"] <= 0:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(["date", "_score", "code"], ascending=[True, False, True])
    if spec["top_n"] >= 9999:
        return ordered.copy()
    return ordered.groupby(ordered["date"].dt.strftime("%Y-%m-%d"), group_keys=False).head(int(spec["top_n"])).copy()


def _metrics(selected: pd.DataFrame, expected_dates: int, task_mode: str) -> dict[str, Any]:
    expected_dates = max(int(expected_dates), 0)
    if selected.empty:
        cash = pd.Series([BANK_RETURN_20D] * expected_dates, dtype=float)
        return {
            "decision_dates": 0,
            "decision_coverage": 0.0,
            "selected_count_mean": 0.0,
            "raw_positive_20d_rate": None,
            "raw_avg_return_20d": None,
            "cash_blended_avg_return_20d": round(float(cash.mean()), 4) if len(cash) else None,
            "loss_20d_over_5_rate": None,
            "return_20d_std": None,
            "stability_score": None,
            "gt_pending_rate": 1.0 if expected_dates == 0 else 0.0,
        }
    if task_mode == "portfolio_pool_optimize":
        daily = selected.groupby(selected["date"].dt.strftime("%Y-%m-%d")).agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
        values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
        selected_count_mean = float(daily["selected_count"].mean()) if not daily.empty else 0.0
        decision_dates = int(len(daily))
    else:
        values = pd.to_numeric(selected["return_20d"], errors="coerce").dropna()
        selected_count_mean = float(len(selected) / max(selected["date"].dt.strftime("%Y-%m-%d").nunique(), 1))
        decision_dates = int(selected["date"].dt.strftime("%Y-%m-%d").nunique())
    skipped = max(expected_dates - decision_dates, 0)
    cash_blended = pd.concat([values, pd.Series([BANK_RETURN_20D] * skipped, dtype=float)], ignore_index=True)
    avg = float(values.mean()) if len(values) else None
    std = float(values.std(ddof=0)) if len(values) else None
    loss = float((values <= -5).mean()) if len(values) else None
    return {
        "decision_dates": decision_dates,
        "decision_coverage": round(float(decision_dates / max(expected_dates, decision_dates, 1)), 4),
        "selected_count_mean": round(selected_count_mean, 4),
        "raw_positive_20d_rate": round(float((values > 0).mean()), 4) if len(values) else None,
        "raw_avg_return_20d": round(avg, 4) if avg is not None else None,
        "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4) if len(cash_blended) else None,
        "loss_20d_over_5_rate": round(loss, 4) if loss is not None else None,
        "return_20d_std": round(std, 4) if std is not None else None,
        "stability_score": round(float(avg - 0.5 * std - 10 * loss), 4) if avg is not None and std is not None and loss is not None else None,
        "gt_pending_rate": 0.0,
    }


def _write_systematic_reports(experiments: pd.DataFrame, frame: pd.DataFrame) -> None:
    if experiments.empty:
        return
    group_cols = ["task_mode", "strategy_name", "strategy_family", "ablation_group", "is_baseline", "decision_frequency", "top_n"]
    aggregate = experiments.groupby(group_cols).agg(
        panel_blocks=("time_block", "count"),
        raw_positive_20d_rate_mean=("raw_positive_20d_rate", "mean"),
        raw_positive_20d_rate_std=("raw_positive_20d_rate", "std"),
        raw_avg_return_20d_mean=("raw_avg_return_20d", "mean"),
        cash_blended_avg_return_20d_mean=("cash_blended_avg_return_20d", "mean"),
        decision_coverage_mean=("decision_coverage", "mean"),
        loss_20d_over_5_rate_mean=("loss_20d_over_5_rate", "mean"),
        stability_score_mean=("stability_score", "mean"),
        hit_blocks=("raw_positive_20d_rate", lambda s: int((pd.to_numeric(s, errors="coerce") >= 0.60).sum())),
    ).reset_index()
    aggregate["rank_score"] = (
        aggregate["raw_positive_20d_rate_mean"].fillna(0)
        + 0.02 * aggregate["raw_avg_return_20d_mean"].fillna(0)
        + 0.08 * aggregate["decision_coverage_mean"].fillna(0)
        - 0.10 * aggregate["loss_20d_over_5_rate_mean"].fillna(0)
    )
    aggregate = aggregate.sort_values(["task_mode", "rank_score"], ascending=[True, False])
    aggregate.to_csv(REPORT_DIR / "systematic_experiment_aggregate.csv", index=False, encoding="utf-8-sig")

    diagnostics = experiments.groupby(["task_mode", "time_block", "strategy_name"]).agg(
        panels=("panel", "nunique"),
        raw_positive_20d_rate_mean=("raw_positive_20d_rate", "mean"),
        raw_avg_return_20d_mean=("raw_avg_return_20d", "mean"),
        decision_coverage_mean=("decision_coverage", "mean"),
        loss_20d_over_5_rate_mean=("loss_20d_over_5_rate", "mean"),
    ).reset_index()
    diagnostics.to_csv(REPORT_DIR / "systematic_experiment_diagnostics.csv", index=False, encoding="utf-8-sig")

    baseline = aggregate[aggregate["is_baseline"].astype(bool)].copy()
    baseline.to_csv(REPORT_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    ablation = aggregate[~aggregate["is_baseline"].astype(bool)].copy()
    ablation.to_csv(REPORT_DIR / "systematic_ablation_aggregate.csv", index=False, encoding="utf-8-sig")

    best_port = aggregate[aggregate["task_mode"].eq("portfolio_pool_optimize")].head(1)
    best_single = aggregate[aggregate["task_mode"].eq("single_stock_watch")].head(1)
    summary = [
        "# Systematic Optimization Summary",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 实验覆盖",
        "",
        f"- ground truth 行数：{len(frame)}。",
        f"- 系统化实验行数：{len(experiments)}。",
        "- 覆盖两种任务模式、3 次 panel、2023-2026H1、baseline、ablation、不同频率和 TopN。",
        "- 本轮未调用 DeepSeek，新表里的 `full_agent` 是基于现有 DeepSeek 经验和本地规则的可回测 proxy，正式指标仍需 Flash/Pro 决策卡复核。",
        "",
        "## 最佳组合模式候选",
        "",
        _markdown_table(best_port),
        "",
        "## 最佳单支模式候选",
        "",
        _markdown_table(best_single),
        "",
        "## 风险判断",
        "",
        "- 若指标在 2025/2026 明显好于 2023/2024，必须视为日期泛化风险。",
        "- cash blended 只能说明体验和防守，不可单独证明选股能力。",
        "- 当前仍缺财务披露日和完整新闻事件表，结论不能写成最终通过。",
    ]
    (REPORT_DIR / "systematic_optimization_summary.md").write_text("\n".join(summary), encoding="utf-8")

    _write_mode_report("single_stock_watch", aggregate, diagnostics, REPORT_DIR / "single_stock_mode_report.md")
    _write_mode_report("portfolio_pool_optimize", aggregate, diagnostics, REPORT_DIR / "portfolio_pool_mode_report.md")
    _write_ablation_report(aggregate)
    _write_baseline_report(baseline)
    _write_failure_and_acceptance(experiments, aggregate)
    _write_news_feature_ablation(aggregate)


def _write_mode_report(task_mode: str, aggregate: pd.DataFrame, diagnostics: pd.DataFrame, path: Path) -> None:
    top = aggregate[aggregate["task_mode"].eq(task_mode)].head(12)
    diag = diagnostics[diagnostics["task_mode"].eq(task_mode)].head(80)
    title = "Single Stock Mode Report" if task_mode == "single_stock_watch" else "Portfolio Pool Mode Report"
    lines = [
        f"# {title}",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Top Results",
        "",
        _markdown_table(top),
        "",
        "## Time Block Diagnostics",
        "",
        _markdown_table(diag),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ablation_report(aggregate: pd.DataFrame) -> None:
    rows = []
    for task in ["portfolio_pool_optimize", "single_stock_watch"]:
        full = aggregate[(aggregate["task_mode"].eq(task)) & (aggregate["ablation_group"].eq("full_agent"))].head(1)
        if full.empty:
            continue
        full_pos = float(full.iloc[0]["raw_positive_20d_rate_mean"])
        for _, row in aggregate[aggregate["task_mode"].eq(task)].iterrows():
            rows.append(
                {
                    "task_mode": task,
                    "ablation_group": row["ablation_group"],
                    "strategy_name": row["strategy_name"],
                    "raw_positive_20d_rate_mean": row["raw_positive_20d_rate_mean"],
                    "delta_vs_full_agent": None if pd.isna(row["raw_positive_20d_rate_mean"]) else round(float(row["raw_positive_20d_rate_mean"]) - full_pos, 4),
                    "decision_coverage_mean": row["decision_coverage_mean"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "ablation_report.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Ablation Report",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 消融结果",
        "",
        _markdown_table(table),
        "",
        "## 解释规则",
        "",
        "- `delta_vs_full_agent` 为负，说明该组分可能有边际贡献；为正则说明 full_agent 当前组合方式未必最优。",
        "- `no_memory` 当前为结构性占位，因为 memory 经验尚未完全数值化接入；后续必须把 accepted/rejected/observe 显式转成 evidence 特征。",
    ]
    (REPORT_DIR / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_baseline_report(baseline: pd.DataFrame) -> None:
    lines = [
        "# Baseline Comparison",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "下表均为 baseline，用于参考，不代表系统最终策略。",
        "",
        _markdown_table(baseline),
    ]
    (REPORT_DIR / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _write_failure_and_acceptance(experiments: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    weak = experiments[(pd.to_numeric(experiments["raw_positive_20d_rate"], errors="coerce") < 0.60) & experiments["strategy_name"].str.contains("candidate_full_agent_proxy", na=False)].copy()
    weak = weak.sort_values(["time_block", "raw_positive_20d_rate"]).head(80)
    weak.to_csv(REPORT_DIR / "failure_case_samples.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Failure Case Review",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 低于 0.60 的 full-agent-proxy 样本",
        "",
        _markdown_table(weak.head(30)),
        "",
        "## 主要失败模式",
        "",
        "- H2023/H2024 的低胜率仍是日期泛化短板。",
        "- 新闻和财务披露日缺口使 Agent 无法稳定区分真实风险和信息不足。",
        "- 当前 memory 和 Book Skill 仍未完全转成可量化反证。",
    ]
    (REPORT_DIR / "failure_case_review.md").write_text("\n".join(lines), encoding="utf-8")

    task_best = aggregate.sort_values("rank_score", ascending=False).groupby("task_mode").head(1)
    pass_rows = []
    for _, row in task_best.iterrows():
        pass_rows.append(
            {
                "task_mode": row["task_mode"],
                "best_strategy": row["strategy_name"],
                "raw_positive_20d_rate_mean": row["raw_positive_20d_rate_mean"],
                "hit_blocks": row["hit_blocks"],
                "panel_blocks": row["panel_blocks"],
                "passes_current_target_0_65": bool(pd.notna(row["raw_positive_20d_rate_mean"]) and float(row["raw_positive_20d_rate_mean"]) >= 0.65),
                "passes_time_generalization": bool(int(row["hit_blocks"]) == int(row["panel_blocks"])),
                "decision": "not_final_pass_without_flash_pro_and_data_upgrade",
            }
        )
    check = pd.DataFrame(pass_rows)
    check.to_csv(REPORT_DIR / "final_acceptance_check.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Final Acceptance Check",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        _markdown_table(check),
        "",
        "## 结论",
        "",
        "- 本轮完成系统化离线评估和交付物生成，但尚未完成 DeepSeek Flash/Pro 真实决策卡复核。",
        "- 若未同时满足当前 test >= 0.65、往期半年块 >= 0.60、未来 zeroshot >= 0.60，不得宣称日期泛化通过。",
    ]
    (REPORT_DIR / "final_acceptance_check.md").write_text("\n".join(lines), encoding="utf-8")


def _write_news_feature_ablation(aggregate: pd.DataFrame) -> None:
    subset = aggregate[aggregate["ablation_group"].isin(["full_agent", "no_news"])]
    lines = [
        "# News Feature Ablation",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        _markdown_table(subset),
        "",
        "## 判断",
        "",
        "- 只有当 full_agent 稳定优于 no_news，且新闻覆盖率足够，才能把新闻通道写成已验证优势。",
        "- 当前新闻事件表仍需按 available_at 补全，新闻优势暂列为待验证。",
    ]
    (REPORT_DIR / "news_feature_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def _write_memory_ledgers(experiments: pd.DataFrame, top_skills: pd.DataFrame) -> None:
    _append_or_replace_csv(
        MEMORY_DIR / "strategy_experience_ledger.csv",
        "experience_id",
        [
            {
                "experience_id": "EXP-20260625-008",
                "source_round": "systematic_goal_deliverables",
                "task_mode": "all",
                "rule_or_observation": "systematic matrix now covers two task modes, three panels, seven time blocks, baselines and ablations",
                "train_blocks": "H2023_1-H2025_2 proxy search",
                "validation_block": "H2026_1_YTD plus historical diagnostics",
                "metric_before": "portfolio best candidate raw_pos=0.6448 from prior positive-rate search",
                "metric_after": f"systematic rows={len(experiments)}; final pass still false without Flash/Pro and data upgrade",
                "accepted_or_rejected": "accepted",
                "failure_condition": "recent-window lift does not generalize to all historical blocks",
                "next_action": "run DeepSeek Flash validation on frozen candidates after data/world-model upgrade",
            }
        ],
    )
    _append_or_replace_csv(
        MEMORY_DIR / "book_skill_adaptation_ledger.csv",
        "experience_id",
        [
            {
                "experience_id": "BS-20260625-005",
                "source_round": "systematic_goal_deliverables",
                "strategy_id": "top20_grounded_skill_cards",
                "source_book": "multiple",
                "source_status": "grounded_skill_cards_generated",
                "task_mode": "all",
                "trigger_count": f"top_rows={len(top_skills)}",
                "metric_summary": "book_skills/grounded_skill_cards.yaml and book_skill_validation_report.md generated",
                "accepted_or_rejected": "observe",
                "failure_condition": "Book Skill contribution must pass no_bookskill ablation and time-block validation",
                "next_action": "use grounded cards in DeepSeek evidence pack and rerun ablation",
            }
        ],
    )
    _append_text(
        MEMORY_DIR / "data_source_upgrade.md",
        "\n## Systematic Goal Update - 2026-06-25\n\n"
        "- `reports/date_generalization/data_upgrade_decision.md` recommends standardizing data before claiming 0.65/0.60/0.60 generalization.\n"
        "- Current policy allows authorized Tushare Pro / paid_standardized sources; no token was read and no paid API was called in this run.\n"
        "- Next safe step: offline cache adapter with >=0.7s request interval and <=4MB/s download bandwidth.\n",
    )
    _write_csv(
        MEMORY_DIR / "news_world_model_ledger.csv",
        [
            {
                "experience_id": "NEWS-20260625-001",
                "source_round": "systematic_goal_deliverables",
                "rule_or_observation": "news fields need timestamp-safe event table before claiming alpha",
                "metric_before": "existing sparse news_* columns",
                "metric_after": "news_world_model_v2 schema and ablation report generated",
                "accepted_or_rejected": "observe",
                "failure_condition": "no_news matches or beats full_agent, or news_missing_rate high",
                "next_action": "backfill announcements/news with available_at and rerun ablation",
            }
        ],
    )
    _write_csv(
        MEMORY_DIR / "ablation_findings_ledger.csv",
        [
            {
                "experience_id": "ABL-20260625-001",
                "source_round": "systematic_experiment_matrix",
                "task_mode": "all",
                "rule_or_observation": "ablation must be reported before treating news/peer/book/memory as advantages",
                "metric_before": "component contribution not consistently isolated",
                "metric_after": f"systematic rows={len(experiments)}",
                "accepted_or_rejected": "accepted",
                "failure_condition": "component has no measured lift or only works in recent blocks",
                "next_action": "lower priority for non-contributing channels and record counter evidence",
            }
        ],
    )
    _write_csv(
        MEMORY_DIR / "failure_case_ledger.csv",
        [
            {
                "failure_id": "FAIL-20260625-001",
                "source_round": "systematic_goal_deliverables",
                "task_mode": "portfolio_pool_optimize",
                "failure_pattern": "early blocks H2023/H2024 remain weaker than recent blocks",
                "countermeasure": "add disclosure-date-safe financial data, richer news, and peer graph before strategy freeze",
                "status": "open",
            }
        ],
    )


def _write_strategy_search_reports(experiments: pd.DataFrame) -> None:
    lines = [
        "# Strategy Search Space",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 可优化对象",
        "",
        "- Python gate：回撤、过热、同行广度、新闻风险、财报缺口。",
        "- 组合策略：TopN、现金防守、行业分散、同类/跨行业候选池。",
        "- 单支策略：模拟研究暴露、风险降级、信息不足。",
        "- 新闻/同行/Book Skill/memory：作为 DeepSeek evidence pack 的可审计证据。",
        "",
        "## 防过拟合规则",
        "",
        "- 每次只引入 1-3 条规则。",
        "- 用历史块训练，用下一块验证，test 不参与调参。",
        "- 近期有效但早期失效的规则标记为日期过拟合。",
    ]
    (REPORT_DIR / "strategy_search_space.md").write_text("\n".join(lines), encoding="utf-8")
    tree_lines = [
        "# Tree-Based Gate Experiment",
        "",
        "本轮未引入外部机器学习依赖；以可解释 rule-tree 方式记录候选 gate。",
        "",
        "## Rule Tree",
        "",
        "1. 若池子处于回撤窗口，允许组合研究暴露。",
        "2. 若新闻风险高或披露日缺失，降权或转入信息不足。",
        "3. 若同行广度高且个股相对同行不弱，提高候选排序。",
        "4. 若高位过热且无官方证据，降低候选排序。",
        "",
        "## 过拟合检查",
        "",
        "- 任何树规则必须在 H2023/H2024/H2025/H2026 均报告表现。",
        "- 若只改善 H2025/H2026，不得进入最终锁定策略。",
    ]
    (REPORT_DIR / "tree_based_gate_experiment.md").write_text("\n".join(tree_lines), encoding="utf-8")
    _write_csv(
        REPORT_DIR / "strategy_version_changes.csv",
        [
            {
                "strategy_version": "systematic_goal_v1",
                "change": "introduced systematic experiment matrix, news world model v2, Book Skill validation, and market data upgrade decision",
                "train_blocks": "H2023_1-H2025_2",
                "validation_blocks": "H2026_1_YTD plus historical diagnostics",
                "status": "observe_not_final_locked",
            }
        ],
    )


def _write_user_guides(experiments: pd.DataFrame) -> None:
    user_lines = [
        "# A 股研究 Agent 用户手册",
        "",
        "本系统输出研究辅助型操作建议，不自动交易，不接券商接口，不承诺收益。",
        "",
        "## 能做什么",
        "",
        "- 单支股票模式：帮助用户做买入/卖出/加减仓/持有/等待建议、盯盘复核、风险和信息缺口检查。",
        "- 多股组合模式：在候选池中排序，说明 TopN、候选操作优先级、现金防守和反证原因。",
        "- 辅助研究分级仍使用四档：`继续深挖`、`放入观察`、`暂时剔除`、`信息不足`，但不能代替操作建议。",
        "",
        "## 系统如何判断",
        "",
        "系统按固定流程读取：量价/Python gate、新闻/公告、同行/相关股票、Book Skill、历史 memory、反证和数据缺口。DeepSeek 决策前必须看到这些材料，最后输出研究分级和理由摘要。",
        "",
        "## Baseline 是什么",
        "",
        "baseline 是参考线，包括原始 Top3/Top5/Top10、Python only、随机 TopN、全候选池等权和现金防守。它们用于判断系统是否真的改进，而不是最终建议。",
        "",
        "## 当前状态",
        "",
        "- 单支模式已有盯盘/排雷雏形。",
        "- 组合模式已有候选提升，但早期时间块仍弱，不能宣称最终日期泛化通过。",
        "- 数据缺口集中在财务披露日、新闻事件表、地域/概念和新闻共现图谱。",
        "",
        "## 用户该如何使用",
        "",
        "先选择任务：单支股票复核，或多股候选池比较。系统会说明数据是否足够、哪些技能被触发、哪些反证存在，以及下一步应继续研究还是暂时放弃研究。",
    ]
    (DOCS_DIR / "USER_GUIDE_AGENT_RESEARCH_SYSTEM.md").write_text("\n".join(user_lines), encoding="utf-8")
    tech_lines = [
        "# A 股研究 Agent 技术工作流",
        "",
        "## Pipeline",
        "",
        "preflight -> data_cache_build_or_validate -> evidence_pack_build -> deepseek_decision -> schema_validation -> metric_backfill_after_gt_maturity -> failure_reflection -> strategy_update_proposal -> strategy_freeze -> next_block_or_test_validation -> user_report",
        "",
        "## Walk-forward",
        "",
        "时间块按 H2023_1 -> H2023_2 -> H2024_1 -> H2024_2 -> H2025_1 -> H2025_2 -> H2026_1_YTD 推进。第 t 块的后验只能用于更新第 t+1 块之前的策略，test 不参与调参。",
        "",
        "## Evidence Pack",
        "",
        "- Python gate 特征",
        "- 新闻/公告 world model v2",
        "- 同行/相关股票图谱",
        "- Book Skill grounded cards",
        "- memory accepted/rejected/observe",
        "- counter evidence 和 data_missing_flags",
        "",
        "## Model Policy",
        "",
        "- 训练/搜索/ablation 使用 deepseek-v4-flash。",
        "- 策略冻结后的小规模复核使用 deepseek-v4-pro。",
        "- 本地 deterministic runner 只作为 baseline/fallback。",
    ]
    (DOCS_DIR / "TECHNICAL_WORKFLOW_AGENT_RESEARCH_SYSTEM.md").write_text("\n".join(tech_lines), encoding="utf-8")


def _write_deepseek_reports() -> None:
    usage_files = list(REPORT_DIR.glob("*usage*.csv"))
    rows = []
    for path in usage_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        rows.append({"source_file": path.name, "rows": len(df), "columns": ";".join(df.columns.astype(str).tolist())})
    usage = pd.DataFrame(rows) if rows else pd.DataFrame([{"source_file": "none", "rows": 0, "columns": "no_usage_file_found"}])
    usage.to_csv(REPORT_DIR / "deepseek_usage_summary.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "deepseek_invalid_outputs.jsonl").write_text("", encoding="utf-8")
    lines = [
        "# Flash / Pro Comparison",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "- 训练、搜索、ablation 和失败反思默认使用 `deepseek-v4-flash`。",
        "- `deepseek-v4-pro` 只用于策略冻结后的小规模复核和正式用户推理。",
        "- DeepSeek 默认高并发策略：`max_workers=0` 自动使用 `min(evidence_pack_count, model_concurrency_limit)`。",
        "- 当前模型并发上限配置：`deepseek-v4-flash=2500`，`deepseek-v4-pro=500`，`deepseek-chat=2500`，`deepseek-reasoner=500`。",
        "- usage 需要记录 requested/effective workers、模型上限、429/timeout/invalid；出现 429 或 timeout 时下一 shard 降并发。",
        "- 本轮系统化交付未新增 DeepSeek 调用，避免在数据/实验框架未稳定前浪费 token。",
        "- 现有 Flash panel 报告显示 60 张 evidence pack 并发可稳定运行，invalid 为 0；但最终验收仍需 Pro 复核。",
    ]
    (REPORT_DIR / "flash_pro_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = DATA_CACHE / "cache_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest.setdefault("research_only", True)
    manifest.setdefault("no_broker", True)
    manifest.setdefault("no_auto_trade", True)
    manifest.setdefault("no_paid_source_used", True)
    manifest.setdefault("files", {})
    manifest["files"]["news_event_schema.csv"] = len(NEWS_FEATURES)
    manifest["files"]["news_world_model_schema.csv"] = len(NEWS_FEATURES)
    manifest["news_world_model_version"] = "news_world_model_v2"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_or_replace_csv(path: Path, key: str, rows: list[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    replace_keys = {row[key] for row in rows}
    merged = [row for row in existing if row.get(key) not in replace_keys]
    merged.extend(rows)
    fieldnames: list[str] = []
    for row in merged:
        for col in row.keys():
            if col not in fieldnames:
                fieldnames.append(col)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            writer.writerow(row)


def _append_text(path: Path, text: str) -> None:
    current = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    marker = text.strip().splitlines()[0]
    if marker in current:
        return
    path.write_text(current.rstrip() + "\n" + text, encoding="utf-8")


def _num(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[field], errors="coerce").fillna(0.0)


def _none_to_float(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return value


def _skill_validation_status(returns: pd.Series, source_status: str) -> str:
    if source_status != "grounded":
        return "weak_until_grounded"
    if len(returns) < 100:
        return "sample_insufficient"
    pos = float((returns > 0).mean())
    if pos >= 0.60:
        return "watching_strong_candidate"
    if pos <= 0.45:
        return "downweight_or_counter_evidence"
    return "observe"


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


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "无数据"
    data = df.copy()
    if len(data) > 60:
        data = data.head(60)
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()
