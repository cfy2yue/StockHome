from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src import APP_PREFIX, DISCLAIMER
from src.analysis.backtest_simple import backtest_breakout
from src.analysis.candlestick import analyze_candlestick
from src.analysis.comparison import write_candidate_matrix
from src.analysis.financial_quality import analyze_financial_quality
from src.analysis.risk_checks import build_counterevidence
from src.analysis.scoring import score_all
from src.analysis.strategy_matcher import load_strategy_cards, match_strategies
from src.analysis.trend import analyze_trend
from src.analysis.valuation import analyze_valuation
from src.data.akshare_adapter import AKShareAdapter
from src.data.source_registry import SourceRegistry
from src.data.multisource_adapter import classify_source
from src.reports.report_generator import render_final_review, render_stock_report, write_text
from src.reports.structured_response import BookSkillCitation, InputFlowEvidence, render_research_answer
from src.world_model.world_model import build_world_model
from src.skill_bridge import SkillBridge


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    full = ROOT / path if not Path(path).is_absolute() else Path(path)
    return yaml.safe_load(full.read_text(encoding="utf-8")) or {}


def infer_output_dir(config: dict, config_path: str | None) -> Path:
    if config.get("output_dir"):
        return ROOT / config["output_dir"]
    if config_path:
        return ROOT / "reports" / "test_runs" / Path(config_path).stem
    return ROOT / "reports" / "latest"


def resolve_stock(adapter: AKShareAdapter, query: str, assumed: dict | None = None) -> dict:
    result = adapter.resolve_stock(query)
    if result.ok and result.data:
        first = result.data[0]
        return {"code": str(first.get("代码") or first.get("code")), "name": str(first.get("名称") or first.get("name"))}
    if assumed:
        return {"code": assumed["code"], "name": assumed["name"]}
    return {"code": query, "name": "信息不足"}


def _stock_from_candidate(adapter: AKShareAdapter, candidate: dict, fallback_query: str | None = None) -> dict:
    if candidate.get("code") and candidate.get("name"):
        return {"code": str(candidate["code"]), "name": str(candidate["name"])}
    query = candidate.get("query_name") or candidate.get("stock") or fallback_query or candidate.get("code") or "600888"
    return resolve_stock(adapter, str(query), candidate.get("assumed_match"))


def analyze_one_stock(adapter: AKShareAdapter, registry: SourceRegistry, stock: dict, task: str, dry_run: bool) -> tuple[dict, dict, dict, dict]:
    label = f"{stock['name']}({stock['code']})"
    quote = adapter.current_quote(stock["code"], stock["name"])
    registry.add(f"{label} 当前行情", "成功" if quote.ok else "失败", quote.source, quote.fetched_at or "", quote.error or quote.warning or "无", "否" if quote.ok else "是")
    quote_data = quote.data if isinstance(quote.data, dict) else {}
    stock["price"] = quote_data.get("最新价", "信息不足")
    stock["industry"] = quote_data.get("行业", "信息不足")

    daily = adapter.daily_history(stock["code"])
    registry.add(f"{label} 日线行情", "成功" if daily.ok else "失败", daily.source, daily.fetched_at or "", daily.error or daily.warning or "无", "否" if daily.data else "是")
    financial_result = adapter.financial_summary(stock["code"])
    registry.add(f"{label} 财务摘要", "成功" if financial_result.ok else "失败", financial_result.source, financial_result.fetched_at or "", financial_result.error or financial_result.warning or "无", "是")
    valuation_result = adapter.valuation(stock["code"])
    registry.add(f"{label} 估值指标", "成功" if valuation_result.ok else "失败", valuation_result.source, valuation_result.fetched_at or "", valuation_result.error or valuation_result.warning or "无", "是")

    world = build_world_model(stock["code"], stock["name"], dry_run=dry_run)
    for label, key in [("新闻", "news"), ("公告", "announcements"), ("市场环境", "market"), ("行业事件", "sector")]:
        result = world[key]
        registry.add(f"{stock['name']}({stock['code']}) {label}", "成功" if result.ok else "失败", result.source, result.fetched_at or "", result.error or result.warning or "无", "是" if not result.ok else "否")

    # === Skill Bridge 增强层（用户无感知） ===
    skill = SkillBridge()
    skill_data: dict[str, object] = {}
    if skill.available and not dry_run:
        tech = skill.technical_bundle(stock["code"])
        if tech.ok:
            registry.add(f"{label} Skill技术指标", "成功", tech.source, tech.fetched_at or "", "无", "否")
            skill_data["technical_bundle"] = tech.data
        else:
            registry.add(f"{label} Skill技术指标", "失败", tech.source, tech.fetched_at or "", tech.error or "Skill未激活", "是")
        # 舆情搜索
        sentiment = skill.research_search(f"{stock['name']} 最新公告 研报", limit=5)
        if sentiment.ok:
            registry.add(f"{label} Skill舆情搜索", "成功", sentiment.source, sentiment.fetched_at or "", "无", "否")
            skill_data["sentiment"] = sentiment.data
        else:
            registry.add(f"{label} Skill舆情搜索", "失败", sentiment.source, sentiment.fetched_at or "", sentiment.error or "Skill未激活", "是")
    elif skill.available and dry_run:
        registry.add(f"{label} Skill增强", "跳过", "skill_bridge / dry-run", "", "dry-run 模式下不调用 Skill", "否")
    else:
        registry.add(f"{label} Skill增强", "未激活", "skill_bridge / unavailable", "", "Kimi Work stock-assistant 不可用，使用现有数据源", "否")
    # === Skill Bridge 结束 ===

    trend = analyze_trend(daily.data or [])
    candle = analyze_candlestick(daily.data or [])
    financial = analyze_financial_quality(financial_result.data)
    valuation = analyze_valuation(valuation_result.data if isinstance(valuation_result.data, dict) else {})
    strategy = match_strategies(task if task != "full" else "single_stock_analysis")
    counter = build_counterevidence(financial, valuation, trend, world)
    scoring = score_all(financial, valuation, trend, world, strategy, counter)
    backtest = backtest_breakout(daily.data or [])

    analyses = {
        "financial": financial,
        "valuation": valuation,
        "trend": trend,
        "candlestick": candle,
        "world": world,
        "strategy": strategy,
        "counter": counter,
        "backtest": backtest,
        "skill_data": skill_data,
    }
    return stock, analyses, scoring, counter


def input_flows_from_registry(registry: SourceRegistry, limit: int = 8) -> list[InputFlowEvidence]:
    flows: list[InputFlowEvidence] = []
    for row in registry.rows[:limit]:
        meta = classify_source(row["来源"])
        flows.append(
            InputFlowEvidence(
                name=row["模块"],
                source=row["来源"],
                source_tier=meta["tier"],
                data_date=row["时间"] or "信息不足",
                realtime=meta["freshness"],
                official=meta["official"],
                model_estimate=meta["model_estimate"],
                evidence=[f"状态：{row['状态']}"],
                missing=[] if row["问题"] == "无" else [row["问题"]],
            )
        )
    return flows


def book_citations(analyses: dict, limit: int = 6) -> list[BookSkillCitation]:
    citations: list[BookSkillCitation] = []
    for card in analyses.get("strategy", {}).get("cards", [])[:limit]:
        source = card.get("source", {})
        citations.append(
            BookSkillCitation(
                strategy_id=card.get("strategy_id", "信息不足"),
                book=source.get("book", "信息不足"),
                chapter=source.get("chapter", "信息不足"),
                page_range=source.get("page_range", "信息不足"),
                skill_type=card.get("skill_type", card.get("category", "信息不足")),
                extraction_method=source.get("extraction_method", "信息不足"),
                confidence=source.get("confidence", "信息不足"),
                usage=card.get("principle", "用于本轮研究判断的框架依据。"),
            )
        )
    return citations


def render_answer(question: str, task: str, stocks: list[dict], analyses_list: list[dict], scoring_list: list[dict], counter_list: list[dict], registry: SourceRegistry) -> str:
    is_multi = len(stocks) > 1
    if is_multi:
        rating = "放入观察"
        decision = f"本轮已对 {len(stocks)} 只候选股做横向 dry-run 比较；分数只用于排序，不代表投资结论。"
        support = [f"{s['name']}({s['code']})：{score['rating']}，综合分 {score['total']}" for s, score in zip(stocks, scoring_list)]
        uncertainty = "多股比较仍受数据缺口、行业口径、公告原文和实时估值字段影响。"
        counter = "若公告、财务、行业景气或趋势结构出现重大反证，当前排序应重新计算。"
    else:
        rating = scoring_list[0]["rating"]
        stock = stocks[0]
        decision = f"本轮对 {stock['name']}({stock['code']}) 的研究分级为：{rating}。"
        support = [f"综合分 {scoring_list[0]['total']}；详见 stock_report.md 与 data_status.md。"]
        uncertainty = counter_list[0]["strongest"]
        counter = counter_list[0]["strongest"]

    citations: list[BookSkillCitation] = []
    for analyses in analyses_list:
        for item in book_citations(analyses, limit=3):
            if item.strategy_id not in {x.strategy_id for x in citations}:
                citations.append(item)
        if len(citations) >= 8:
            break

    return render_research_answer(
        question_understanding=f"我理解你的问题是：{question}。本轮任务类型：{task}。",
        capabilities=["行情/量价分析", "财务与估值检查", "新闻/公告 world model", "核心三书 book skills", "反证复核"],
        decision=decision,
        rating=rating,
        input_flows=input_flows_from_registry(registry),
        book_skills=citations,
        support=support,
        uncertainty=uncertainty,
        counterevidence=counter,
        next_step="复核官方公告原文、当前估值/行业字段、行业指数强弱、关键支撑压力和成交量确认。",
        choices=[
            "查看单股完整报告",
            "只看支撑压力和趋势失效位",
            "补官方公告原文复核",
            "做同业横向比较",
            "用某条 book skill 做轻量回测",
        ],
    )


def run_pipeline(config_path: str | None, stock_arg: str | None, task: str, mode: str, dry_run: bool, question_arg: str | None = None) -> Path:
    config = load_config(config_path)
    output_dir = infer_output_dir(config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    query = stock_arg or config.get("query_name") or config.get("stock") or "600888"
    task = config.get("task", task)
    question = question_arg or config.get("question") or config.get("description") or f"研究 {query}"
    adapter = AKShareAdapter(dry_run=dry_run)
    registry = SourceRegistry()

    candidates = config.get("candidates") or []
    if candidates and task == "multi_stock_comparison":
        stocks: list[dict] = []
        analyses_list: list[dict] = []
        scoring_list: list[dict] = []
        counter_list: list[dict] = []
        matrix_rows: list[dict] = []
        for candidate in candidates:
            stock = _stock_from_candidate(adapter, candidate, query)
            stock, analyses, scoring, counter = analyze_one_stock(adapter, registry, stock, task, dry_run)
            stocks.append(stock)
            analyses_list.append(analyses)
            scoring_list.append(scoring)
            counter_list.append(counter)
            safe_code = stock["code"].replace(".", "_")
            write_text(output_dir / f"stock_report_{safe_code}.md", render_stock_report(stock, analyses, scoring))
            write_text(output_dir / f"final_review_{safe_code}.md", render_final_review(stock, scoring, counter))
            matrix_rows.append(
                {
                    "股票": f"{stock['name']}({stock['code']})",
                    "研究分级": scoring["rating"],
                    "综合分": scoring["total"],
                    "最大不确定性": counter["strongest"],
                    "数据完整度": scoring["scores"]["data_completeness"],
                }
            )
        # === 多股比较 Skill 增强：批量快照 ===
        skill = SkillBridge()
        if skill.available and not dry_run and skill._check_enabled("peer_comparison"):
            codes = [s["code"] for s in stocks]
            snap = skill.quote_snapshot(codes)
            if snap.ok:
                registry.add("多股比较 Skill报价快照", "成功", snap.source, snap.fetched_at or "", "无", "否")
                # 将快照数据附加到第一个 stock 的 skill_data 中，以便在报告中展示
                if analyses_list:
                    analyses_list[0]["skill_data"] = analyses_list[0].get("skill_data", {})
                    analyses_list[0]["skill_data"]["peer_snapshot"] = snap.data
            else:
                registry.add("多股比较 Skill报价快照", "失败", snap.source, snap.fetched_at or "", snap.error or "Skill未激活", "是")
        # === 多股比较 Skill 增强结束 ===
        if stocks:
            write_text(output_dir / "stock_report.md", render_stock_report(stocks[0], analyses_list[0], scoring_list[0]))
            write_text(output_dir / "final_review.md", render_final_review(stocks[0], scoring_list[0], counter_list[0]))
        write_candidate_matrix(output_dir / "candidate_matrix.xlsx", matrix_rows)
        write_text(output_dir / "answer.md", render_answer(question, task, stocks, analyses_list, scoring_list, counter_list, registry))
        registry.write(output_dir / "data_status.md")
        write_text(output_dir / "source_status.md", render_source_status())
        print(APP_PREFIX)
        print()
        print(DISCLAIMER)
        print(f"已生成报告目录：{output_dir}")
        print(f"候选数量：{len(stocks)}")
        return output_dir

    stock = resolve_stock(adapter, query, config.get("assumed_match"))
    stock, analyses, scoring, counter = analyze_one_stock(adapter, registry, stock, task if task != "full" else "single_stock_analysis", dry_run)

    write_text(output_dir / "stock_report.md", render_stock_report(stock, analyses, scoring))
    write_text(output_dir / "final_review.md", render_final_review(stock, scoring, counter))
    write_text(output_dir / "answer.md", render_answer(question, task, [stock], [analyses], [scoring], [counter], registry))
    registry.write(output_dir / "data_status.md")
    write_text(output_dir / "source_status.md", render_source_status())
    write_candidate_matrix(
        output_dir / "candidate_matrix.xlsx",
        [
            {
                "股票": f"{stock['name']}({stock['code']})",
                "研究分级": scoring["rating"],
                "综合分": scoring["total"],
                "最大不确定性": counter["strongest"],
                "数据完整度": scoring["scores"]["data_completeness"],
            }
        ],
    )
    print(APP_PREFIX)
    print()
    print(DISCLAIMER)
    print(f"已生成报告目录：{output_dir}")
    print(f"研究分级：{scoring['rating']}")
    return output_dir


def render_source_status() -> str:
    cards = load_strategy_cards()
    lines = [
        APP_PREFIX,
        "",
        "# Source Status",
        "",
        "| 策略 ID | 书名 | 章节 | 页码 | 置信度 |",
        "|---|---|---|---|---|",
    ]
    for c in cards:
        s = c["source"]
        lines.append(f"| {c['strategy_id']} | {s['book']} | {s['chapter']} | {s['page_range']} | {s['confidence']} |")
    if not cards:
        lines.append("| 信息不足 | 信息不足 | 信息不足 | 信息不足 | 信息不足 |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="A股研究Agent A 股候选股研究 Agent")
    parser.add_argument("--config", help="YAML 配置文件")
    parser.add_argument("--stock", help="股票代码或名称")
    parser.add_argument("--task", default="single_stock_analysis", help="任务类型")
    parser.add_argument("--question", help="具体研究问题，例如：这次回调是否像主力撤退？")
    parser.add_argument("--mode", default="full", help="运行模式")
    parser.add_argument("--dry-run", action="store_true", help="使用本地样例数据，不强依赖网络接口")
    args = parser.parse_args()
    run_pipeline(args.config, args.stock, args.task, args.mode, args.dry_run, args.question)


if __name__ == "__main__":
    main()
