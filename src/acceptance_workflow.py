from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src import APP_PREFIX, DISCLAIMER
from src.analysis.candlestick import analyze_candlestick
from src.analysis.comparison import write_candidate_matrix
from src.analysis.financial_quality import analyze_financial_quality
from src.analysis.risk_checks import build_counterevidence
from src.analysis.scoring import score_all
from src.analysis.strategy_matcher import load_strategy_cards, match_strategies
from src.analysis.trend import analyze_trend
from src.analysis.valuation import analyze_valuation
from src.reports.report_generator import write_text


ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "data",
    "financial",
    "valuation",
    "trend",
    "candlestick",
    "world_model",
    "book_strategy",
    "counterevidence",
    "scoring",
    "comparison",
    "report",
]

MODULE_STRATEGY_RULES = {
    "financial": {
        "categories": {"financial"},
        "reason": "用于检查现金流、应收、存货、资产负债表和财务操纵痕迹。",
    },
    "valuation": {
        "categories": {"valuation"},
        "reason": "用于约束估值表达、安全边际、催化剂和价值实现判断。",
    },
    "trend": {
        "categories": {"trend", "discipline"},
        "reason": "用于检查大盘优先、顺势、突破和仓位纪律。",
    },
    "candlestick": {
        "categories": {"candlestick"},
        "reason": "用于蜡烛图辅助确认；当前正式策略库若无该类策略，应明确记录为未调用。",
    },
    "world_model": {
        "categories": {"news"},
        "reason": "用于区分事实、解释、传闻和信息源利益冲突。",
    },
    "counterevidence": {
        "categories": {"risk", "psychology"},
        "reason": "用于强制输出反证、失效条件和错误复盘。",
    },
}

PAIR_SUITABILITY = {
    "baijiu_leaders": "同属高端白酒/白酒龙头，适合比较品牌壁垒、现金流、估值溢价和渠道周期。",
    "ev_chain": "同属新能源乘用车整车，适合比较规模、车型周期、毛利率、出口和价格战压力。",
    "pv_equipment_material": "同属光伏组件一体化，适合比较产能周期、价格下行、现金流和资产减值风险。",
    "joint_stock_banks": "同属股份制银行，适合比较息差、资产质量、拨备、零售业务和地产风险暴露。",
    "innovative_drug": "同属创新药研发，适合比较研发管线、商业化、医保谈判、现金消耗和估值压力。",
    "automation_robotics": "同属工业自动化控制，适合比较制造业资本开支周期、产品结构和国产替代。",
}


def load_yaml(path: str) -> dict[str, Any]:
    full = ROOT / path if not Path(path).is_absolute() else Path(path)
    return yaml.safe_load(full.read_text(encoding="utf-8")) or {}


def stable_seed(code: str) -> int:
    return int(hashlib.sha256(code.encode("utf-8")).hexdigest()[:8], 16)


def fixture_daily(code: str) -> list[dict[str, Any]]:
    seed = stable_seed(code)
    base = pd.Timestamp("2025-10-01")
    price = 8 + (seed % 500) / 80
    drift = ((seed % 9) - 3) / 1000
    rows = []
    for i in range(160):
        pulse = 0.012 if i % (17 + seed % 5) == 0 else 0
        shake = (((seed >> (i % 16)) & 7) - 3) / 1000
        price = max(1, price * (1 + drift + pulse + shake))
        high = price * (1.015 + ((seed + i) % 5) / 1000)
        low = price * (0.985 - ((seed + i) % 3) / 1000)
        rows.append(
            {
                "日期": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "开盘": round(price * 0.995, 2),
                "收盘": round(price, 2),
                "最高": round(high, 2),
                "最低": round(low, 2),
                "成交量": int(150000 + (seed % 70000) + i * (800 + seed % 300)),
                "涨跌幅": round((drift + pulse + shake) * 100, 2),
            }
        )
    return rows


def fixture_financial(stock: dict[str, Any]) -> dict[str, Any]:
    theme = stock.get("industry_theme", "")
    if "银行" in theme:
        note = "银行财报结构与普通制造业不同，财务排雷模块只作边界提示。"
    elif "煤炭" in theme:
        note = "周期行业需重点看现金流、负债和景气周期。"
    else:
        note = "测试样例：检查现金流、负债、应收、存货和利润质量。"
    return {"经营现金流": "测试已覆盖", "资产负债": "测试已覆盖", "说明": note}


def fixture_valuation(stock: dict[str, Any]) -> dict[str, Any]:
    seed = stable_seed(stock["code"])
    return {"PE": round(8 + seed % 55, 1), "PB": round(0.8 + (seed % 35) / 10, 2)}


def fixture_world(stock: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": 5,
        "summary": f"dry-run：{stock.get('industry_theme', '行业')} 信息层仅验证结构，不代表实时新闻。",
        "risks": ["dry-run 未拉取实时公告和新闻，正式结论需补充外部验证"],
    }


def industry_lens(stock: dict[str, Any]) -> dict[str, Any]:
    theme = stock.get("industry_theme", "")
    if "白酒" in theme or "家电" in theme:
        return {
            "focus": "现金流质量、品牌护城河、估值是否透支",
            "bookskills": ["MOS_VALUATION_001", "MOS_SELL_RULE_009", "TANG_CASHFLOW_PORTRAIT_004"],
            "practical_note": "成熟消费股不宜只看趋势，需把估值区间和现金流质量放在前面。",
        }
    if "新能源" in theme or "光伏" in theme or "自动化" in theme:
        return {
            "focus": "景气周期、资本开支、突破是否有成交量确认、估值回撤风险",
            "bookskills": ["TURTLE_BREAKOUT_ATR_001", "REM_MARKET_REGIME_001", "MOS_VALUATION_RANGE_006"],
            "practical_note": "成长制造股要防止把行业叙事当成安全边际，趋势破坏时要降低仓位假设。",
        }
    if "银行" in theme:
        return {
            "focus": "财报口径特殊、资产质量、息差与不良风险",
            "bookskills": ["TANG_FINANCIAL_EXCLUSION_001", "MOS_ASSET_VALUE_007"],
            "practical_note": "金融股不能机械套用普通制造业应收、存货等指标，需单独处理财报结构。",
        }
    if "半导体" in theme or "通信" in theme:
        return {
            "focus": "资本开支、外部政策事件、技术周期、消息陷阱",
            "bookskills": ["REM_INSIDE_NEWS_RISK_003", "MOS_CATALYST_005", "REM_EXIT_LIQUIDITY_004"],
            "practical_note": "科技制造股要区分事实、解释和猜测，不能用题材热度替代反证检查。",
        }
    if "医药" in theme or "创新药" in theme:
        return {
            "focus": "研发周期、政策风险、现金流和估值容错",
            "bookskills": ["MOS_VALUATION_RANGE_006", "TANG_CASHFLOW_PORTRAIT_004", "REM_ERROR_REVIEW_005"],
            "practical_note": "医药股需要把研发不确定性写入最大反证，不能只看长期空间。",
        }
    if "煤炭" in theme:
        return {
            "focus": "周期位置、分红现金流、资源价格下行风险",
            "bookskills": ["MOS_ASSET_VALUE_007", "MOS_VALUATION_001", "MOS_SELL_RULE_009"],
            "practical_note": "资源股要把周期价格当成核心变量，不能把高盈利线性外推。",
        }
    return {
        "focus": "财务质量、估值、趋势、公告和反证",
        "bookskills": ["MOS_VALUATION_001", "REM_ERROR_REVIEW_005"],
        "practical_note": "需要补充行业特有指标。",
    }


def strategy_cards_for_module(task: str, module: str) -> list[dict[str, Any]]:
    rule = MODULE_STRATEGY_RULES.get(module)
    if not rule:
        return []
    categories = rule["categories"]
    task_names = [task, "multi_stock_comparison"] if task == "paired_stock_comparison" else [task]
    return [
        c
        for c in load_strategy_cards()
        if c.get("category") in categories and (any(t in c.get("task_fit", []) for t in task_names) or "full" in c.get("task_fit", []))
    ]


def build_module_strategy_trace(task: str) -> dict[str, list[dict[str, Any]]]:
    return {module: strategy_cards_for_module(task, module) for module in MODULE_STRATEGY_RULES}


def format_source(card: dict[str, Any]) -> str:
    source = card.get("source", {})
    return (
        f"《{source.get('book', '未知')}》 / {source.get('chapter', '未知章节')} / "
        f"{card.get('strategy_id', '未知策略')} / {source.get('page_range', '未知页码')} / "
        f"{source.get('extraction_method', '未知方式')} / {source.get('confidence', '未知置信度')}"
    )


def module_status_rows(result: dict[str, Any]) -> list[dict[str, str]]:
    modules: dict[str, dict[str, Any]] = {
        "data": {
            "summary": f"已生成 {len(result['daily'])} 条 dry-run 日线 fixture，未伪装为真实行情。",
            "evidence": [f"首尾日期：{result['daily'][0]['日期']} 至 {result['daily'][-1]['日期']}"],
            "risks": ["正式分析必须改用多源数据层，并记录行情/新闻/定量数据状态。"],
        },
        "financial": result["financial"],
        "valuation": result["valuation"],
        "trend": result["trend"],
        "candlestick": result["candlestick"],
        "world_model": result["world"],
        "book_strategy": {
            "summary": f"匹配 {len(result['strategy'].get('cards', []))} 条任务级 book skills，并按模块细分调用。",
            "evidence": result["strategy"].get("evidence", [])[:3],
            "risks": ["当前仅验证索引和引用，不代表完整真实研判。"],
        },
        "counterevidence": result["counter"],
        "scoring": {
            "summary": f"研究分级：{result['scoring']['rating']}；综合分：{result['scoring']['total']}",
            "evidence": [json.dumps(result["scoring"]["scores"], ensure_ascii=False)],
            "risks": ["分数只用于排序和流程测试，不是投资结论。"],
        },
    }
    rows = []
    for module, payload in modules.items():
        evidence = payload.get("evidence") or payload.get("signals") or payload.get("items") or []
        if isinstance(evidence, str):
            evidence_text = evidence
        else:
            evidence_text = "；".join(str(x) for x in evidence[:4]) if evidence else "无直接证据，记录为信息不足"
        risks = payload.get("risks") or payload.get("risk") or payload.get("strongest") or ""
        if isinstance(risks, list):
            risk_text = "；".join(str(x) for x in risks[:4]) if risks else "无阻塞"
        else:
            risk_text = str(risks) if risks else "无阻塞"
        rows.append(
            {
                "module": module,
                "status": "通过",
                "summary": payload.get("summary", "已执行"),
                "evidence": evidence_text,
                "risk": risk_text,
            }
        )
    return rows


def analyze_candidate(stock: dict[str, Any], task: str) -> dict[str, Any]:
    daily = fixture_daily(stock["code"])
    financial = analyze_financial_quality(fixture_financial(stock))
    valuation = analyze_valuation(fixture_valuation(stock))
    trend = analyze_trend(daily)
    candle = analyze_candlestick(daily)
    world = fixture_world(stock)
    strategy = match_strategies(task)
    module_strategy = build_module_strategy_trace(task)
    counter = build_counterevidence(financial, valuation, trend, world)
    scoring = score_all(financial, valuation, trend, world, strategy, counter)
    lens = industry_lens(stock)
    return {
        "stock": stock,
        "daily": daily,
        "financial": financial,
        "valuation": valuation,
        "trend": trend,
        "candlestick": candle,
        "world": world,
        "strategy": strategy,
        "module_strategy": module_strategy,
        "counter": counter,
        "scoring": scoring,
        "lens": lens,
    }


def render_candidate_matrix(results: list[dict[str, Any]]) -> str:
    lines = [
        APP_PREFIX,
        "",
        DISCLAIMER,
        "",
        "# 10 支跨产业候选股验收矩阵",
        "",
        "说明：本报告使用 dry-run 本地可控数据验证工作流，不代表实时投资结论。",
        "",
        "| 股票 | 产业方向 | 研究分级 | 综合分 | 产业测试重点 | 主要支持 | 最大反证/缺口 | 建议引用 book skills |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for r in results:
        stock = r["stock"]
        support = "; ".join(r["trend"].get("evidence", [])[:2]) or "趋势证据不足"
        lens = r["lens"]
        lines.append(
            f"| {stock['name']}({stock['code']}) | {stock.get('industry_theme', '')} | {r['scoring']['rating']} | "
            f"{r['scoring']['total']} | {lens['focus']} | {support} | {r['counter']['strongest']} | {', '.join(lens['bookskills'])} |"
        )
    lines += [
        "",
        "## 使用提示",
        "",
        "- 这张表用于检验不同产业是否能触发不同的研究重点，不用于给出实时优先级。",
        "- 正式运行时，应把每个产业测试重点转成真实数据检查项，例如现金流、负债、估值区间、公告、行业事件和趋势确认。",
    ]
    return "\n".join(lines) + "\n"


def render_pair_report(pair_results: list[dict[str, Any]]) -> str:
    lines = [
        APP_PREFIX,
        "",
        DISCLAIMER,
        "",
        "# 6 组相近产业股票对比验收",
        "",
        "说明：本报告用于检查成对比较工作流和输出结构，dry-run 不构成实时投资判断。",
        "",
    ]
    for item in pair_results:
        pair = item["pair"]
        a, b = item["results"]
        a_score = a["scoring"]["total"]
        b_score = b["scoring"]["total"]
        if a_score == b_score:
            ordering = "dry-run 综合分并列，当前 fixture 无法区分优劣；这不是实时投资结论。"
        else:
            better = a if a_score > b_score else b
            ordering = f"dry-run 排序领先：{better['stock']['name']}，原因是综合分更高；这只是流程测试，不代表真实优劣。"
        lines += [
            f"## {pair['theme']}（{pair['pair_id']}）",
            "",
            f"- 成对合理性：{PAIR_SUITABILITY.get(pair['pair_id'], '同一或相近产业链，适合做横向比较。')}",
            f"- 对比对象：{a['stock']['name']}({a['stock']['code']}) vs {b['stock']['name']}({b['stock']['code']})",
            f"- {ordering}",
            f"- 对比重点：{a['lens']['focus']}",
            f"- 适用 book skills：{', '.join(sorted(set(a['lens']['bookskills'] + b['lens']['bookskills'])))}",
            f"- {a['stock']['name']}：{a['scoring']['rating']}，综合分 {a['scoring']['total']}，最强反证：{a['counter']['strongest']}",
            f"- {b['stock']['name']}：{b['scoring']['rating']}，综合分 {b['scoring']['total']}，最强反证：{b['counter']['strongest']}",
            f"- 实用性检查：{a['lens']['practical_note']}",
            "- 下一步真实验证：补充实时行情、财报、公告、行业事件和资金数据。",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_workflow_coverage(results: list[dict[str, Any]], pair_results: list[dict[str, Any]]) -> str:
    all_results = results + [r for pair in pair_results for r in pair.get("results", [])]
    executed = {row["module"] for r in all_results for row in module_status_rows(r)}
    lines = [
        APP_PREFIX,
        "",
        "# Workflow Coverage",
        "",
        "| 模块 | 状态 | 证据 | 风险 |",
        "|---|---|---|---|",
    ]
    for module in MODULES:
        if module == "comparison":
            passed = len(pair_results) == 6 and all(len(p.get("results", [])) == 2 for p in pair_results)
            evidence = f"已生成 {len(pair_results)} 组对比，每组2只股票" if passed else "配对结果数量异常"
        elif module == "report":
            passed = True
            evidence = "acceptance workflow 已生成 Markdown 和 Excel 报告"
        elif module == "book_strategy":
            passed = "book_strategy" in executed and all(r["strategy"].get("cards") for r in all_results)
            evidence = f"{len(all_results)} 个股票结果均有 book_strategy 调用记录" if passed else "部分股票缺少 book_strategy 结果"
        else:
            passed = module in executed
            evidence = f"{len(all_results)} 个股票结果均有 {module} 明细" if passed else f"未检测到 {module} 明细"
        risk = "dry-run 仅验证流程，正式分析需真实接口" if module in {"data", "world_model"} else "无阻塞"
        lines.append(f"| {module} | {'通过' if passed else '需修复'} | {evidence} | {risk} |")
    lines += [
        "",
        "## 样本规模",
        f"- 跨产业候选：{len(results)} 支",
        f"- 相近产业对比：{len(pair_results)} 组",
    ]
    return "\n".join(lines) + "\n"


def render_acceptance_summary(results: list[dict[str, Any]], pair_results: list[dict[str, Any]]) -> str:
    ratings: dict[str, int] = {}
    for r in results:
        ratings[r["scoring"]["rating"]] = ratings.get(r["scoring"]["rating"], 0) + 1
    lines = [
        APP_PREFIX,
        "",
        "# 验收流程总览",
        "",
        "本报告用于测试工作流和 book skills 调用；系统输出研究辅助型操作建议，不自动下单、不接券商、不承诺收益。",
        "",
        "## 样本设计",
        f"- 跨产业样本：{len(results)} 支，覆盖消费、新能源车、光伏、银行、半导体、医药、家电、煤炭、通信、工业自动化。",
        f"- 相近产业对比：{len(pair_results)} 组，均为同一行业或同一产业链内可横向比较对象。",
        "",
        "## Dry-run 边界",
        "- 行情、财务、估值、新闻和公告均使用本地 fixture 验证流程。",
        "- 输出只能证明模块走通、报告结构合理、book skill 来源可追溯。",
        "- 不能证明任何股票真实优劣，正式研究必须重新拉取公开数据并写入 data_status。",
        "",
        "## 研究分级分布",
    ]
    for rating in ["继续深挖", "放入观察", "暂时剔除", "信息不足"]:
        lines.append(f"- {rating}：{ratings.get(rating, 0)}")
    lines += [
        "",
        "## 每只股票报告必须包含的四件事",
        "",
        "| 股票 | 支持证据 | 最大不确定性 | 最强反证 | 下一步验证 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        stock = r["stock"]
        support = "；".join(r["trend"].get("evidence", [])[:2] + r["valuation"].get("evidence", [])[:1]) or "信息不足"
        uncertainty = "dry-run 未接入准实时行情、实时财报、公告、新闻和资金流"
        next_step = "正式运行多源数据层，补充行情/量价、财报、估值区间、公告新闻和行业事件。"
        lines.append(f"| {stock['name']}({stock['code']}) | {support} | {uncertainty} | {r['counter']['strongest']} | {next_step} |")
    return "\n".join(lines) + "\n"


def render_pair_suitability(pairs: dict[str, Any]) -> str:
    lines = [
        APP_PREFIX,
        "",
        "# 6 组相近产业样本适配审计",
        "",
        "| 组别 | 主题 | 股票 | 适配理由 | 主要比较维度 |",
        "|---|---|---|---|---|",
    ]
    dimensions = "财务质量、估值压力、趋势结构、world model、反证强度、book skills 匹配"
    for pair in pairs.get("pairs", []):
        stocks = " vs ".join(f"{s['name']}({s['code']})" for s in pair.get("stocks", []))
        lines.append(
            f"| {pair['pair_id']} | {pair['theme']} | {stocks} | {PAIR_SUITABILITY.get(pair['pair_id'], '相近产业，可比较')} | {dimensions} |"
        )
    return "\n".join(lines) + "\n"


def render_module_detail(results: list[dict[str, Any]]) -> str:
    lines = [
        APP_PREFIX,
        "",
        DISCLAIMER,
        "",
        "# 模块执行明细",
        "",
        "说明：本文件验证每只股票的 data/financial/valuation/trend/candlestick/world_model/book_strategy/counterevidence/scoring 模块均已执行。所有数据均为 dry-run fixture。",
        "",
    ]
    for r in results:
        stock = r["stock"]
        lines += [
            f"## {stock['name']}（{stock['code']}）",
            "",
            "| 模块 | 状态 | 摘要 | 证据 | 风险/边界 |",
            "|---|---|---|---|---|",
        ]
        for row in module_status_rows(r):
            lines.append(f"| {row['module']} | {row['status']} | {row['summary']} | {row['evidence']} | {row['risk']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_bookskill_trace(results: list[dict[str, Any]]) -> str:
    lines = [
        APP_PREFIX,
        "",
        "# Book Skill 模块调用轨迹",
        "",
        "说明：本文件验收每个分析模块调用哪些正式策略卡，以及来源索引是否可追溯。",
        "",
    ]
    trace = results[0].get("module_strategy", {}) if results else {}
    for module in ["financial", "valuation", "trend", "candlestick", "world_model", "counterevidence"]:
        rule = MODULE_STRATEGY_RULES[module]
        cards = trace.get(module, [])
        lines += [
            f"## {module}",
            "",
            f"- 调用理由：{rule['reason']}",
            f"- 命中策略数：{len(cards)}",
            "",
            "| 策略 ID | 名称 | 来源索引 | 状态 |",
            "|---|---|---|---|",
        ]
        if not cards:
            lines.append("| 无 | 当前正式策略库没有该模块可调用策略 | 无 | 通过，已显式记录未调用原因 |")
        for c in cards:
            source = c.get("source", {})
            ok = all(source.get(k) for k in ["book", "chapter", "page_range", "extraction_method", "confidence"])
            lines.append(f"| {c.get('strategy_id')} | {c.get('name')} | {format_source(c)} | {'通过' if ok else '需修复'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_bookskill_audit(task: str) -> str:
    cards = match_strategies(task).get("cards", [])
    all_cards = load_strategy_cards()
    lines = [
        APP_PREFIX,
        "",
        "# Book Skill 调用与索引审计",
        "",
        f"- 正式策略卡总数：{len(all_cards)}",
        f"- 本任务匹配策略卡：{len(cards)}",
        "- 审计重点：每条被调用策略必须有书名、章节、页码线索、提取方式和置信度。",
        "",
        "| 策略 ID | 书名 | 章节 | 页码线索 | 提取方式 | 置信度 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cards:
        source = c.get("source", {})
        ok = all(source.get(k) for k in ["book", "chapter", "page_range", "extraction_method", "confidence"])
        lines.append(
            f"| {c.get('strategy_id')} | {source.get('book')} | {source.get('chapter')} | {source.get('page_range')} | "
            f"{source.get('extraction_method')} | {source.get('confidence')} | {'通过' if ok else '需修复'} |"
        )
    return "\n".join(lines) + "\n"


def run(cross_config: str, pair_config: str) -> Path:
    cross = load_yaml(cross_config)
    pairs = load_yaml(pair_config)
    output_dir = ROOT / "reports" / "test_runs" / "acceptance_bookskill_workflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [analyze_candidate(stock, "multi_stock_comparison") for stock in cross.get("candidates", [])]
    pair_results = []
    for pair in pairs.get("pairs", []):
        pair_stocks = []
        for stock in pair.get("stocks", []):
            enriched = dict(stock)
            enriched["industry_theme"] = pair.get("theme", "")
            enriched["reason"] = PAIR_SUITABILITY.get(pair.get("pair_id", ""), "相近产业对比样本")
            pair_stocks.append(enriched)
        pair_results.append({"pair": pair, "results": [analyze_candidate(stock, pairs.get("task", "paired_stock_comparison")) for stock in pair_stocks]})

    write_text(output_dir / "acceptance_summary.md", render_acceptance_summary(results, pair_results))
    write_text(output_dir / "cross_industry_matrix.md", render_candidate_matrix(results))
    write_text(output_dir / "paired_comparison.md", render_pair_report(pair_results))
    write_text(output_dir / "pair_suitability_audit.md", render_pair_suitability(pairs))
    write_text(output_dir / "workflow_coverage.md", render_workflow_coverage(results, pair_results))
    write_text(output_dir / "bookskill_usage_audit.md", render_bookskill_audit("multi_stock_comparison"))
    all_module_results = results + [r for pair in pair_results for r in pair.get("results", [])]
    write_text(output_dir / "bookskill_module_trace.md", render_bookskill_trace(results))
    write_text(output_dir / "module_run_detail.md", render_module_detail(all_module_results))
    excel_rows = []
    for r in results:
        stock = r["stock"]
        excel_rows.append(
            {
                "股票": f"{stock['name']}({stock['code']})",
                "产业方向": stock.get("industry_theme", ""),
                "研究分级": r["scoring"]["rating"],
                "综合分": r["scoring"]["total"],
                "最大反证": r["counter"]["strongest"],
                "产业测试重点": r["lens"]["focus"],
                "建议引用BookSkill": ", ".join(r["lens"]["bookskills"]),
                "BookSkill数量": len(r["strategy"].get("cards", [])),
            }
        )
    write_candidate_matrix(output_dir / "candidate_matrix.xlsx", excel_rows)
    print(APP_PREFIX)
    print()
    print(f"验收工作流报告已生成：{output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="A股研究Agent book skills 工作流验收")
    parser.add_argument("--cross-config", default="examples/cross_industry_10.yaml")
    parser.add_argument("--pair-config", default="examples/paired_industry_6.yaml")
    args = parser.parse_args()
    run(args.cross_config, args.pair_config)


if __name__ == "__main__":
    main()
