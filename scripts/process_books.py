from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import yaml


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "ref"
RAW_DIR = ROOT / "data" / "ocr_private" / "raw_text"
PAGE_DIR = ROOT / "data" / "ocr_private" / "page_text"
CLEAN_DIR = ROOT / "data" / "book_processed" / "clean_text"
CHAPTER_DIR = ROOT / "data" / "book_processed" / "chapter_text"
BOOK_SKILL_DIR = ROOT / "book_skills"
REPORT_DIR = ROOT / "reports" / "book_extraction"


@dataclass
class StrategyTemplate:
    strategy_id: str
    name: str
    category: str
    task_fit: list[str]
    book_hint: str
    keywords: list[str]
    principle: str
    computable_rules: list[str]
    manual_checks: list[str]
    invalid_conditions: list[str]
    a_share_adaptation: list[str]
    report_sentence_template: str


TEMPLATES = [
    StrategyTemplate(
        "SPECULATION_TREND_RISK_001",
        "顺势而为并预设退出条件",
        "trend",
        ["single_stock_analysis", "multi_stock_comparison", "strategy_check"],
        "专业投机原理",
        ["趋势", "止损", "风险", "资金管理"],
        "交易判断要先识别主要趋势和风险承受边界，错误时及时退出，而不是让观点绑架仓位。",
        ["价格位于 20/60 日均线上方且均线斜率向上时记为趋势支持", "跌破 60 日均线且放量时记为趋势破坏风险"],
        ["是否已有明确的退出条件", "是否因为叙事而忽略风险"],
        ["震荡市中趋势信号频繁失真", "流动性不足或涨跌停限制导致无法按计划退出"],
        ["A 股存在涨跌停和停牌风险，止损只能作为风险预案，不能保证成交"],
        "本判断参考《专业投机原理》的趋势与风险控制思想。",
    ),
    StrategyTemplate(
        "REMINISCENCES_DISCIPLINE_001",
        "尊重市场而非固执预测",
        "psychology",
        ["single_stock_analysis", "user_discussion", "counterevidence"],
        "股票作手回忆录",
        ["市场", "趋势", "错误", "投机", "等待"],
        "当价格行为与原逻辑冲突时，应把市场反馈当作反证，而不是继续寻找理由证明自己正确。",
        [],
        ["是否把短期波动误读成确定机会", "是否能等待更清晰的证据"],
        ["缺少成交量和趋势数据时不适用", "只适合纪律检查，不适合作为单独买卖依据"],
        ["A 股消息面和涨跌停会强化短期噪声，需要与财务和公告交叉验证"],
        "本判断参考《股票作手回忆录》的市场纪律思想。",
    ),
    StrategyTemplate(
        "DOW_TREND_CONFIRM_001",
        "趋势需要指数、成交量和结构确认",
        "trend",
        ["single_stock_analysis", "multi_stock_comparison", "trend_analysis"],
        "道氏理论",
        ["主要趋势", "次级", "成交量", "平均指数", "确认"],
        "趋势判断应关注主要趋势、次级回撤和成交量确认，不能只看单日涨跌。",
        ["20/60/120 日收益和均线方向共同确认", "成交量高于 20 日均量时记为确认增强"],
        ["行业和宽基指数是否同向", "回撤是次级调整还是趋势反转"],
        ["横盘震荡或题材股急拉时容易误判", "没有成交量数据时置信度降低"],
        ["A 股行业轮动强，应增加行业相对强弱比较"],
        "本判断参考《道氏理论》的趋势确认原则。",
    ),
    StrategyTemplate(
        "TURTLE_BREAKOUT_ATR_001",
        "突破信号必须结合波动和仓位风险",
        "risk",
        ["strategy_check", "backtest_strategy", "trend_analysis"],
        "海龟交易法则",
        ["突破", "N", "波动", "头寸", "止损"],
        "突破类策略需要用波动度约束风险，不能只因价格创新高就提高确定性。",
        ["20 日突破记为短期突破", "ATR 近似 N 值，用于衡量波动风险", "突破后 5/20/60 日收益用于轻量历史验证"],
        ["突破是否伴随成交量", "是否处在大盘弱势或行业逆风中"],
        ["跳空、涨跌停、停牌会影响可执行性", "低流动性股票不适合机械突破"],
        ["A 股涨跌停限制会改变突破后的成交和止损执行"],
        "本判断参考《海龟交易法则》的突破、波动和风险控制框架。",
    ),
    StrategyTemplate(
        "CANDLE_CONFIRM_001",
        "蜡烛图只作辅助确认",
        "trend",
        ["strategy_check", "trend_analysis", "single_stock_analysis"],
        "日本蜡烛图技术",
        ["蜡烛图", "反转", "确认", "支撑", "阻挡"],
        "蜡烛图形态适合辅助识别情绪和短期反转，但必须等待趋势、成交量或后续价格确认。",
        ["识别长上影、长下影、实体方向和近 5 日价格位置", "单日反转信号只作为低权重提示"],
        ["是否出现在重要支撑阻力附近", "是否有后续确认K线"],
        ["单根K线不能作为最终结论", "缺少复权行情时形态可能失真"],
        ["A 股除权除息和涨跌停会改变K线解释，需要使用复权数据交叉检查"],
        "本判断参考《日本蜡烛图技术》的形态确认思想。",
    ),
    StrategyTemplate(
        "FINANCIAL_QUALITY_RED_FLAG_001",
        "利润必须和现金流、资产质量互相印证",
        "financial",
        ["single_stock_analysis", "multi_stock_comparison", "financial_red_flag"],
        "手把手教你读财报",
        ["现金流", "应收", "存货", "商誉", "负债", "利润"],
        "财报分析要检查利润质量、现金流匹配、应收存货和债务压力，避免只看净利润增长。",
        ["经营现金流/净利润过低记为利润质量风险", "资产负债率过高记为财务安全风险", "应收或存货增速高于收入增速记为质量风险"],
        ["是否存在关联交易或异常科目", "会计政策是否导致利润高估"],
        ["金融类公司财报结构不同，不适用普通制造业口径", "缺少报表明细时只能给信息不足"],
        ["A 股财报季存在披露滞后，需标注报告期"],
        "本判断参考《手把手教你读财报》的财务质量排雷框架。",
    ),
    StrategyTemplate(
        "MARGIN_OF_SAFETY_001",
        "安全边际优先于乐观叙事",
        "valuation",
        ["single_stock_analysis", "multi_stock_comparison", "financial_red_flag"],
        "安全边际",
        ["安全边际", "价值", "风险", "价格", "投资"],
        "价值判断要先考虑下行保护和错误空间，价格过度反映乐观预期时应降低结论强度。",
        ["PE/PB 高于自身或行业常识区间时记为估值压力", "盈利质量弱而估值高时提高反证权重"],
        ["资产价值是否可靠", "热门叙事是否已经透支价格"],
        ["周期股和亏损股估值指标可能失真", "没有可比估值数据时不作强判断"],
        ["A 股主题炒作会放大估值波动，需和财务质量、公告风险一起判断"],
        "本判断参考《安全边际》的下行风险和安全边际思想。",
    ),
]


def safe_slug(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "_", name).strip("_")


def clean_text(text: str) -> str:
    lines = []
    seen = {}
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        if re.search(r"(www\.|扫描|下载|论坛|仅供|更多电子书)", line, re.I):
            continue
        seen[line] = seen.get(line, 0) + 1
        if seen[line] <= 3:
            lines.append(line)
    return "\n".join(lines)


def extract_chapter_candidates(text: str) -> list[str]:
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if 2 <= len(line) <= 40 and re.search(r"(第[一二三四五六七八九十0-9]+[章节篇部]|目录|序言|前言|附录)", line):
            candidates.append(line)
    uniq = []
    for item in candidates:
        if item not in uniq:
            uniq.append(item)
    return uniq[:40]


def find_pages(page_texts: list[str], keywords: Iterable[str]) -> list[int]:
    pages = []
    for idx, page in enumerate(page_texts, start=1):
        if any(k in page for k in keywords):
            pages.append(idx)
    return pages[:8]


def load_ocr_page(slug: str, page_no: int) -> tuple[str, float | None]:
    text_path = PAGE_DIR / f"{slug}_page_{page_no:04d}.ocr.txt"
    status_path = PAGE_DIR / f"{slug}_page_{page_no:04d}.ocr.json"
    if not text_path.exists():
        return "", None
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    confidence = None
    if status_path.exists():
        try:
            confidence = json.loads(status_path.read_text(encoding="utf-8")).get("ocr_confidence")
        except Exception:
            confidence = None
    return text, confidence


def page_range_text(pages: list[int]) -> str:
    if not pages:
        return "未在当前可读文本中确认"
    if len(pages) == 1:
        return f"PDF 第 {pages[0]} 页"
    return f"PDF 第 {min(pages)}-{max(pages)} 页附近"


def source_chapter(chapters: list[str]) -> str:
    return chapters[0] if chapters else "章节未可靠识别"


def render_skill_doc(filename: str, title: str, strategy_ids: list[str], cards: list[dict]) -> None:
    used = [c for c in cards if c["strategy_id"] in strategy_ids]
    lines = [
        f"# {title}",
        "",
        "## 这个 skill 解决什么问题",
        "将本地书籍中可追溯的原则转化为 A 股研究中的检查清单和报告模板。",
        "",
        "## 适用任务",
    ]
    tasks = sorted({t for c in used for t in c["task_fit"]}) or ["用户答疑"]
    lines += [f"- {t}" for t in tasks]
    lines += [
        "",
        "## 不适用场景",
        "- 作为单通道、无多源确认的操作结论",
        "- 缺少来源、数据缺失或低置信 OCR 内容",
        "",
        "## 核心原则",
    ]
    lines += [f"- {c['principle']}" for c in used] or ["- 未在当前可读文本中确认"]
    lines += ["", "## 可计算信号"]
    calc = [rule for c in used for rule in c["computable_rules"]]
    lines += [f"- {x}" for x in calc] or ["- 暂无稳定可计算信号"]
    lines += ["", "## 人工判断项"]
    manual = [rule for c in used for rule in c["manual_checks"]]
    lines += [f"- {x}" for x in manual] or ["- 需要人工结合报告上下文判断"]
    lines += ["", "## 一票否决或强风险条件"]
    invalids = [rule for c in used for rule in c["invalid_conditions"]]
    lines += [f"- {x}" for x in invalids] or ["- 信息不足"]
    lines += ["", "## 与 A 股适配时的注意事项"]
    adapts = [rule for c in used for rule in c["a_share_adaptation"]]
    lines += [f"- {x}" for x in adapts] or ["- 需标注数据期和来源"]
    lines += [
        "",
        "## 报告输出模板",
        "本系统输出研究辅助型操作建议，不自动下单、不接券商、不承诺收益。判断需包含操作建议、仓位/阈值、支持证据、最大不确定性、最强反证和下一步验证信息。",
        "",
        "## 来源索引",
    ]
    for c in used:
        s = c["source"]
        lines.append(f"- 来源：{s['book']}，{s['chapter']}，{s['page_range']}，{c['strategy_id']}，{s['extraction_method']}")
    if not used:
        lines.append("- 未在当前可读文本中确认")
    (BOOK_SKILL_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [RAW_DIR, PAGE_DIR, CLEAN_DIR, CHAPTER_DIR, BOOK_SKILL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    inventory = []
    cards = []
    low_confidence = []
    ocr_failed = ["# OCR 失败或未处理页面", ""]

    for pdf in sorted(REF_DIR.glob("*.pdf")):
        slug = safe_slug(pdf.stem)
        doc = fitz.open(pdf)
        page_texts = []
        page_status = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text_len = len(text.strip())
            status = "text_layer" if text_len >= 30 else "needs_ocr"
            ocr_confidence = None
            if status == "needs_ocr":
                ocr_text, ocr_confidence = load_ocr_page(slug, i)
                if len(ocr_text.strip()) >= 30:
                    text = ocr_text
                    text_len = len(text.strip())
                    status = "ocr"
            page_texts.append(text)
            page_status.append({"page": i, "status": status, "text_length": text_len, "ocr_confidence": ocr_confidence})
            (PAGE_DIR / f"{slug}_page_{i:04d}.txt").write_text(text, encoding="utf-8", errors="ignore")
            if status == "needs_ocr":
                ocr_failed.append(f"- {pdf.name} / PDF 第 {i} 页：文本层不足，当前未启用 OCR。")
        raw_text = "\n\n".join(page_texts)
        cleaned = clean_text(raw_text)
        chapters = extract_chapter_candidates(cleaned)
        RAW_DIR.joinpath(f"{slug}.txt").write_text(raw_text, encoding="utf-8", errors="ignore")
        CLEAN_DIR.joinpath(f"{slug}.txt").write_text(cleaned, encoding="utf-8", errors="ignore")
        CHAPTER_DIR.joinpath(f"{slug}_chapters.md").write_text(
            "# 章节候选\n\n" + "\n".join(f"- {c}" for c in chapters) + "\n",
            encoding="utf-8",
        )
        text_pages = sum(1 for p in page_status if p["status"] == "text_layer")
        ocr_pages = sum(1 for p in page_status if p["status"] == "ocr")
        needs_ocr = sum(1 for p in page_status if p["status"] == "needs_ocr")
        has_text_layer = text_pages >= max(1, int(len(page_status) * 0.2))
        has_readable_text = (text_pages + ocr_pages) >= max(1, int(len(page_status) * 0.2))
        book_entry = {
            "file": pdf.name,
            "path": str(pdf),
            "size_bytes": pdf.stat().st_size,
            "pages": len(page_status),
            "text_layer_pages": text_pages,
            "ocr_pages": ocr_pages,
            "needs_ocr_pages": needs_ocr,
            "has_text_layer": has_text_layer,
            "has_readable_text": has_readable_text,
            "chapters_detected": chapters,
            "page_status_sample": page_status[:10],
        }
        inventory.append(book_entry)

        for tmpl in TEMPLATES:
            if tmpl.book_hint in pdf.stem:
                pages = find_pages(page_texts, tmpl.keywords)
                if pages:
                    method = "ocr" if any(page_status[p - 1]["status"] == "ocr" for p in pages) else "text_layer"
                    confidence = "high" if len(pages) >= 3 and method == "text_layer" and has_text_layer else "medium"
                    cards.append(
                        {
                            "strategy_id": tmpl.strategy_id,
                            "name": tmpl.name,
                            "category": tmpl.category,
                            "task_fit": tmpl.task_fit,
                            "source": {
                                "book": tmpl.book_hint,
                                "chapter": source_chapter(chapters),
                                "page_range": page_range_text(pages),
                                "extraction_method": method,
                                "confidence": confidence,
                            },
                            "principle": tmpl.principle,
                            "computable_rules": tmpl.computable_rules,
                            "manual_checks": tmpl.manual_checks,
                            "invalid_conditions": tmpl.invalid_conditions,
                            "a_share_adaptation": tmpl.a_share_adaptation,
                            "report_sentence_template": tmpl.report_sentence_template,
                        }
                    )
                else:
                    low_confidence.append(
                        {
                            "book": tmpl.book_hint,
                            "strategy_id": tmpl.strategy_id,
                            "note": "未在当前可读文本中确认",
                            "keywords": tmpl.keywords,
                        }
                    )
        doc.close()

    (ROOT / "data" / "book_inventory.yaml").write_text(
        yaml.safe_dump({"books": inventory}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (BOOK_SKILL_DIR / "source_manifest.yaml").write_text(
        yaml.safe_dump({"sources": inventory}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    # Auto cards are only extraction drafts. Do not overwrite the curated
    # strategy_cards.yaml built from deep-dive reading notes.
    (BOOK_SKILL_DIR / "auto_strategy_cards.yaml").write_text(
        yaml.safe_dump(cards, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    mapping = [{"strategy_id": c["strategy_id"], "source": c["source"], "task_fit": c["task_fit"]} for c in cards]
    (BOOK_SKILL_DIR / "auto_rule_mapping.yaml").write_text(
        yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (BOOK_SKILL_DIR / "invalid_conditions.md").write_text(
        "# 强风险和失效条件\n\n"
        + "\n".join(f"- {c['strategy_id']}：{'; '.join(c['invalid_conditions'])}" for c in cards)
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "low_confidence_notes.md").write_text(
        "# 低置信笔记\n\n"
        + "\n".join(f"- {x['book']} / {x['strategy_id']}：{x['note']}；关键词：{', '.join(x['keywords'])}" for x in low_confidence)
        + ("\n" if low_confidence else "- 暂无。\n"),
        encoding="utf-8",
    )
    (REPORT_DIR / "ocr_failed_pages.md").write_text("\n".join(ocr_failed) + "\n", encoding="utf-8")

    coverage_lines = [
        "# 书籍提取覆盖报告",
        "",
        "| 书籍 | 页数 | 文本层页 | 需 OCR 页 | 章节候选 | 已确认策略 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for b in inventory:
        related = [c["strategy_id"] for c in cards if c["source"]["book"] in b["file"]]
        coverage_lines.append(
            f"| {b['file']} | {b['pages']} | {b['text_layer_pages']} | {b['needs_ocr_pages']} | {len(b['chapters_detected'])} | {', '.join(related) or '未确认'} |"
        )
    (BOOK_SKILL_DIR / "coverage_report.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "book_extraction_summary.md").write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# 来源审计报告",
        "",
        "## 检查结论",
        f"- 正式策略卡数量：{len(cards)}",
        f"- 低置信记录数量：{len(low_confidence)}",
        "- 所有正式策略卡均包含书名、章节、页码范围、提取方式和置信度。",
        "- 未确认内容只进入 low_confidence_notes.md，不进入正式判断。",
        "",
        "## 正式策略来源",
    ]
    for c in cards:
        s = c["source"]
        audit_lines.append(f"- {c['strategy_id']}：{s['book']} / {s['chapter']} / {s['page_range']} / {s['confidence']}")
    (BOOK_SKILL_DIR / "source_audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    (BOOK_SKILL_DIR / "00_BOOK_SKILLS_README.md").write_text(
        "# Book Skills README\n\n"
        "本目录由本地 PDF 文本提取和来源审计生成。正式策略位于 `strategy_cards.yaml`，低置信内容位于 `reports/book_extraction/low_confidence_notes.md`。\n",
        encoding="utf-8",
    )
    skill_groups = {
        "01_market_regime_and_trend_skill.md": ("市场状态与趋势 skill", ["DOW_TREND_CONFIRM_001", "SPECULATION_TREND_RISK_001"]),
        "02_single_stock_evaluation_skill.md": ("单股综合评估 skill", ["FINANCIAL_QUALITY_RED_FLAG_001", "MARGIN_OF_SAFETY_001", "DOW_TREND_CONFIRM_001"]),
        "03_multi_stock_comparison_skill.md": ("多股横向比较 skill", ["FINANCIAL_QUALITY_RED_FLAG_001", "MARGIN_OF_SAFETY_001", "DOW_TREND_CONFIRM_001"]),
        "04_operator_discipline_skill.md": ("交易纪律 skill", ["REMINISCENCES_DISCIPLINE_001", "SPECULATION_TREND_RISK_001"]),
        "05_breakout_position_risk_skill.md": ("突破和仓位风险 skill", ["TURTLE_BREAKOUT_ATR_001"]),
        "06_candlestick_timing_skill.md": ("蜡烛图时机 skill", ["CANDLE_CONFIRM_001"]),
        "07_financial_statement_quality_skill.md": ("财务质量 skill", ["FINANCIAL_QUALITY_RED_FLAG_001"]),
        "08_value_margin_of_safety_skill.md": ("安全边际 skill", ["MARGIN_OF_SAFETY_001"]),
        "09_news_world_model_skill.md": ("新闻和 world model skill", ["REMINISCENCES_DISCIPLINE_001"]),
        "10_counterevidence_and_review_skill.md": ("反证和复核 skill", ["REMINISCENCES_DISCIPLINE_001", "MARGIN_OF_SAFETY_001"]),
    }
    for filename, (title, ids) in skill_groups.items():
        render_skill_doc(filename, title, ids, cards)

    print(f"processed_books={len(inventory)} strategy_cards={len(cards)} low_confidence={len(low_confidence)}")


if __name__ == "__main__":
    main()
