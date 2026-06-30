from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEEP_DIVE_DIR = ROOT / "book_skills" / "deep_dive"
CORE_DIR = ROOT / "book_skills" / "core"


BOOK_ORDER = [
    "专业投机原理",
    "日本蜡烛图技术",
    "道氏理论",
]


@dataclass
class RuleRow:
    strategy_id: str
    kind: str
    book: str
    category: str
    principle: str
    source: str
    computable_rule: str
    manual_check: str
    invalid_condition: str
    a_share_adaptation: str
    formal_flag: str


def split_md_table_row(line: str) -> list[str]:
    raw = line.strip()
    if not raw.startswith("|"):
        return []
    cells = [cell.strip() for cell in raw.strip("|").split("|")]
    return cells


def infer_book(path: Path) -> str:
    name = path.name
    for book in BOOK_ORDER:
        if book in name:
            return book
    raise ValueError(f"无法识别书名: {path}")


def infer_kind(strategy_id: str) -> str | None:
    if strategy_id.startswith(("DOW-A", "PPS-M", "CANDLE_MACRO")):
        return "macro"
    if strategy_id.startswith(("DOW-B", "PPS-Q", "CANDLE_Q")):
        return "quantitative"
    return None


def extract_rows(path: Path) -> list[RuleRow]:
    book = infer_book(path)
    rows: list[RuleRow] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = split_md_table_row(line)
        if len(cells) < 9:
            continue
        strategy_id = cells[0]
        kind = infer_kind(strategy_id)
        if not kind:
            continue
        rows.append(
            RuleRow(
                strategy_id=strategy_id,
                kind=kind,
                book=book,
                category=cells[1],
                principle=cells[2],
                source=cells[3],
                computable_rule=cells[4],
                manual_check=cells[5],
                invalid_condition=cells[6],
                a_share_adaptation=cells[7],
                formal_flag=cells[8],
            )
        )
    return rows


def source_chapter(source: str) -> str:
    if "，" in source:
        return source.split("，", 1)[0].replace("《", "").replace("》", "")
    if "/" in source:
        parts = source.split("/")
        return parts[1] if len(parts) > 1 else source
    return source[:80]


def source_page_range(source: str) -> str:
    tokens = []
    for marker in ["OCR_PAGE", "OCR_PAGE_", "OCR_PAGE "]:
        if marker in source:
            idx = source.find(marker)
            tokens.append(source[idx:])
            break
    return tokens[0] if tokens else "页码线索见来源字段；需人工复核"


def confidence_for(row: RuleRow) -> str:
    text = " ".join(
        [
            row.formal_flag,
            row.source,
            row.manual_check,
            row.invalid_condition,
            row.a_share_adaptation,
        ]
    )
    if "否" in row.formal_flag or "暂缓" in row.formal_flag:
        return "low"
    if "候选" in row.formal_flag or "部分" in row.formal_flag or "需复核" in text or "冲突" in text:
        return "medium"
    return "high"


def task_fit_for(row: RuleRow) -> list[str]:
    fits = ["strategy_check", "trend_analysis", "single_stock_analysis"]
    if row.kind == "macro":
        fits += ["market_regime", "user_discussion"]
    else:
        fits += ["backtest_strategy"]
    if any(key in row.principle for key in ["相对强弱", "个股", "行业", "指数确认"]):
        fits.append("multi_stock_comparison")
    return sorted(set(fits))


def yaml_card(row: RuleRow) -> dict:
    return {
        "strategy_id": row.strategy_id,
        "name": row.principle[:42],
        "category": "macro" if row.kind == "macro" else "technical",
        "priority": 1,
        "skill_type": "宏观策略和思想" if row.kind == "macro" else "具体定量策略和分析标准",
        "task_fit": task_fit_for(row),
        "source": {
            "book": row.book,
            "chapter": source_chapter(row.source),
            "page_range": source_page_range(row.source),
            "raw_source": row.source,
            "extraction_method": "full_ocr_txt_deep_dive",
            "confidence": confidence_for(row),
        },
        "principle": row.principle,
        "computable_rules": [] if row.computable_rule in {"无", "[]"} else [row.computable_rule],
        "manual_checks": [] if row.manual_check in {"无", "[]"} else [row.manual_check],
        "invalid_conditions": [] if row.invalid_condition in {"无", "[]"} else [row.invalid_condition],
        "a_share_adaptation": [] if row.a_share_adaptation in {"无", "[]"} else [row.a_share_adaptation],
        "formal_status": row.formal_flag,
        "report_sentence_template": (
            f"本判断参考《{row.book}》的 {row.strategy_id}：{row.principle}"
        ),
    }


def is_formal_card(row: RuleRow) -> bool:
    if confidence_for(row) == "low":
        return False
    formal = row.formal_flag
    return formal.startswith("是")


def md_table(rows: list[RuleRow], kind: str) -> str:
    header = (
        "| ID | 核心书 | 类别 | 判据/策略 | 来源章节与页码线索 | "
        "可计算规则 | 人工判断项 | 失效条件 | A股适配 | 正式状态 | 置信度 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                [
                    row.strategy_id,
                    row.book,
                    row.category,
                    row.principle.replace("|", "/"),
                    row.source.replace("|", "/"),
                    row.computable_rule.replace("|", "/"),
                    row.manual_check.replace("|", "/"),
                    row.invalid_condition.replace("|", "/"),
                    row.a_share_adaptation.replace("|", "/"),
                    row.formal_flag,
                    confidence_for(row),
                ]
            )
            + " |"
        )
    title = "宏观策略和思想" if kind == "macro" else "具体定量策略和分析标准"
    intro = (
        f"# {title}\n\n"
        "本文件由 `scripts/build_core_book_skills.py` 从三本核心书 deep dive 机械抽取生成。\n"
        "上游文本来自全书 OCR 合并 txt 与逐章通读整理，正式引用时仍要保留书名、章节、OCR_PAGE 线索和置信度。\n\n"
        "核心书优先级：`专业投机原理`、`日本蜡烛图技术`、`道氏理论`。\n\n"
    )
    rules = (
        "## Reference-only 使用硬规则\n\n"
        "- DeepSeek/Agent 决策前默认只读取 `config/agent_workflow_strategy.yaml` 中的 `default_evidence_pack_files`，不整文件读取本总表。\n"
        "- 本总表只用于人工复核、离线 grounding、来源追踪或重建 `book_skills/grounded_skill_cards.yaml`；正式报告如引用 Book Skill，必须来自 grounded cards 或经人工复核后的策略 ID。\n"
        "- `正式状态` 为 `否`、`暂缓` 或置信度为 `low` 的行，只能作为覆盖审计和反证提醒，不能作为正式判断依据。\n"
        "- 任何趋势、支撑压力、高抛低吸、回撤、突破、反转、资金情绪判断，都必须优先经过 grounded cards、失效条件和反证清单；本文件只提供可追溯来源和离线扩展土壤。\n"
        "- 本文件给研究依据和启发，不能单独触发买入/卖出/加减仓，最终操作建议必须由多通道 Agent 综合生成。\n"
        "- 若书中没有明确阈值，不得把 A 股工程化适配写成原书阈值。\n"
        "- 每次报告必须写支持证据、最大不确定性、最强反证和下一步验证信息。\n\n"
        "## 判据总表\n\n"
    )
    return intro + rules + header + "\n".join(body) + "\n"


def workflow_note(macro_rows: list[RuleRow], quantitative_rows: list[RuleRow]) -> str:
    return f"""# Core Book Skills Reference

本目录是 Book Skill 的 reference-only 判据库。当前三本核心书已完成全书 OCR、合并 txt、deep dive 通读整理，并由构建脚本抽取为判据总表。为保持用户端工作流轻量，deep dive 和自动生成草稿已归档到项目外；本目录只保留来源、覆盖审计、失效限制和离线 grounding 所需信息。

## 两类 Reference-only 判据

1. `macro_principles.md`：宏观策略和思想，共 {len(macro_rows)} 条。
2. `quantitative_rules.md`：具体定量策略和分析标准，共 {len(quantitative_rows)} 条。

## 决策调用边界

- DeepSeek/Agent 决策前默认只读取 `config/agent_workflow_strategy.yaml` 中的 `default_evidence_pack_files`，不整文件读取本目录的大表。
- 本目录只用于人工复核、离线 grounding、补充来源追踪或重建 `book_skills/grounded_skill_cards.yaml`。
- 正式报告如引用 Book Skill，必须来自 grounded cards 或经人工复核后的策略 ID，并写明：书名、章节/OCR_PAGE、提取方式、置信度、适用性和失效条件。
- 蜡烛图只作为时机与验证工具，不能单独决定研究分级。
- 道氏理论用于市场/行业/个股的趋势层级和确认框架。
- 《专业投机原理》用于风险优先、趋势改变、2B、1-2-3、赔率、仓位风险尺度和宏观信用周期框架。
- `正式状态` 为 `否`、`暂缓` 或置信度为 `low` 的条目只能作为覆盖审计和反证提醒，不得作为正式判断依据。

## 上游文件

- `book_skills/deep_dive/专业投机原理_deep_dive.md`
- `book_skills/deep_dive/日本蜡烛图技术_deep_dive.md`
- `book_skills/deep_dive/道氏理论_deep_dive.md`
- `data/book_processed/clean_text/专业投机原理_珍藏版.txt`
- `data/book_processed/clean_text/日本蜡烛图技术.txt`
- `data/book_processed/clean_text/道氏理论_高清.txt`

## 生成命令

```bash
python scripts/build_core_book_skills.py
```
"""


def main() -> None:
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[RuleRow] = []
    for book in BOOK_ORDER:
        matches = [p for p in DEEP_DIVE_DIR.glob("*_deep_dive.md") if book in p.name]
        if not matches:
            raise FileNotFoundError(f"缺少 deep dive: {book}")
        all_rows.extend(extract_rows(matches[0]))

    macro_rows = [row for row in all_rows if row.kind == "macro"]
    quantitative_rows = [row for row in all_rows if row.kind == "quantitative"]

    (CORE_DIR / "macro_principles.md").write_text(md_table(macro_rows, "macro"), encoding="utf-8")
    (CORE_DIR / "quantitative_rules.md").write_text(
        md_table(quantitative_rows, "quantitative"), encoding="utf-8"
    )
    (CORE_DIR / "README.md").write_text(
        workflow_note(macro_rows, quantitative_rows), encoding="utf-8"
    )

    formal_rows = [row for row in all_rows if is_formal_card(row)]
    low_rows = [row for row in all_rows if not is_formal_card(row)]
    cards = [yaml_card(row) for row in formal_rows]
    low_cards = [yaml_card(row) for row in low_rows]
    (CORE_DIR / "core_strategy_cards.yaml").write_text(
        yaml.safe_dump(cards, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (CORE_DIR / "low_confidence_or_deferred_cards.yaml").write_text(
        yaml.safe_dump(low_cards, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    strategy_cards = ROOT / "book_skills" / "strategy_cards.yaml"
    supporting_cards = ROOT / "book_skills" / "supporting_strategy_cards.yaml"
    if strategy_cards.exists() and not supporting_cards.exists():
        shutil.copy2(strategy_cards, supporting_cards)
    strategy_cards.write_text(
        "# 本文件由 scripts/build_core_book_skills.py 生成，核心三书优先。\n"
        "# 旧版非核心策略卡首次生成时已备份到 supporting_strategy_cards.yaml。\n"
        "# 低置信、否决、暂缓条目不进入本正式策略卡，见 book_skills/core/low_confidence_or_deferred_cards.yaml。\n"
        + yaml.safe_dump(cards, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"generated macro={len(macro_rows)} quantitative={len(quantitative_rows)} formal_cards={len(cards)} low_or_deferred={len(low_cards)}"
    )


if __name__ == "__main__":
    main()
