from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_CONFIDENCE = {"high", "medium"}
REQUIRED_SOURCE_FIELDS = ("book", "chapter", "page_range", "extraction_method", "confidence")

TASK_ALIASES = {
    "paired_stock_comparison": ["paired_stock_comparison", "multi_stock_comparison"],
}


def is_formal_strategy_card(card: dict) -> bool:
    source = card.get("source", {})
    if source.get("confidence") not in ALLOWED_CONFIDENCE:
        return False
    formal_status = str(card.get("formal_status", ""))
    if not formal_status.startswith("是"):
        return False
    if not all(source.get(field) for field in REQUIRED_SOURCE_FIELDS):
        return False
    return bool(card.get("strategy_id") and card.get("principle"))


def load_strategy_cards(include_deferred: bool = False) -> list[dict]:
    path = ROOT / "book_skills" / "strategy_cards.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if include_deferred:
        return data
    return [card for card in data if is_formal_strategy_card(card)]


def match_strategies(task: str) -> dict:
    task_names = TASK_ALIASES.get(task, [task])
    cards = [c for c in load_strategy_cards() if any(t in c.get("task_fit", []) for t in task_names) or task == "full"]
    evidence = []
    for c in cards:
        s = c["source"]
        evidence.append(f"来源：《{s['book']}》 / {s['chapter']} / {c['strategy_id']} / {s['page_range']}")
    score = min(10, 3 + len(cards))
    return {"score": score, "cards": cards, "evidence": evidence or ["当前任务未匹配到正式书籍策略来源"]}
