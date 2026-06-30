from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUNDED_CARDS_PATH = ROOT / "book_skills" / "grounded_skill_cards.yaml"

CARD_FIELDS_FOR_EVIDENCE = [
    "strategy_id",
    "source_book",
    "chapter",
    "page_range",
    "extraction_method",
    "confidence",
    "source_status",
    "validation_status",
    "applicable_condition",
    "failure_condition",
    "user_output_boundary",
]


def split_triggered_skills(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw]
    else:
        values = [item.strip() for item in re.split(r"[;,，、\s]+", str(raw))]
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value.lower() in {"nan", "none", "null"}:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_book_skill_candidates(
    triggered_skills: Any,
    *,
    grounded_cards_path: str | Path = DEFAULT_GROUNDED_CARDS_PATH,
    max_cards: int = 4,
) -> list[dict[str, Any]]:
    ids = split_triggered_skills(triggered_skills)
    if not ids:
        return []
    cards = _load_grounded_cards(str(Path(grounded_cards_path).resolve()))
    resolved: list[dict[str, Any]] = []
    for strategy_id in ids[:max_cards]:
        card = cards.get(strategy_id)
        if not card:
            resolved.append(
                {
                    "strategy_id": strategy_id,
                    "source_status": "missing_grounded_card",
                    "confidence": "low",
                    "validation_status": "weak_until_grounded",
                    "applicable_condition": "未找到 grounded card；只能作为弱线索，不能作为强证据。",
                    "failure_condition": "缺少书名、章节、页码或失效条件时，DeepSeek 不得据此单独触发买入/加仓或升级辅助分级。",
                    "user_output_boundary": "只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。",
                }
            )
            continue
        resolved.append(_evidence_card(card))
    if len(ids) > max_cards:
        resolved.append(
            {
                "strategy_id": "__truncated__",
                "source_status": "truncated_for_prompt_budget",
                "confidence": "low",
                "validation_status": "not_evaluated",
                "applicable_condition": f"本决策触发 {len(ids)} 条 Book Skill，仅展示前 {max_cards} 条；其余不得作为强证据。",
                "failure_condition": "若关键策略被截断，必须降低置信度或要求人工复核。",
                "user_output_boundary": "只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。",
            }
        )
    return resolved


@lru_cache(maxsize=8)
def _load_grounded_cards(path_text: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        return {}
    cards: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        strategy_id = str(item.get("strategy_id") or "").strip()
        if strategy_id:
            cards[strategy_id] = item
    return cards


def _evidence_card(card: dict[str, Any]) -> dict[str, Any]:
    out = {field: _clean_text(card.get(field)) for field in CARD_FIELDS_FOR_EVIDENCE if _clean_text(card.get(field))}
    out["evidence_policy"] = "source_and_conditions_only_no_per_decision_future_results"
    return out


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= 260:
        return text
    return text[:257].rstrip() + "..."
