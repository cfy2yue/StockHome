from pathlib import Path

import yaml

from src.analysis.strategy_matcher import is_formal_strategy_card, load_strategy_cards


def test_strategy_cards_have_sources():
    cards = yaml.safe_load(Path("book_skills/strategy_cards.yaml").read_text(encoding="utf-8"))
    assert cards
    ids = [card["strategy_id"] for card in cards]
    assert len(ids) == len(set(ids))
    for card in cards:
        source = card["source"]
        assert card["strategy_id"]
        assert card["principle"]
        assert source["book"]
        assert source["chapter"]
        assert source["page_range"]
        assert source["extraction_method"]
        assert source["confidence"] in {"high", "medium"}
        assert source["raw_source"]
        assert "OCR_PAGE" in source["page_range"] or "页码" in source["page_range"]
        assert str(card.get("formal_status", "")).startswith("是")


def test_load_strategy_cards_filters_non_formal_cards():
    cards = load_strategy_cards()
    assert cards
    for card in cards:
        assert card["source"]["confidence"] in {"high", "medium"}
        assert str(card.get("formal_status", "")).startswith("是")


def test_formal_strategy_filter_rejects_non_formal_variants():
    base = {
        "strategy_id": "TEST",
        "principle": "测试",
        "formal_status": "是",
        "source": {
            "book": "测试书",
            "chapter": "第一章",
            "page_range": "OCR_PAGE 1",
            "raw_source": "测试来源",
            "extraction_method": "full_ocr_txt_deep_dive",
            "confidence": "high",
        },
    }
    assert is_formal_strategy_card(base)
    for status in ["候选", "部分：quant", "可合并", "暂缓", "否"]:
        card = {**base, "formal_status": status}
        assert not is_formal_strategy_card(card)
    missing_status = dict(base)
    missing_status.pop("formal_status")
    assert not is_formal_strategy_card(missing_status)
    low = {**base, "source": {**base["source"], "confidence": "low"}}
    assert not is_formal_strategy_card(low)
    missing_source = {**base, "source": {**base["source"], "page_range": ""}}
    assert not is_formal_strategy_card(missing_source)
