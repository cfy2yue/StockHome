from __future__ import annotations

from pathlib import Path

from src.agent_training.book_skill_resolver import resolve_book_skill_candidates, split_triggered_skills


def test_split_triggered_skills_deduplicates_common_separators() -> None:
    assert split_triggered_skills("A;B，A C、none") == ["A", "B", "C"]


def test_resolver_uses_grounded_fields_and_hides_performance_metrics(tmp_path: Path) -> None:
    cards = tmp_path / "cards.yaml"
    cards.write_text(
        """
- strategy_id: TEST-001
  source_book: 测试书
  chapter: 第一章
  page_range: OCR_PAGE 1-2
  extraction_method: full_ocr_txt_deep_dive
  confidence: high
  source_status: grounded
  validation_status: observe
  raw_positive_20d_rate: 0.99
  applicable_condition: 需要跨通道确认。
  failure_condition: 遇到强反证降权。
  user_output_boundary: 只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。
""",
        encoding="utf-8",
    )

    resolved = resolve_book_skill_candidates("TEST-001;MISSING-001", grounded_cards_path=cards)

    assert resolved[0]["source_book"] == "测试书"
    assert resolved[0]["page_range"] == "OCR_PAGE 1-2"
    assert resolved[0]["evidence_policy"] == "source_and_conditions_only_no_per_decision_future_results"
    assert "raw_positive_20d_rate" not in resolved[0]
    assert resolved[1]["strategy_id"] == "MISSING-001"
    assert resolved[1]["source_status"] == "missing_grounded_card"
