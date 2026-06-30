from __future__ import annotations

from pathlib import Path

from src.agent_training.memory_context import load_compact_memory_context


def test_load_compact_memory_context_reads_recent_structured_rows(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "strategy_experience_ledger.csv").write_text(
        "experience_id,rule_or_observation,metric_after,accepted_or_rejected,failure_condition,next_action\n"
        "EXP-1,old rule,old metric,observe,old failure,old action\n"
        "EXP-2,new rule,new metric,accepted,new failure,new action\n",
        encoding="utf-8",
    )

    text = load_compact_memory_context(
        tmp_path,
        ledger_files=["memory/strategy_experience_ledger.csv"],
        rows_per_file=1,
        max_chars=1000,
    )

    assert "EXP-2" in text
    assert "new rule" in text
    assert "new metric" not in text
    assert "metric_after" not in text
    assert "EXP-1" not in text
    assert "## memory/strategy_experience_ledger.csv" in text


def test_load_compact_memory_context_filters_forbidden_fallback_columns(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "custom.csv").write_text(
        "experience_id,metric_before,metric_after,gt_status,rule_or_observation,next_action\n"
        "EXP-3,before,after,pass,test rule,test action\n",
        encoding="utf-8",
    )

    text = load_compact_memory_context(
        tmp_path,
        ledger_files=["memory/custom.csv"],
        rows_per_file=1,
        max_chars=1000,
    )

    assert "EXP-3" in text
    assert "test rule" in text
    assert "test action" in text
    assert "metric_before" not in text
    assert "metric_after" not in text
    assert "gt_status" not in text
    assert "before" not in text
    assert "after" not in text
    assert "pass" not in text


def test_load_compact_memory_context_redacts_hindsight_tokens_inside_text(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "failure_case_ledger.csv").write_text(
        "failure_id,failure_pattern,countermeasure,status\n"
        "FAIL-1,python_only exposure_avg_return_20d was bad; keep rule status,review,open\n",
        encoding="utf-8",
    )

    text = load_compact_memory_context(
        tmp_path,
        ledger_files=["memory/failure_case_ledger.csv"],
        rows_per_file=1,
        max_chars=1000,
    )

    assert "return_20d" not in text
    assert "exposure_avg" not in text
    assert "[redacted_hindsight_metric]" in text
    assert "keep rule status" in text


def test_load_compact_memory_context_respects_max_chars(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    long_text = "x" * 500
    (memory / "failure_case_ledger.csv").write_text(
        "failure_id,failure_pattern,countermeasure,status\n"
        f"FAIL-1,{long_text},{long_text},open\n",
        encoding="utf-8",
    )

    text = load_compact_memory_context(
        tmp_path,
        ledger_files=["memory/failure_case_ledger.csv"],
        rows_per_file=1,
        max_chars=160,
        row_char_limit=120,
    )

    assert len(text) <= 160
    assert "FAIL-1" in text
