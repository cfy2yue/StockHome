from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.build_historical_announcement_smoke import _parse_dates, _summary_row, _write_plan


def test_parse_dates_dedupes_and_validates() -> None:
    assert _parse_dates("20240430,20240430,20241031") == ["20240430", "20241031"]
    try:
        _parse_dates("2024-04-30")
    except ValueError as exc:
        assert "invalid YYYYMMDD" in str(exc)
    else:
        raise AssertionError("expected invalid date")


def test_summary_row_flags_possible_row_cap(tmp_path: Path) -> None:
    row = _summary_row("20240430", "ok", 6000, tmp_path / "x.csv")
    assert row["possible_row_cap"] is True
    assert row["date"] == "20240430"


def test_write_plan_omits_credentials(tmp_path: Path) -> None:
    path = tmp_path / "plan.md"
    args = SimpleNamespace(execute=False, max_dates=2, request_interval_seconds=0.7, force=False)
    _write_plan(path, ["20240430"], args)
    text = path.read_text(encoding="utf-8")
    assert "token" in text.lower()
    assert "sk-" not in text
    assert "20240430" in text
