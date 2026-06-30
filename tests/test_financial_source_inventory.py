from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_financial_source_inventory import build_inventory, clean_date_text, csv_row_count, partition_date


def test_partition_date_and_clean_date_text() -> None:
    assert partition_date("trade_date_20240430.csv") == "2024-04-30"
    assert clean_date_text("20240430") == "2024-04-30"
    assert clean_date_text("") == ""


def test_inventory_detects_cached_financial_tables(tmp_path: Path) -> None:
    fina_dir = tmp_path / "tables" / "fina_indicator"
    anns_dir = tmp_path / "tables" / "anns_d"
    fina_dir.mkdir(parents=True)
    anns_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240430", "end_date": "20231231", "roe": "9.1"},
            {"ts_code": "000002.SZ", "f_ann_date": "20260131", "end_date": "20251231", "roe": "5.0"},
        ]
    ).to_csv(fina_dir / "sample.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240430", "title": "2023年年度报告", "url": "u"},
            {"ts_code": "000003.SZ", "ann_date": "20260131", "title": "业绩预告", "url": "u"},
        ]
    ).to_csv(anns_dir / "20240430_20260131.csv", index=False, encoding="utf-8-sig")

    tables, coverage, gap_plan = build_inventory(tmp_path)

    assert int(tables.loc[tables["table"].eq("fina_indicator"), "rows"].iloc[0]) == 2
    assert int(tables.loc[tables["table"].eq("anns_d"), "rows"].iloc[0]) == 2
    assert coverage.loc[coverage["table"].eq("anns_d"), "financial_title_rows"].sum() == 2
    assert set(gap_plan["table"]).issuperset({"fina_indicator", "forecast", "express", "anns_d"})


def test_csv_row_count_handles_header_only(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n", encoding="utf-8")
    assert csv_row_count(path) == 0
