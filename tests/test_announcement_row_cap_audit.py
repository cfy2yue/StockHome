from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.audit_announcement_row_cap_coverage import (
    build_row_cap_audit,
    build_row_cap_detail,
    render_report,
)


def _write_anns(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_row_cap_audit_flags_capped_partition_and_detail(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _write_anns(
        cache_dir / "tables" / "anns_d" / "20240430_20240430.csv",
        [
            {"ts_code": "000001.SZ", "name": "A", "title": "2024年一季度报告"},
            {"ts_code": "000001.SZ", "name": "A", "title": "关于收到关注函的回复公告"},
            {"ts_code": "600000.SH", "name": "B", "title": "关于召开董事会的公告"},
        ],
    )
    _write_anns(
        cache_dir / "tables" / "anns_d" / "20240501_20240501.csv",
        [{"ts_code": "000002.SZ", "name": "C", "title": "投资者关系活动记录表"}],
    )

    audit = build_row_cap_audit(cache_dir, row_cap_threshold=3)
    detail = build_row_cap_detail(cache_dir, audit)

    capped = audit[audit["partition"].eq("20240430_20240430")].iloc[0]
    assert bool(capped["possible_row_cap"]) is True
    assert capped["rows"] == 3
    assert capped["unique_stocks"] == 2
    assert capped["financial_report_rows"] == 1
    assert capped["audit_or_inquiry_rows"] == 1
    assert not detail.empty
    assert {"exchange_suffix", "code_prefix3", "title_bucket", "top_stock"} <= set(detail["detail_type"])


def test_row_cap_report_has_boundary_and_no_secret_like_text(tmp_path: Path) -> None:
    audit = pd.DataFrame(
        [
            {
                "partition": "20240430_20240430",
                "ann_date": "20240430",
                "rows": 3,
                "possible_row_cap": True,
                "unique_stocks": 2,
                "top_stock_share": 0.6667,
                "sh_count": 1,
                "sz_count": 2,
                "bj_count": 0,
                "financial_report_rows": 1,
                "forecast_or_express_rows": 0,
                "audit_or_inquiry_rows": 1,
                "risk_warning_rows": 0,
                "investor_relation_rows": 0,
                "routine_governance_rows": 1,
                "output_path": str(tmp_path / "x.csv"),
            }
        ]
    )
    detail = pd.DataFrame(
        [{"ann_date": "20240430", "detail_type": "title_bucket", "bucket": "financial_report", "rows": 1, "share": 0.3333, "example": "2024年一季度报告"}]
    )
    args = SimpleNamespace(execute_probe=False)

    text = render_report(audit, detail, pd.DataFrame(), args, tmp_path / "audit.csv", tmp_path / "detail.csv", tmp_path / "probe.csv")

    assert "不能视为全量公告覆盖" in text
    assert "不直接扩大 DeepSeek round" in text
    assert "sk-" not in text
    assert "tushare_token" not in text
