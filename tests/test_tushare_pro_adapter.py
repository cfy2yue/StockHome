from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.tushare_pro_adapter import (
    DEFAULT_REQUEST_INTERVAL_SECONDS,
    TushareCacheConfig,
    TushareProAdapter,
    read_token,
    write_cache_manifest,
    write_coverage_outputs,
)


class FakePro:
    def stock_basic(self, **kwargs):  # noqa: ANN003
        assert kwargs["list_status"] == "L"
        return pd.DataFrame([{"ts_code": "000001.SZ", "name": "平安银行"}])

    def daily(self, **kwargs):  # noqa: ANN003
        return pd.DataFrame([{"ts_code": kwargs.get("ts_code", "000001.SZ"), "trade_date": kwargs.get("trade_date", "20260105")}])


class FailingPro:
    def daily(self, **kwargs):  # noqa: ANN003
        raise RuntimeError("simulated interface failure")


class QueryOnlyPro:
    def query(self, interface, **kwargs):  # noqa: ANN001, ANN003
        return pd.DataFrame([{"interface": interface, "src": kwargs.get("src", "")}])


class EmptyPro:
    def anns_d(self, **kwargs):  # noqa: ANN003
        return pd.DataFrame()


class CappedAnnouncementPro:
    def anns_d(self, **kwargs):  # noqa: ANN003
        return pd.DataFrame({"ts_code": ["000001.SZ"] * 6000, "ann_date": ["20250430"] * 6000})


def test_read_token_prefers_environment_without_logging(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token-value")
    missing_file = tmp_path / "missing_token.txt"
    assert read_token(missing_file) == "test-token-value"


def test_tushare_adapter_writes_cache_and_manifest_without_secret(tmp_path: Path):
    sleeps: list[float] = []
    config = TushareCacheConfig(cache_dir=tmp_path / "cache", token_path=tmp_path / "token.txt", request_interval_seconds=0.01)
    adapter = TushareProAdapter(config, pro=FakePro(), sleeper=sleeps.append)

    frame = adapter.call("stock_basic", exchange="", list_status="L")
    output = adapter.write_table("stock_basic", frame, partition="list_status_L")
    manifest_path = write_cache_manifest(config.cache_dir, records=adapter.records, dry_run=False, notes=["unit test"])
    _, report_path = write_coverage_outputs(config.cache_dir, tmp_path / "reports", adapter.records, dry_run=False)

    assert output.exists()
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "paid_standardized" in manifest_text
    assert "token/key: 未写入" in report_text
    assert "test-token-value" not in manifest_text
    assert "test-token-value" not in report_text


def test_request_interval_is_clamped_to_project_minimum(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache", request_interval_seconds=0.01)
    assert config.request_interval_seconds == DEFAULT_REQUEST_INTERVAL_SECONDS


def test_request_timeout_is_clamped_to_safe_minimum(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache", request_timeout_seconds=1)
    assert config.request_timeout_seconds == 5.0


def test_manifest_records_table_files_without_credentials(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=FakePro(), sleeper=lambda _: None)
    adapter.write_table("stock_basic", adapter.call("stock_basic", exchange="", list_status="L"), partition="list_status_L")
    manifest_path = write_cache_manifest(config.cache_dir, records=adapter.records, dry_run=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_type"] == "paid_standardized"
    assert manifest["table_files"] == ["tables/stock_basic/list_status_L.csv"]
    assert "test-token-value" not in json.dumps(manifest, ensure_ascii=False)


def test_coverage_dedupes_rows_by_output_path(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=FakePro(), sleeper=lambda _: None)
    for _ in range(2):
        adapter.write_table("daily", adapter.call("daily", trade_date="20260105"), partition="trade_date_20260105")

    _, report_path = write_coverage_outputs(config.cache_dir, tmp_path / "reports", adapter.records, dry_run=False)
    report = report_path.read_text(encoding="utf-8")
    assert "| daily" in report
    assert "          2 " in report or "          2 |" in report
    assert "      1 " in report or "      1 |" in report


def test_empty_success_is_visible_and_writes_readable_csv(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=EmptyPro(), sleeper=lambda _: None)
    frame = adapter.call("anns_d", start_date="20260106", end_date="20260107")
    output = adapter.write_table("anns_d", frame, partition="20260106_20260107")
    _, report_path = write_coverage_outputs(config.cache_dir, tmp_path / "reports", adapter.records, dry_run=False)

    assert adapter.records[0].status == "ok_empty"
    assert pd.read_csv(output).empty
    report = report_path.read_text(encoding="utf-8")
    assert "ok_empty" not in report
    assert "| anns_d" in report
    assert "empty" in report
    assert "empty_ok_requests" in report


def test_announcement_row_cap_risk_is_reported(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=CappedAnnouncementPro(), sleeper=lambda _: None)
    frame = adapter.call("anns_d", start_date="20250430", end_date="20250430")
    adapter.write_table("anns_d", frame, partition="20250430_20250430")
    _, report_path = write_coverage_outputs(config.cache_dir, tmp_path / "reports", adapter.records, dry_run=False)

    report = report_path.read_text(encoding="utf-8")
    assert "possible_row_cap_requests" in report
    assert "6000" in report


def test_adapter_records_failed_call_without_secret(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=FailingPro(), sleeper=lambda _: None)
    try:
        adapter.call("daily", trade_date="20260105", token="hidden")
    except RuntimeError:
        pass

    assert adapter.records[0].status == "failed"
    payload = adapter.records[0].as_dict()
    assert payload["params"]["token"] == "<redacted>"
    assert "hidden" not in json.dumps(payload, ensure_ascii=False)


def test_adapter_falls_back_to_generic_query_for_optional_interfaces(tmp_path: Path):
    config = TushareCacheConfig(cache_dir=tmp_path / "cache")
    adapter = TushareProAdapter(config, pro=QueryOnlyPro(), sleeper=lambda _: None)
    frame = adapter.call("major_news", src="新浪财经")

    assert frame.iloc[0]["interface"] == "major_news"
    assert adapter.records[0].interface == "major_news"
    assert adapter.records[0].status == "ok"
