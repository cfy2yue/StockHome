from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.build_tushare_cache import _date_shards, _execute_interfaces, _write_fetch_plan_manifest


def test_date_shards_are_inclusive_and_capped_by_end_date() -> None:
    assert _date_shards("20260101", "20260110", 3) == [
        ("20260101", "20260103"),
        ("20260104", "20260106"),
        ("20260107", "20260109"),
        ("20260110", "20260110"),
    ]


def test_date_shards_single_request_when_disabled() -> None:
    assert _date_shards("20260101", "20260110", 0) == [("20260101", "20260110")]


def test_date_shards_reject_reverse_range() -> None:
    with pytest.raises(ValueError, match="end_date"):
        _date_shards("20260110", "20260101", 3)


class FakeAdapter:
    def __init__(self, cache_dir: Path) -> None:
        self.config = SimpleNamespace(cache_dir=cache_dir)
        self.calls: list[tuple[str, dict]] = []

    def call(self, interface: str, **params):  # noqa: ANN001, ANN003
        self.calls.append((interface, params))
        return pd.DataFrame([{"ts_code": params.get("ts_code", "000001.SZ"), "ann_date": "20260430"}])

    def write_table(self, table_name: str, frame: pd.DataFrame, *, partition: str | None = None) -> Path:
        path = self.config.cache_dir / "tables" / table_name / f"{partition or table_name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path


def _write_stock_basic(cache_dir: Path) -> None:
    stock_dir = cache_dir / "tables" / "stock_basic"
    stock_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "A"},
            {"ts_code": "000002.SZ", "name": "B"},
            {"ts_code": "000003.SZ", "name": "C"},
        ]
    ).to_csv(stock_dir / "list_status_L.csv", index=False)


def test_financial_stock_interfaces_require_max_stocks(tmp_path: Path) -> None:
    _write_stock_basic(tmp_path)
    args = SimpleNamespace(
        interfaces=["forecast"],
        start_date="20260101",
        end_date="20260630",
        max_stocks=0,
        stock_offset=0,
        skip_existing=False,
        max_trade_dates=0,
        ann_shard_days=0,
        max_ann_shards=0,
        news_srcs="",
        major_news_srcs="",
        news_start_datetime="",
        news_end_datetime="",
    )

    with pytest.raises(RuntimeError, match="forecast requires --max-stocks"):
        _execute_interfaces(FakeAdapter(tmp_path), args)


def test_financial_stock_interfaces_are_bounded_by_max_stocks(tmp_path: Path) -> None:
    _write_stock_basic(tmp_path)
    adapter = FakeAdapter(tmp_path)
    args = SimpleNamespace(
        interfaces=["forecast", "express"],
        start_date="20260101",
        end_date="20260630",
        max_stocks=2,
        stock_offset=0,
        skip_existing=False,
        max_trade_dates=0,
        ann_shard_days=0,
        max_ann_shards=0,
        news_srcs="",
        major_news_srcs="",
        news_start_datetime="",
        news_end_datetime="",
    )

    _execute_interfaces(adapter, args)

    assert [name for name, _ in adapter.calls] == ["forecast", "forecast", "express", "express"]
    assert {params["ts_code"] for _, params in adapter.calls} == {"000001.SZ", "000002.SZ"}
    assert (tmp_path / "tables" / "forecast" / "000001.SZ.csv").exists()
    assert (tmp_path / "tables" / "express" / "000002.SZ.csv").exists()


def test_stock_interfaces_support_offset_and_skip_existing(tmp_path: Path) -> None:
    _write_stock_basic(tmp_path)
    existing = tmp_path / "tables" / "forecast" / "000002.SZ.csv"
    existing.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ts_code": "000002.SZ"}]).to_csv(existing, index=False)
    adapter = FakeAdapter(tmp_path)
    args = SimpleNamespace(
        interfaces=["forecast"],
        start_date="20260101",
        end_date="20260630",
        max_stocks=2,
        stock_offset=1,
        skip_existing=True,
        max_trade_dates=0,
        ann_shard_days=0,
        max_ann_shards=0,
        news_srcs="",
        major_news_srcs="",
        news_start_datetime="",
        news_end_datetime="",
    )

    _execute_interfaces(adapter, args)

    assert adapter.calls == [("forecast", {"ts_code": "000003.SZ", "start_date": "20260101", "end_date": "20260630"})]
    assert (tmp_path / "tables" / "forecast" / "000003.SZ.csv").exists()


def test_dry_run_fetch_plan_manifest_does_not_use_main_cache_manifest(tmp_path: Path) -> None:
    args = SimpleNamespace(
        interfaces=["forecast", "express"],
        start_date="20230101",
        end_date="20260630",
        max_stocks=5,
        stock_offset=10,
        skip_existing=True,
        max_trade_dates=0,
        ann_shard_days=0,
        max_ann_shards=0,
        request_timeout_seconds=20,
    )

    path = _write_fetch_plan_manifest(tmp_path, args, ["dry run"])

    assert path.name == "fetch_plan_manifest.json"
    assert path.exists()
    assert not (tmp_path / "cache_manifest.json").exists()
    assert "fetch_plan_only_no_api_call" in path.read_text(encoding="utf-8")
