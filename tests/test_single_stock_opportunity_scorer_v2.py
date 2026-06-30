from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_single_stock_opportunity_scorer_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("audit_single_stock_opportunity_scorer_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_high_complexity_logistic_is_diagnostic_not_green() -> None:
    module = _module()

    status = module._status(
        h_dpos=0.14,
        h_dmean=3.2,
        prior_hit=0.75,
        prior_mean_hit=0.75,
        active=0.12,
        top_share=0.02,
        model_family="logistic",
        feature_count=159,
    )

    assert status == "yellow_diagnostic_high_complexity"


def test_low_complexity_additive_can_be_green() -> None:
    module = _module()

    status = module._status(
        h_dpos=0.05,
        h_dmean=1.4,
        prior_hit=1.0,
        prior_mean_hit=1.0,
        active=0.20,
        top_share=0.01,
        model_family="additive_bin",
        feature_count=16,
    )

    assert status == "green_candidate"


def test_clean_feature_list_excludes_future_and_result_fields() -> None:
    module = _module()
    rows = 120
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="D").astype(str),
            "code": ["000001"] * rows,
            "safe_feature": list(range(rows)),
            "return_20d": [1.0] * rows,
            "single_stock_label": ["increase_research"] * rows,
        }
    )

    clean = module._clean_feature_list(["safe_feature", "return_20d", "single_stock_label"], frame)

    assert clean == ["safe_feature"]


def test_agent_preview_rejects_future_columns_and_instruction_terms() -> None:
    module = _module()
    with pytest.raises(ValueError, match="future/result fields"):
        module._reject_agent_preview_leak(pd.DataFrame([{"date": "2026-01-01", "code": "000001", "return_20d": 1.0}]))

    with pytest.raises(ValueError, match="disallowed instruction"):
        module._reject_agent_preview_leak(pd.DataFrame([{"date": "2026-01-01", "code": "000001", "note": "买入"}]))
