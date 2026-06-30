from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent_training.decision_card import validate_decision_card
from src.agent_training.deepseek_client import DEFAULT_MODEL, get_api_key, load_project_dotenv, mask_key
from src.agent_training.policy_runner import metrics_from_daily_returns


def _valid_card() -> dict[str, object]:
    return {
        "type": "agent_decision_card",
        "agent_policy_version": "test_policy",
        "variant": "full_agent",
        "step": 1,
        "train_blocks": "H2023_1",
        "valid_block": "H2023_2",
        "decision_date": "2023-07-04",
        "code": "000001",
        "name": "测试股票",
        "task_mode": "portfolio_pool",
        "research_grade": "继续深挖",
        "simulated_action": "增加研究暴露",
        "simulated_weight_change": 1.0,
        "python_signal_summary": "python score ok",
        "news_signal_summary": "news neutral",
        "book_skill_evidence": "",
        "memory_experience_used": "none",
        "counter_evidence": "无强反证",
        "accepted_quant_tool_ids": "none",
        "quant_tool_adoption_decision": "not_applicable",
        "quant_tool_override_reasons": "none",
        "final_agent_reasoning_summary": "schema test",
        "confidence_level": 0.72,
        "data_missing_flags": "",
        "error_reflection": "not evaluated",
        "research_only": True,
        "not_investment_instruction": True,
    }


def test_mask_key_does_not_expose_full_secret() -> None:
    key = "sk-" + "1234567890abcdef"
    masked = mask_key(key)
    assert masked.startswith("sk-")
    assert masked.endswith("cdef")
    assert "1234567890ab" not in masked


def test_load_project_dotenv_reads_untracked_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-local-test\n", encoding="utf-8")
    loaded = load_project_dotenv(env_file)
    assert loaded == env_file
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-local-test"


def test_get_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("src.agent_training.deepseek_client.load_project_dotenv", lambda path=None: None)
    monkeypatch.setattr("src.agent_training.deepseek_client._get_windows_user_env", lambda env_name: "")
    monkeypatch.setattr("src.agent_training.deepseek_client._get_local_key_file", lambda: "")
    with pytest.raises(RuntimeError, match="missing DEEPSEEK_API_KEY"):
        get_api_key()


def test_get_api_key_reads_untracked_local_key_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / "ds_api.txt").write_text("sk-local-file-test\n", encoding="utf-8")
    monkeypatch.setattr("src.agent_training.deepseek_client.load_project_dotenv", lambda path=None: None)
    monkeypatch.setattr("src.agent_training.deepseek_client._get_windows_user_env", lambda env_name: "")
    monkeypatch.setattr("src.agent_training.deepseek_client.project_root", lambda: tmp_path)
    assert get_api_key() == "sk-local-file-test"


def test_decision_card_accepts_project_research_grades() -> None:
    card = _valid_card()
    assert validate_decision_card(card) is card


def test_decision_card_rejects_investment_instruction() -> None:
    card = _valid_card()
    card["research_grade"] = "买入"
    with pytest.raises(ValueError, match="invalid research_grade"):
        validate_decision_card(card)


def test_decision_card_sanitizes_strong_promise_terms_only() -> None:
    card = _valid_card()
    card["book_skill_evidence"] = "未提供明确买入信号。"
    card["final_agent_reasoning_summary"] = "不宜加仓，避免输出强烈推荐。"
    card["error_reflection"] = "后验发现卖出措辞不适合用户端。"
    validated = validate_decision_card(card)
    text = "\n".join(
        [
            str(validated["book_skill_evidence"]),
            str(validated["final_agent_reasoning_summary"]),
            str(validated["error_reflection"]),
        ]
    )
    for term in ["强烈推荐", "目标价必达"]:
        assert term not in text
    assert "买入" in text
    assert "加仓" in text
    assert "卖出" in text
    assert "研究优先级较高" in text


def test_policy_metrics_are_stable_for_cash_and_exposure() -> None:
    metrics = metrics_from_daily_returns([1.0, -2.0, 3.0], exposure_dates=2)
    assert metrics["decision_dates"] == 3
    assert metrics["exposure_decision_dates"] == 2
    assert metrics["cash_decision_dates"] == 1
    assert metrics["positive_20d_rate"] == 0.6667


def test_default_decision_model_is_deepseek_pro() -> None:
    assert DEFAULT_MODEL == "deepseek-v4-pro"
