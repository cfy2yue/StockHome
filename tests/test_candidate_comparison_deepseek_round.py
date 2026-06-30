from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run_candidate_comparison_deepseek_round.py"
    spec = importlib.util.spec_from_file_location("run_candidate_comparison_deepseek_round", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _group() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "comparison_group_id": "CMP1",
                "comparison_scenario": "same_sector",
                "repeat_seed": 0,
                "time_block": "H2026_1",
                "date": "2026-01-06",
                "candidate_count": 2,
                "candidate_codes": "000001;000002",
                "candidate_names": "A;B",
                "industry_context": "银行",
                "code": "000001",
                "name": "A",
                "tushare_industry": "银行",
                "tushare_area": "深圳",
                "p1_default_selector_v1": 0.9,
                "rev_chip_core": 0.9,
                "single_watch_proxy": 0.8,
                "news_warning_score": 0.1,
                "financial_quality_risk_score": 0.2,
                "corr_peer_relative_return_20d": 1.0,
                "triggered_skills": "SKILL-1",
            },
            {
                "comparison_group_id": "CMP1",
                "comparison_scenario": "same_sector",
                "repeat_seed": 0,
                "time_block": "H2026_1",
                "date": "2026-01-06",
                "candidate_count": 2,
                "candidate_codes": "000001;000002",
                "candidate_names": "A;B",
                "industry_context": "银行",
                "code": "000002",
                "name": "B",
                "tushare_industry": "银行",
                "tushare_area": "深圳",
                "p1_default_selector_v1": 0.1,
                "rev_chip_core": 0.1,
                "single_watch_proxy": 0.1,
                "news_warning_score": 0.8,
                "financial_quality_risk_score": 0.7,
                "corr_peer_relative_return_20d": -1.0,
                "triggered_skills": "SKILL-2",
            },
        ]
    )


def test_candidate_pack_ablation_hides_requested_channels() -> None:
    module = _load_script()

    anchor = module.build_candidate_pack(_group(), variant="ranker_anchor_agent", agent_policy_version="test")
    no_news = module.build_candidate_pack(_group(), variant="no_news", agent_policy_version="test")
    no_quant = module.build_candidate_pack(_group(), variant="no_quant", agent_policy_version="test")
    no_book = module.build_candidate_pack(_group(), variant="no_bookskill", agent_policy_version="test")

    assert anchor["default_ranked_candidates"][0]["code"] == "000001"
    assert "anchor_policy" in anchor
    assert no_news["candidates"][0]["features"]["news_warning_score"] == "hidden_by_ablation"
    assert no_quant["candidates"][0]["scores"] == {"hidden_by_ablation": True}
    assert no_book["candidates"][0]["features"]["triggered_skills"] == "hidden_by_ablation"


def test_ranker_anchor_prompt_requires_actionable_top2_when_no_hard_counter() -> None:
    module = _load_script()
    pack = module.build_candidate_pack(_group(), variant="ranker_anchor_agent", agent_policy_version="test")
    messages = module.build_candidate_messages(pack)
    joined = "\n".join(message["content"] for message in messages)

    assert "默认Top1/Top2若无硬反证" in joined
    assert "条件化小仓" in joined
    assert "新闻预警>=0.6" in joined


def test_validate_candidate_card_rejects_prohibited_terms_and_unknown_codes() -> None:
    module = _load_script()
    pack = module.build_candidate_pack(_group(), variant="full_agent", agent_policy_version="test")
    parsed = {
        "top_research_codes": ["000001"],
        "ranked_candidates": [
            {
                "rank": 1,
                "code": "000001",
                "name": "A",
                "research_grade": "继续深挖",
                "priority_reason": "证据较完整",
                "counter_evidence": "none",
                "data_missing_flags": "none",
            }
        ],
        "comparison_summary": "只作研究优先级",
        "confidence_level": 0.7,
        "data_missing_summary": "none",
        "research_only": True,
        "not_investment_instruction": True,
    }
    card = module.validate_candidate_card(pack, parsed)
    assert card["top_research_codes"] == ["000001"]
    assert card["task_mode"] == "candidate_comparison"
    assert "rank_override_audit" in card

    with_buy = dict(parsed)
    with_buy["comparison_summary"] = "建议买入，但需要满足阈值"
    assert module.validate_candidate_card(pack, with_buy)["comparison_summary"]

    bad = dict(parsed)
    bad["comparison_summary"] = "目标价必达"
    try:
        module.validate_candidate_card(pack, bad)
    except ValueError as exc:
        assert "prohibited" in str(exc)
    else:
        raise AssertionError("expected prohibited strong-claim rejection")

    bad_code = dict(parsed)
    bad_code["ranked_candidates"] = [dict(parsed["ranked_candidates"][0], code="999999")]
    try:
        module.validate_candidate_card(pack, bad_code)
    except ValueError as exc:
        assert "not in pack" in str(exc)
    else:
        raise AssertionError("expected unknown code rejection")


def test_panel_index_selects_disjoint_bucket_groups() -> None:
    module = _load_script()
    rows = []
    for idx in range(6):
        rows.append(
            {
                "comparison_group_id": f"G{idx}",
                "comparison_scenario": "same_sector",
                "time_block": "H2026_1",
                "repeat_seed": idx,
                "date": f"2026-01-{idx + 1:02d}",
            }
        )
    plan = pd.DataFrame(rows)

    panel0 = module.select_group_ids(plan, max_groups=0, panel_index=0, groups_per_bucket=2)
    panel1 = module.select_group_ids(plan, max_groups=0, panel_index=1, groups_per_bucket=2)

    assert panel0 == ["G0", "G1"]
    assert panel1 == ["G2", "G3"]
    assert not set(panel0) & set(panel1)


def test_cross_sector_anchor_uses_rank_avg_rev_watch() -> None:
    module = _load_script()
    group = _group().copy()
    group["comparison_scenario"] = "cross_sector"
    group.loc[group["code"].eq("000001"), "rev_chip_core"] = 10.0
    group.loc[group["code"].eq("000001"), "single_watch_proxy"] = 0.0
    group.loc[group["code"].eq("000002"), "rev_chip_core"] = 0.0
    group.loc[group["code"].eq("000002"), "single_watch_proxy"] = 10.0
    pack = module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")

    assert "rank_avg_rev_watch" in pack["candidate_score_policy"]
    for item in pack["default_ranked_candidates"]:
        assert "rank_avg_rev_watch" in item
    assert {item["code"] for item in pack["default_ranked_candidates"][:2]} == {"000001", "000002"}


def test_same_sector_score_override_can_use_rank_avg_rev_watch() -> None:
    module = _load_script()
    group = pd.concat([_group(), _group().iloc[[1]].copy()], ignore_index=True)
    group.loc[2, "code"] = "000003"
    group.loc[2, "name"] = "C"
    group["candidate_count"] = 3
    group["candidate_codes"] = "000001;000002;000003"
    group["candidate_names"] = "A;B;C"
    group = group.drop(columns=["p1_default_selector_v1"], errors="ignore")
    group.loc[group["code"].eq("000001"), "rev_chip_core"] = 10.0
    group.loc[group["code"].eq("000001"), "single_watch_proxy"] = 0.0
    group.loc[group["code"].eq("000002"), "rev_chip_core"] = 0.0
    group.loc[group["code"].eq("000002"), "single_watch_proxy"] = 10.0
    group.loc[group["code"].eq("000003"), "rev_chip_core"] = 0.0
    group.loc[group["code"].eq("000003"), "single_watch_proxy"] = 9.0

    default_pack = module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")
    override_pack = module.build_candidate_pack(
        group,
        variant="ranker_anchor_agent",
        agent_policy_version="test",
        same_sector_score="rank_avg_rev_watch",
    )

    assert default_pack["default_ranked_candidates"][0]["code"] == "000001"
    assert override_pack["default_ranked_candidates"][0]["code"] == "000002"
    assert "same_sector使用rank_avg_rev_watch" in override_pack["candidate_score_policy"]


def test_reused_comparison_group_id_is_rejected() -> None:
    module = _load_script()
    rows = _group()
    duplicated = pd.concat([rows, rows.copy()], ignore_index=True)
    duplicated.loc[2:, "date"] = "2026-01-07"

    try:
        module.validate_unique_comparison_groups(duplicated, source="test_rows")
    except ValueError as exc:
        assert "reused comparison_group_id" in str(exc)
    else:
        raise AssertionError("expected reused comparison_group_id rejection")


def test_duplicate_candidate_code_inside_group_is_rejected() -> None:
    module = _load_script()
    group = _group().copy()
    group.loc[1, "code"] = "000001"

    try:
        module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")
    except ValueError as exc:
        assert "duplicate codes" in str(exc)
    else:
        raise AssertionError("expected duplicate candidate code rejection")


def test_ranker_anchor_actionability_postcheck_rewrites_zero_top_pick_without_hard_counter() -> None:
    module = _load_script()
    group = _group()
    pack = module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")
    parsed = {
        "top_research_codes": ["000001", "000002"],
        "ranked_candidates": [
            {
                "rank": 1,
                "code": "000001",
                "name": "A",
                "operation_recommendation": "等待不买",
                "position_threshold": "新仓0%",
                "buy_or_add_trigger": "等待",
                "reduce_or_sell_trigger": "风险扩散",
                "research_grade": "继续深挖",
                "priority_reason": "锚点第一",
                "counter_evidence": "none",
                "data_missing_flags": "none",
            },
            {
                "rank": 2,
                "code": "000002",
                "name": "B",
                "operation_recommendation": "等待不买",
                "position_threshold": "新仓0%",
                "buy_or_add_trigger": "等待",
                "reduce_or_sell_trigger": "风险扩散",
                "research_grade": "继续深挖",
                "priority_reason": "锚点第二但有硬风险",
                "counter_evidence": "news risk",
                "data_missing_flags": "none",
            },
        ],
        "comparison_summary": "候选对比",
        "confidence_level": 0.5,
        "data_missing_summary": "none",
        "research_only": True,
        "not_investment_instruction": True,
    }

    card = module.validate_candidate_card(pack, parsed)

    assert card["ranked_candidates"][0]["position_threshold"].startswith("新仓10%-20%试探")
    assert card["ranked_candidates"][1]["position_threshold"] == "新仓0%"
    assert card["actionability_postcheck_audit"][0]["code"] == "000001"


def test_ranker_anchor_actionability_postcheck_does_not_touch_hard_counter() -> None:
    module = _load_script()
    group = _group()
    pack = module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")
    parsed = {
        "top_research_codes": ["000002"],
        "ranked_candidates": [
            {
                "rank": 1,
                "code": "000002",
                "name": "B",
                "operation_recommendation": "等待不买",
                "position_threshold": "新仓0%",
                "buy_or_add_trigger": "等待",
                "reduce_or_sell_trigger": "风险扩散",
                "research_grade": "继续深挖",
                "priority_reason": "有硬风险",
                "counter_evidence": "news risk",
                "data_missing_flags": "none",
            },
        ],
        "comparison_summary": "候选对比",
        "confidence_level": 0.5,
        "data_missing_summary": "none",
        "research_only": True,
        "not_investment_instruction": True,
    }

    card = module.validate_candidate_card(pack, parsed)

    assert card["ranked_candidates"][0]["position_threshold"] == "新仓0%"
    assert card["actionability_postcheck_audit"] == "none"


def test_candidate_card_replaces_not_applicable_triggers() -> None:
    module = _load_script()
    group = _group()
    pack = module.build_candidate_pack(group, variant="ranker_anchor_agent", agent_policy_version="test")
    parsed = {
        "top_research_codes": ["000001"],
        "ranked_candidates": [
            {
                "rank": 3,
                "code": "000001",
                "name": "A",
                "operation_recommendation": "持有",
                "position_threshold": "新仓0%",
                "buy_or_add_trigger": "不适用",
                "reduce_or_sell_trigger": "不适用",
                "research_grade": "放入观察",
                "priority_reason": "下位候选",
                "counter_evidence": "同行弱",
                "data_missing_flags": "none",
            },
        ],
        "comparison_summary": "候选对比",
        "confidence_level": 0.5,
        "data_missing_summary": "none",
        "research_only": True,
        "not_investment_instruction": True,
    }

    card = module.validate_candidate_card(pack, parsed)
    item = card["ranked_candidates"][0]

    assert item["buy_or_add_trigger"] != "不适用"
    assert item["reduce_or_sell_trigger"] != "不适用"
    assert "Top2" in item["buy_or_add_trigger"]
