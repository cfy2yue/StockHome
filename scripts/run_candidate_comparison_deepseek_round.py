"""Run DeepSeek on P1 candidate-comparison packs.

The normal dual-mode runner emits one decision card per stock. P1 needs a
different shape: one pack contains the whole user candidate set, and the agent
must rank 1-2 research priorities while grading every candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_candidate_comparison_workflow_v1 import (  # noqa: E402
    P1_DEFAULT_SCORE,
    SAFE_AGENT_FEATURES,
    SCORE_COLUMNS,
    FUTURE_RESULT_COLUMNS,
    ensure_task_default_score,
    load_candidate_frame,
)
from src.agent_training.decision_card import ALLOWED_RESEARCH_GRADES  # noqa: E402
from src.agent_training.deepseek_client import (  # noqa: E402
    BACKTEST_TRAINING_MODEL,
    FINAL_INFERENCE_MODEL,
    chat_json,
    extract_json_content,
    model_concurrency_limit,
)
from src.agent_training.deepseek_runner import write_jsonl  # noqa: E402

OUTPUT = ROOT / "reports" / "date_generalization"
DEFAULT_CANDIDATE_ROWS = OUTPUT / "candidate_comparison_workflow_v1_candidate_rows_no_gt.csv"
DEFAULT_SAMPLE_PLAN = OUTPUT / "candidate_comparison_workflow_v1_sample_plan.csv"
DEFAULT_METRIC_ROWS = OUTPUT / "candidate_comparison_stability_v1_candidate_rows_eval.csv"
DEFAULT_SAME_SECTOR_SCORE = "rev_chip_core"
DEFAULT_CROSS_SECTOR_SCORE = "rank_avg_rev_watch"

ALLOWED_VARIANTS = {
    "full_agent",
    "ranker_anchor_agent",
    "no_quant",
    "no_news",
    "no_financial",
    "no_peer",
    "no_bookskill",
}

PROHIBITED_TERMS = ["强烈推荐", "目标价必达", "稳赚", "必涨", "自动下单", "无风险收益", "无风险买入", "无风险操作"]
DEFAULT_OUTPUT_MAX_TOKENS = 6144


ChatFn = Callable[..., dict[str, Any]]


def load_candidate_rows(
    path: Path,
    *,
    same_sector_score: str = DEFAULT_SAME_SECTOR_SCORE,
    cross_sector_score: str = DEFAULT_CROSS_SECTOR_SCORE,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    forbidden = sorted((FUTURE_RESULT_COLUMNS | {"return_20d"}) & set(frame.columns))
    if forbidden:
        raise ValueError(f"candidate rows for agent leaked future/result fields: {forbidden}")
    validate_unique_comparison_groups(frame, source=path)
    return ensure_task_default_score(
        frame,
        same_sector_score=same_sector_score,
        cross_sector_score=cross_sector_score,
    )


def validate_unique_comparison_groups(frame: pd.DataFrame, *, source: Path | str = "candidate_rows") -> None:
    """Ensure one comparison_group_id maps to exactly one candidate set.

    Some stability-audit artifacts append several decision frequencies and can
    reuse CMP ids across different dates. The P1 runner selects by group id, so
    reused ids would silently merge unrelated candidate sets into one Agent
    pack. Fail closed instead.
    """
    required = {"comparison_group_id", "comparison_scenario", "time_block", "date", "code"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing candidate group columns: {sorted(missing)}")
    id_cols = ["comparison_scenario", "time_block", "date"]
    if "decision_frequency" in frame.columns:
        id_cols.append("decision_frequency")
    identity = frame.groupby("comparison_group_id", dropna=False)[id_cols].nunique(dropna=False)
    bad_ids = identity[(identity > 1).any(axis=1)].index.astype(str).tolist()
    if bad_ids:
        raise ValueError(
            f"{source} has reused comparison_group_id across distinct candidate sets; "
            f"examples={bad_ids[:5]}. Build a no-GT file with unique ids, e.g. prefix decision_frequency."
        )
    duplicate_codes = (
        frame.assign(_code6=frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6))
        .duplicated(["comparison_group_id", "_code6"])
    )
    if bool(duplicate_codes.any()):
        bad = frame.loc[duplicate_codes, "comparison_group_id"].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"{source} has duplicate candidate codes inside comparison groups; examples={bad}")


def select_group_ids(
    sample_plan: pd.DataFrame,
    *,
    max_groups: int,
    scenarios: set[str] | None = None,
    panel_index: int = 0,
    groups_per_bucket: int = 0,
) -> list[str]:
    plan = sample_plan.copy()
    if scenarios:
        plan = plan[plan["comparison_scenario"].astype(str).isin(scenarios)].copy()
    plan = plan.drop_duplicates("comparison_group_id")
    if groups_per_bucket > 0:
        rows: list[pd.DataFrame] = []
        keys = ["time_block", "comparison_scenario"]
        sort_cols = [col for col in ["time_block", "comparison_scenario", "repeat_seed", "date", "comparison_group_id"] if col in plan.columns]
        plan = plan.sort_values(sort_cols)
        for _, bucket in plan.groupby(keys, sort=True):
            start = max(0, int(panel_index)) * groups_per_bucket
            end = start + groups_per_bucket
            selected = bucket.iloc[start:end]
            if selected.empty:
                continue
            rows.append(selected)
        plan = pd.concat(rows, ignore_index=True) if rows else plan.head(0)
        if max_groups > 0:
            plan = plan.head(max_groups)
        return plan["comparison_group_id"].astype(str).tolist()
    if max_groups > 0:
        # Stable balanced head per scenario/time block, then fill.
        selected = plan.groupby(["comparison_scenario", "time_block"], group_keys=False).head(1)
        if len(selected) < max_groups:
            rest = plan[~plan["comparison_group_id"].isin(selected["comparison_group_id"])]
            selected = pd.concat([selected, rest.head(max_groups - len(selected))], ignore_index=True)
        plan = selected.head(max_groups)
    return plan["comparison_group_id"].astype(str).tolist()


def build_candidate_packs(
    candidate_rows: pd.DataFrame,
    sample_plan: pd.DataFrame,
    *,
    variants: list[str],
    max_groups: int,
    scenarios: set[str] | None,
    agent_policy_version: str,
    same_sector_score: str = DEFAULT_SAME_SECTOR_SCORE,
    cross_sector_score: str = DEFAULT_CROSS_SECTOR_SCORE,
    panel_index: int = 0,
    groups_per_bucket: int = 0,
) -> list[dict[str, Any]]:
    group_ids = select_group_ids(
        sample_plan,
        max_groups=max_groups,
        scenarios=scenarios,
        panel_index=panel_index,
        groups_per_bucket=groups_per_bucket,
    )
    rows = candidate_rows[candidate_rows["comparison_group_id"].astype(str).isin(set(group_ids))].copy()
    packs: list[dict[str, Any]] = []
    for group_id in group_ids:
        group = rows[rows["comparison_group_id"].astype(str).eq(group_id)].copy()
        if group.empty:
            continue
        for variant in variants:
            pack = build_candidate_pack(
                group,
                variant=variant,
                agent_policy_version=agent_policy_version,
                same_sector_score=same_sector_score,
                cross_sector_score=cross_sector_score,
            )
            packs.append(pack)
    return packs


def build_candidate_pack(
    group: pd.DataFrame,
    *,
    variant: str,
    agent_policy_version: str,
    same_sector_score: str = DEFAULT_SAME_SECTOR_SCORE,
    cross_sector_score: str = DEFAULT_CROSS_SECTOR_SCORE,
) -> dict[str, Any]:
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    group = ensure_task_default_score(
        group,
        same_sector_score=same_sector_score,
        cross_sector_score=cross_sector_score,
    )
    duplicated = group["code"].astype(str).str.zfill(6).duplicated()
    if bool(duplicated.any()):
        bad_codes = group.loc[duplicated, "code"].astype(str).str.zfill(6).drop_duplicates().tolist()
        raise ValueError(f"candidate group {group.iloc[0].get('comparison_group_id')} has duplicate codes: {bad_codes[:5]}")
    group["_rank_order"] = pd.to_numeric(group.get(P1_DEFAULT_SCORE), errors="coerce").fillna(-999).rank(method="first", ascending=False)
    group = group.sort_values(["_rank_order", "code"]).drop(columns=["_rank_order"])
    meta = group.iloc[0]
    candidates = [_candidate_payload(row) for _, row in group.iterrows()]
    pack: dict[str, Any] = {
        "type": "candidate_comparison_evidence_pack",
        "agent_policy_version": agent_policy_version,
        "variant": variant,
        "task_mode": "candidate_comparison",
        "comparison_group_id": str(meta["comparison_group_id"]),
        "comparison_scenario": str(meta["comparison_scenario"]),
        "decision_frequency": str(meta.get("decision_frequency") or ""),
        "decision_date": str(meta["date"]),
        "valid_block": str(meta["time_block"]),
        "candidate_count": int(meta["candidate_count"]),
        "industry_context": str(meta.get("industry_context") or ""),
        "task_mode_requirement": (
            "P1候选对比：用户已给2-20支候选。必须输出候选操作优先级、每支股票操作建议和辅助研究分级；"
            "同领域重视相对比较，跨领域重视风险/信息缺口。可以给买入/卖出/加减仓/持有/等待建议，但不得承诺收益或自动执行。"
        ),
        "allowed_research_grades": sorted(ALLOWED_RESEARCH_GRADES),
        "candidate_score_policy": (
            f"p1_default_selector_v1: same_sector使用{same_sector_score}；cross_sector使用{cross_sector_score}"
            "；rank_avg_rev_watch为候选组内rev_chip_core与single_watch_proxy标准化相加，优先稳定Top2；"
            "跨领域只给Top1/Top2研究优先级，不强行宣称唯一胜者。"
        ),
        "default_ranked_candidates": _default_ranked_candidates(group),
        "ablation_policy": variant,
        "candidates": candidates,
        "research_only": True,
        "not_investment_instruction": True,
    }
    if variant == "ranker_anchor_agent":
        pack["anchor_policy"] = (
            "必须以default_ranked_candidates作为排序锚点。Agent职责是审计硬反证、解释差异和信息缺口；"
            "若没有明确负面新闻/财报风险/同行显著落后/过热高波动/筹码强上压/BookSkill失效等硬反证，"
            "top_research_codes应优先保持默认Top1/Top2。若调整默认Top2，必须在rank_override_audit写明硬反证。"
        )
    apply_variant_ablation(pack, variant)
    assert_no_future_fields(pack)
    return pack


def _default_ranked_candidates(group: pd.DataFrame) -> list[dict[str, Any]]:
    ranked = group.copy()
    ranked["_default_score"] = pd.to_numeric(ranked.get(P1_DEFAULT_SCORE), errors="coerce").fillna(-999.0)
    ranked = ranked.sort_values(["_default_score", "code"], ascending=[False, True]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for idx, row in ranked.iterrows():
        rows.append(
            {
                "default_rank": idx + 1,
                "code": str(row.get("code", "")).zfill(6),
                "name": _text(row.get("name")),
                "default_score": _safe_value(row.get(P1_DEFAULT_SCORE)),
                "rev_chip_core": _safe_value(row.get("rev_chip_core")),
                "single_watch_proxy": _safe_value(row.get("single_watch_proxy")),
                "rank_avg_rev_watch": _safe_value(row.get("rank_avg_rev_watch")),
                "ml_ridge_walkforward_v1": _safe_value(row.get("ml_ridge_walkforward_v1")),
                "ml_hgbr_walkforward_v1": _safe_value(row.get("ml_hgbr_walkforward_v1")),
                "research_use": "ranking_anchor_not_future_label",
            }
        )
    return rows


def _candidate_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": str(row.get("code", "")).zfill(6),
        "name": _text(row.get("name")),
        "industry": _text(row.get("tushare_industry")),
        "area": _text(row.get("tushare_area")),
        "scores": {col: _safe_value(row.get(col)) for col in SCORE_COLUMNS if col in row.index},
        "features": {col: _safe_value(row.get(col)) for col in SAFE_AGENT_FEATURES if col in row.index},
    }
    return payload


def apply_variant_ablation(pack: dict[str, Any], variant: str) -> None:
    if variant in {"full_agent", "ranker_anchor_agent"}:
        return
    for candidate in pack.get("candidates", []):
        scores = candidate.get("scores", {})
        features = candidate.get("features", {})
        if variant == "no_quant":
            scores = {"hidden_by_ablation": True}
        elif variant == "no_news":
            _hide_prefixes(features, ["news_", "policy_background", "official_confirmation", "announcement_materiality"])
        elif variant == "no_financial":
            _hide_prefixes(features, ["financial_"])
        elif variant == "no_peer":
            _hide_prefixes(features, ["corr_peer_", "tushare_industry_relative", "tushare_industry_positive", "tushare_industry_news", "tushare_area_relative"])
        elif variant == "no_bookskill":
            features["triggered_skills"] = "hidden_by_ablation"
        candidate["features"] = features
        candidate["scores"] = scores


def _hide_prefixes(features: dict[str, Any], prefixes: list[str]) -> None:
    for key in list(features):
        if any(key.startswith(prefix) for prefix in prefixes):
            features[key] = "hidden_by_ablation"


def build_candidate_messages(pack: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是A股研究辅助Agent，只能输出严格JSON对象。"
        "你可以输出买入、卖出、加仓、减仓、持有、等待或补数据等操作建议，但必须是研究辅助型建议，不能自动下单、不能承诺收益、不能写目标价必达/稳赚/必涨。"
        "你仍需使用辅助研究分级：继续深挖、放入观察、暂时剔除、信息不足。"
        "这是候选对比任务：一次阅读完整候选组，输出Top1/Top2操作优先级、每支股票建议和分级。"
        "同领域候选主要做相对比较；跨领域候选若证据不足，优先给1-2支条件化操作优先级而不是收益保证。"
        "若variant=ranker_anchor_agent，必须以default_ranked_candidates为排序锚点；只有出现明确硬反证，才能调整默认Top1/Top2，并在rank_override_audit解释。"
        "默认Top1/Top2若无硬反证，应给条件化小仓/持有阈值；若给0仓或等待，必须说明硬反证。"
        "量化分数是工具层，不是最终结论；必须结合新闻/财报/同行/BookSkill/K线/筹码/缺失信息。"
        "若通道被hidden_by_ablation，不能脑补该通道信息。"
        "输出字段必须包括：type, agent_policy_version, variant, comparison_group_id, comparison_scenario, "
        "decision_date, top_research_codes, ranked_candidates, comparison_summary, confidence_level, "
        "data_missing_summary, research_only, not_investment_instruction。"
        "ranked_candidates每项包括rank, code, name, operation_recommendation, position_threshold, buy_or_add_trigger, reduce_or_sell_trigger, research_grade, priority_reason, counter_evidence, data_missing_flags。"
        "position_threshold必须写成具体仓位带，例如新仓0%、10%-20%试探、上限30%、已有仓位降至0%-10%；不能写成'人工复核'或空泛表述。"
        "buy_or_add_trigger和reduce_or_sell_trigger必须包含可复核阈值，例如保持Top2、新闻预警<0.5、财务风险<0.5、跌出Top2、新闻预警>=0.6、财务风险>=0.6、硬反证扩散。"
        "若按锚点排序没有调整，rank_override_audit写none；若调整，写JSON数组并列出code、from_default_rank、to_agent_rank、hard_counter_reason。"
    )
    user = {
        "task": "根据时间安全候选对比证据包输出候选研究优先级JSON。",
        "evidence_pack": pack,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, allow_nan=False)},
    ]


def validate_candidate_card(pack: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(parsed, ensure_ascii=False)
    prohibited = [term for term in PROHIBITED_TERMS if term in text]
    if prohibited:
        raise ValueError(f"candidate card contains prohibited instruction terms: {prohibited}")
    candidates = {str(item.get("code", "")).zfill(6): item for item in pack.get("candidates", [])}
    ranked = parsed.get("ranked_candidates")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("missing ranked_candidates")
    normalized_ranked: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in ranked:
        if not isinstance(item, dict):
            raise ValueError("ranked_candidates item must be object")
        code = str(item.get("code", "")).zfill(6)
        if code not in candidates:
            raise ValueError(f"ranked candidate code not in pack: {code}")
        grade = str(item.get("research_grade") or "")
        if grade not in ALLOWED_RESEARCH_GRADES:
            raise ValueError(f"invalid research_grade: {grade}")
        seen_codes.add(code)
        rank = int(item.get("rank") or len(normalized_ranked) + 1)
        operation = _text(item.get("operation_recommendation") or _operation_from_grade(grade))[:200]
        position_threshold = _text(item.get("position_threshold"))
        buy_trigger = _text(item.get("buy_or_add_trigger"))
        sell_trigger = _text(item.get("reduce_or_sell_trigger"))
        if _is_vague_operation_field(position_threshold):
            position_threshold = _position_threshold_from_grade_rank(grade, rank)
        if _is_vague_operation_field(buy_trigger):
            buy_trigger = _buy_trigger_from_grade_rank(grade, rank)
        if _is_vague_operation_field(sell_trigger):
            sell_trigger = _sell_trigger_from_grade_rank(grade, rank)
        normalized_ranked.append(
            {
                "rank": rank,
                "code": code,
                "name": str(item.get("name") or candidates[code].get("name") or ""),
                "operation_recommendation": operation,
                "position_threshold": position_threshold[:200],
                "buy_or_add_trigger": buy_trigger[:200],
                "reduce_or_sell_trigger": sell_trigger[:200],
                "research_grade": grade,
                "priority_reason": _text(item.get("priority_reason"))[:200],
                "counter_evidence": _text(item.get("counter_evidence") or "none")[:200],
                "data_missing_flags": _text(item.get("data_missing_flags") or "none")[:200],
            }
        )
    top_codes = parsed.get("top_research_codes")
    if not isinstance(top_codes, list):
        top_codes = [item["code"] for item in normalized_ranked[:2]]
    top_codes = [str(code).zfill(6) for code in top_codes if str(code).zfill(6) in candidates][:2]
    if not top_codes:
        top_codes = [item["code"] for item in normalized_ranked[:2]]
    confidence = _confidence_value(parsed.get("confidence_level"))
    confidence = min(1.0, max(0.0, confidence))
    card = {
        "type": "candidate_comparison_decision_card",
        "agent_policy_version": pack["agent_policy_version"],
        "variant": pack["variant"],
        "task_mode": "candidate_comparison",
        "comparison_group_id": pack["comparison_group_id"],
        "comparison_scenario": pack["comparison_scenario"],
        "decision_date": pack["decision_date"],
        "valid_block": pack["valid_block"],
        "candidate_count": pack["candidate_count"],
        "top_research_codes": top_codes,
        "ranked_candidates": normalized_ranked,
        "comparison_summary": _text(parsed.get("comparison_summary"))[:240],
        "confidence_level": confidence,
        "data_missing_summary": _text(parsed.get("data_missing_summary") or "none")[:240],
        "rank_override_audit": _text(parsed.get("rank_override_audit") or "none")[:500],
        "research_only": bool(parsed.get("research_only", True)),
        "not_investment_instruction": bool(parsed.get("not_investment_instruction", True)),
    }
    if not card["research_only"] or not card["not_investment_instruction"]:
        raise ValueError("candidate card must be research_only and not_investment_instruction")
    apply_ranker_anchor_actionability_postcheck(pack, card)
    assert_no_future_fields(card)
    return card


def apply_ranker_anchor_actionability_postcheck(pack: dict[str, Any], card: dict[str, Any]) -> None:
    if str(pack.get("variant")) != "ranker_anchor_agent":
        card["actionability_postcheck_audit"] = "not_applicable"
        return
    default_top2 = [str(item.get("code", "")).zfill(6) for item in pack.get("default_ranked_candidates", [])[:2]]
    candidate_by_code = {str(item.get("code", "")).zfill(6): item for item in pack.get("candidates", [])}
    audit: list[dict[str, Any]] = []
    for item in card.get("ranked_candidates", []):
        code = str(item.get("code", "")).zfill(6)
        if code not in default_top2 or int(item.get("rank") or 99) > 2:
            continue
        grade = str(item.get("research_grade") or "")
        if grade not in {"继续深挖", "放入观察"}:
            continue
        hard_counter_reasons = candidate_hard_counter_reasons(candidate_by_code.get(code, {}))
        if hard_counter_reasons or not is_non_actionable_top_pick(item):
            continue
        rank = int(item.get("rank") or 2)
        old = {
            "operation_recommendation": item.get("operation_recommendation"),
            "position_threshold": item.get("position_threshold"),
            "buy_or_add_trigger": item.get("buy_or_add_trigger"),
            "reduce_or_sell_trigger": item.get("reduce_or_sell_trigger"),
        }
        if rank <= 1:
            item["operation_recommendation"] = "条件化试探买入/继续持有"
            item["position_threshold"] = "新仓10%-20%试探，上限30%；已有仓位可继续持有，硬反证出现前不因软缺口清零。"
        else:
            item["operation_recommendation"] = "小仓试探/继续持有"
            item["position_threshold"] = "新仓5%-10%试探，上限20%；已有仓位可低仓持有，需下一决策点确认。"
        item["buy_or_add_trigger"] = _buy_trigger_from_grade_rank(grade, rank)[:200]
        item["reduce_or_sell_trigger"] = _sell_trigger_from_grade_rank(grade, rank)[:200]
        audit.append(
            {
                "code": code,
                "rank": rank,
                "reason": "default_top2_no_hard_counter_non_actionable_rewritten",
                "old": old,
                "new_position_threshold": item["position_threshold"],
            }
        )
    card["actionability_postcheck_audit"] = audit or "none"


def candidate_hard_counter_reasons(candidate: dict[str, Any]) -> list[str]:
    features = candidate.get("features", {}) if isinstance(candidate, dict) else {}
    reasons: list[str] = []
    if numeric_feature(features, "news_warning_score") >= 0.6:
        reasons.append("news_warning_score>=0.6")
    if numeric_feature(features, "financial_quality_risk_score") >= 0.6:
        reasons.append("financial_quality_risk_score>=0.6")
    if numeric_feature(features, "upper_overhang") >= 0.6:
        reasons.append("upper_overhang>=0.6")
    if numeric_feature(features, "corr_peer_relative_return_20d") <= -8.0:
        reasons.append("corr_peer_relative_return_20d<=-8")
    if numeric_feature(features, "tushare_industry_relative_return_20d") <= -8.0:
        reasons.append("tushare_industry_relative_return_20d<=-8")
    gap_text = str(features.get("data_gaps") or "").lower()
    if "critical" in gap_text or "fatal" in gap_text:
        reasons.append("critical_data_gap")
    return reasons


def numeric_feature(features: dict[str, Any], key: str) -> float:
    value = features.get(key)
    try:
        if value in {None, "", "hidden_by_ablation"}:
            return float("nan")
    except TypeError:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_non_actionable_top_pick(item: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(item.get("operation_recommendation") or ""),
            str(item.get("position_threshold") or ""),
            str(item.get("buy_or_add_trigger") or ""),
        ]
    )
    if "新仓0%-10%" in text or "新仓5%-10%" in text or "新仓10%-20%" in text or "上限20%" in text or "上限30%" in text:
        return False
    if "新仓0%" in text or "新仓0％" in text:
        return True
    if any(term in text for term in ["纯等待", "等待不买", "不建议新开仓", "暂不新增", "新仓建议等待"]):
        return True
    if "等待" in text and "试探" not in text and "持有" not in text:
        return True
    if "新仓" not in text and ("已有仓位0%" in text or "已有仓位0%-" in text):
        return True
    return False


def run_deepseek_packs(
    packs: list[dict[str, Any]],
    *,
    model: str,
    chat_fn: ChatFn = chat_json,
    max_tokens: int = DEFAULT_OUTPUT_MAX_TOKENS,
    timeout: int = 60,
    retries: int = 1,
    max_workers: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    workers = max(1, min(model_concurrency_limit(model) if max_workers <= 0 else max_workers, len(packs) or 1, model_concurrency_limit(model)))
    if workers == 1:
        results = [_run_one(index, pack, model=model, chat_fn=chat_fn, max_tokens=max_tokens, timeout=timeout, retries=retries) for index, pack in enumerate(packs)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, index, pack, model=model, chat_fn=chat_fn, max_tokens=max_tokens, timeout=timeout, retries=retries): index
                for index, pack in enumerate(packs)
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["index"])
    ok = [row["card"] for row in results if row["status"] == "ok"]
    invalid = [row["invalid"] for row in results if row["status"] == "invalid"]
    usage = []
    for row in results:
        usage_row = row["usage"]
        usage_row["requested_max_workers"] = max_workers
        usage_row["effective_workers"] = workers
        usage_row["model_concurrency_limit"] = model_concurrency_limit(model)
        usage.append(usage_row)
    return ok, invalid, usage


def _run_one(
    index: int,
    pack: dict[str, Any],
    *,
    model: str,
    chat_fn: ChatFn,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    last_error = ""
    last_response: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = chat_fn(build_candidate_messages(pack), model=model, max_tokens=max_tokens, timeout=timeout, user_id="stock_agent_candidate_comparison")
            last_response = response
            parsed = extract_json_content(response)
            card = validate_candidate_card(pack, parsed)
            return {"index": index, "status": "ok", "card": card, "usage": _usage_row(index, pack, response, model, attempt, "ok")}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt >= retries:
                invalid = {
                    "index": index,
                    "agent_policy_version": pack.get("agent_policy_version"),
                    "comparison_group_id": pack.get("comparison_group_id"),
                    "variant": pack.get("variant"),
                    "model": model,
                    "error": last_error,
                    "raw_content": _raw_content(last_response),
                    "evidence_pack": pack,
                }
                return {"index": index, "status": "invalid", "invalid": invalid, "usage": _usage_row(index, pack, last_response, model, attempt, "invalid")}
    raise RuntimeError("unreachable candidate comparison runner state")


def candidate_metrics(cards: list[dict[str, Any]], candidate_rows: pd.DataFrame, metric_rows_path: Path | None = None) -> pd.DataFrame:
    if not cards:
        return pd.DataFrame()
    gt = _load_metric_returns(metric_rows_path)
    rows = candidate_rows.merge(gt, on=["date", "code"], how="left")
    metric_rows = []
    for card in cards:
        group = rows[rows["comparison_group_id"].astype(str).eq(str(card["comparison_group_id"]))].copy()
        returns = pd.to_numeric(group["return_20d"], errors="coerce")
        if group.empty or not returns.notna().any():
            continue
        top_codes = [str(code).zfill(6) for code in card.get("top_research_codes", [])][:2]
        if not top_codes:
            top_codes = [str(item.get("code", "")).zfill(6) for item in card.get("ranked_candidates", [])[:2]]
        default_ranked = group.copy()
        default_ranked["_default_score"] = pd.to_numeric(default_ranked.get(P1_DEFAULT_SCORE), errors="coerce").fillna(-999.0)
        default_ranked = default_ranked.sort_values(["_default_score", "code"], ascending=[False, True])
        default_top_codes = default_ranked["code"].astype(str).str.zfill(6).head(2).tolist()
        default_top1 = default_top_codes[0] if default_top_codes else ""
        group = group.copy()
        group["_code6"] = group["code"].astype(str).str.zfill(6)
        return_by_code = pd.to_numeric(group.set_index("_code6")["return_20d"], errors="coerce").to_dict()
        selected_return_values = [return_by_code.get(code, np.nan) for code in top_codes]
        selected_returns = pd.Series(selected_return_values, dtype="float64")
        group_mean = float(returns.mean())
        top1_ret = float(selected_returns.iloc[0]) if len(selected_returns) else np.nan
        top2_mean = float(selected_returns.mean()) if selected_returns.notna().any() else np.nan
        metric_rows.append(
            {
                "variant": card["variant"],
                "comparison_scenario": card["comparison_scenario"],
                "valid_block": card["valid_block"],
                "comparison_group_id": card["comparison_group_id"],
                "top_code_count": len(top_codes),
                "group_mean_return_20d": round(group_mean, 6),
                "top1_return_20d": round(top1_ret, 6) if not math.isnan(top1_ret) else np.nan,
                "top2_mean_return_20d": round(top2_mean, 6) if not math.isnan(top2_mean) else np.nan,
                "top1_excess_20d": round(top1_ret - group_mean, 6) if not math.isnan(top1_ret) else np.nan,
                "top2_excess_20d": round(top2_mean - group_mean, 6) if not math.isnan(top2_mean) else np.nan,
                "top1_positive": bool(top1_ret > 0) if not math.isnan(top1_ret) else False,
                "top2_positive_rate": round(float((selected_returns > 0).mean()), 6) if selected_returns.notna().any() else np.nan,
                "top1_is_worst": bool(top1_ret <= returns.min()) if not math.isnan(top1_ret) else False,
                "regret_vs_best": round(float(returns.max() - top1_ret), 6) if not math.isnan(top1_ret) else np.nan,
                "default_top1_code": default_top1,
                "default_top2_codes": ";".join(default_top_codes),
                "agent_top1_matches_default_top1": bool(top_codes and top_codes[0] == default_top1),
                "agent_top2_overlap_default_top2": round(float(len(set(top_codes[:2]) & set(default_top_codes)) / max(1, len(default_top_codes))), 6),
                "confidence_level": card.get("confidence_level"),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(metric_rows)


def _load_metric_returns(metric_rows_path: Path | None) -> pd.DataFrame:
    if metric_rows_path and metric_rows_path.exists() and metric_rows_path.stat().st_size:
        frame = pd.read_csv(metric_rows_path, usecols=lambda col: col in {"date", "code", "return_20d"}, dtype={"code": str}, low_memory=False)
        frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
        return frame.dropna(subset=["date", "code"]).drop_duplicates(["date", "code"])
    return load_candidate_frame()[["date", "code", "return_20d"]].drop_duplicates(["date", "code"])


def aggregate_candidate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in metrics.groupby(["variant", "comparison_scenario"], sort=True):
        rows.append(
            {
                "variant": keys[0],
                "comparison_scenario": keys[1],
                "cards": int(len(group)),
                "top1_excess_mean": round(float(pd.to_numeric(group["top1_excess_20d"], errors="coerce").mean()), 6),
                "top2_excess_mean": round(float(pd.to_numeric(group["top2_excess_20d"], errors="coerce").mean()), 6),
                "top1_positive_rate": round(float(group["top1_positive"].astype(bool).mean()), 6),
                "top2_positive_rate": round(float(pd.to_numeric(group["top2_positive_rate"], errors="coerce").mean()), 6),
                "top1_worst_rate": round(float(group["top1_is_worst"].astype(bool).mean()), 6),
                "regret_mean": round(float(pd.to_numeric(group["regret_vs_best"], errors="coerce").mean()), 6),
                "top1_anchor_match_rate": round(float(group["agent_top1_matches_default_top1"].astype(bool).mean()), 6),
                "top2_anchor_overlap_mean": round(float(pd.to_numeric(group["agent_top2_overlap_default_top2"], errors="coerce").mean()), 6),
                "avg_confidence": round(float(pd.to_numeric(group["confidence_level"], errors="coerce").mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def write_summary(path: Path, *, packs: list[dict[str, Any]], cards: list[dict[str, Any]], invalid: list[dict[str, Any]], aggregate: pd.DataFrame, model: str, called: bool) -> None:
    lines = [
        "# Candidate Comparison DeepSeek Round",
        "",
        "研究辅助型操作建议，不自动交易，不接券商接口，不承诺收益。",
        "",
        f"- model: `{model}`",
        f"- called_deepseek: `{called}`",
        f"- evidence_packs: `{len(packs)}`",
        f"- ok_cards: `{len(cards)}`",
        f"- invalid_outputs: `{len(invalid)}`",
        "",
        "## Aggregate",
        "",
    ]
    if aggregate.empty:
        lines.append("无可用指标。")
    else:
        lines.append(aggregate.to_markdown(index=False))
    lines.append("")
    lines.append("用户端允许输出买入、卖出、加仓、减仓、持有、等待或补数据建议；必须配套仓位/阈值、证据、反证和风险触发，不得承诺收益或自动执行。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_RESULT_COLUMNS or key == "return_20d":
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek candidate-comparison packs.")
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--sample-plan", type=Path, default=DEFAULT_SAMPLE_PLAN)
    parser.add_argument("--metric-rows", type=Path, default=None, help="Optional eval-only date/code/return_20d cache. Never used in evidence packs.")
    parser.add_argument("--output-prefix", default="candidate_comparison_flash_v1")
    parser.add_argument("--variants", default="full_agent,no_quant,no_news,no_financial")
    parser.add_argument("--scenarios", default="same_sector,cross_sector")
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--selection-source", default="sample_plan", choices=["sample_plan", "candidate_rows"])
    parser.add_argument("--panel-index", type=int, default=0)
    parser.add_argument("--groups-per-bucket", type=int, default=0, help="When >0, take this many groups from each time_block x scenario bucket using panel-index offsets.")
    parser.add_argument("--retry-invalid-prefix", default="", help="Optional previous prefix; only rerun packs whose comparison_group_id/variant were invalid there.")
    parser.add_argument("--agent-policy-version", default="candidate_comparison_agent_v1")
    parser.add_argument("--same-sector-score", default=DEFAULT_SAME_SECTOR_SCORE)
    parser.add_argument("--cross-sector-score", default=DEFAULT_CROSS_SECTOR_SCORE)
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_OUTPUT_MAX_TOKENS,
        help="Candidate-comparison JSON is long; default leaves enough completion room to avoid truncation.",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    scenarios = {item.strip() for item in args.scenarios.split(",") if item.strip()} if args.scenarios else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate_rows = load_candidate_rows(
        args.candidate_rows,
        same_sector_score=args.same_sector_score,
        cross_sector_score=args.cross_sector_score,
    )
    sample_plan = (
        _group_meta_from_candidate_rows(candidate_rows)
        if args.selection_source == "candidate_rows"
        else pd.read_csv(args.sample_plan, dtype={"comparison_group_id": str}, low_memory=False)
    )
    packs = build_candidate_packs(
        candidate_rows,
        sample_plan,
        variants=variants,
        max_groups=args.max_groups,
        scenarios=scenarios,
        agent_policy_version=args.agent_policy_version,
        same_sector_score=args.same_sector_score,
        cross_sector_score=args.cross_sector_score,
        panel_index=args.panel_index,
        groups_per_bucket=args.groups_per_bucket,
    )
    if args.retry_invalid_prefix:
        retry_keys = _load_invalid_pair_keys(args.retry_invalid_prefix)
        packs = [pack for pack in packs if (str(pack.get("comparison_group_id")), str(pack.get("variant"))) in retry_keys]
    prefix = _safe_prefix(args.output_prefix)
    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    metrics_path = OUTPUT / f"{prefix}_metrics.csv"
    aggregate_path = OUTPUT / f"{prefix}_aggregate.csv"
    summary_path = OUTPUT / f"{prefix}_summary.md"
    write_jsonl(str(evidence_path), packs)
    print(f"candidate comparison evidence packs: {len(packs)}")
    print(f"wrote: {evidence_path}")
    if args.call_deepseek:
        cards, invalid, usage = run_deepseek_packs(
            packs,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_workers=args.max_workers,
        )
    else:
        cards, invalid, usage = [], [], []
    write_jsonl(str(decision_path), cards)
    write_jsonl(str(invalid_path), invalid)
    pd.DataFrame(usage).to_csv(usage_path, index=False, encoding="utf-8-sig")
    metrics = candidate_metrics(cards, candidate_rows, metric_rows_path=args.metric_rows) if cards else pd.DataFrame()
    aggregate = aggregate_candidate_metrics(metrics)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    write_summary(summary_path, packs=packs, cards=cards, invalid=invalid, aggregate=aggregate, model=args.model, called=args.call_deepseek)
    print(f"ok_cards={len(cards)} invalid_outputs={len(invalid)}")
    print(f"wrote: {decision_path}")
    print(f"wrote: {invalid_path}")
    print(f"wrote: {usage_path}")
    print(f"wrote: {metrics_path}")
    print(f"wrote: {aggregate_path}")
    print(f"wrote: {summary_path}")


def _group_meta_from_candidate_rows(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "comparison_group_id",
        "comparison_scenario",
        "repeat_seed",
        "time_block",
        "date",
        "candidate_count",
        "candidate_codes",
        "candidate_names",
        "industry_context",
    ]
    keep = [col for col in cols if col in candidate_rows.columns]
    meta = candidate_rows[keep].drop_duplicates("comparison_group_id").copy()
    if "task_mode" not in meta:
        meta["task_mode"] = "candidate_comparison"
    return meta


def _load_invalid_pair_keys(prefix: str) -> set[tuple[str, str]]:
    safe = _safe_prefix(prefix)
    path = OUTPUT / f"{safe}_invalid_outputs.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    keys: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pack = item.get("evidence_pack") if isinstance(item, dict) else {}
        source = pack if isinstance(pack, dict) else item
        keys.add((str(source.get("comparison_group_id")), str(source.get("variant"))))
    if not keys:
        raise ValueError(f"no invalid pair keys found in {path}")
    return keys


def _usage_row(index: int, pack: dict[str, Any], response: dict[str, Any], model: str, attempt: int, status: str) -> dict[str, Any]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {
        "index": index,
        "agent_policy_version": pack.get("agent_policy_version"),
        "comparison_group_id": pack.get("comparison_group_id"),
        "variant": pack.get("variant"),
        "model": model,
        "attempt": attempt,
        "status": status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
    }


def _raw_content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"].get("content", ""))[:2000]
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_") or "candidate_comparison"


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _confidence_value(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip()
        mapping = {
            "高": 0.8,
            "较高": 0.7,
            "中高": 0.65,
            "中等": 0.5,
            "中": 0.5,
            "较低": 0.3,
            "低": 0.2,
        }
        if text in mapping:
            return mapping[text]
        if "高" in text:
            return 0.7
        if "中" in text:
            return 0.5
        if "低" in text:
            return 0.3
    try:
        return min(1.0, max(0.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _operation_from_grade(grade: str) -> str:
    if grade == "继续深挖":
        return "可小仓试探买入或继续持有，等待下一决策点确认后再考虑加仓"
    if grade == "暂时剔除":
        return "新仓回避；已有仓位减仓或卖出复核"
    if grade == "信息不足":
        return "暂不交易，先补关键数据"
    return "暂不新增买入/加仓，等待升级或下调阈值"


def _is_vague_operation_field(value: str) -> bool:
    text = _text(value).strip()
    if not text:
        return True
    vague_terms = [
        "人工复核",
        "未给出",
        "多通道正向确认",
        "出现硬反证",
        "风险扩散",
        "待确认",
        "视情况",
        "不适用",
        "等待",
        "无",
        "none",
        "null",
        "na",
    ]
    has_number = any(ch.isdigit() for ch in text)
    if any(term in text for term in vague_terms) and not has_number:
        return True
    return False


def _position_threshold_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        if rank <= 1:
            return "新仓10%-20%试探，上限30%；已持有者可持有但不追高，二次确认后再考虑加到30%-50%。"
        return "新仓最多10%-15%试探，上限20%；未保持组内Top2前不加仓。"
    if grade == "放入观察":
        return "新仓0%；已有仓位控制在20%-30%观察仓，若后续跌出组内Top2或反证扩大则降至0%-10%。"
    if grade == "暂时剔除":
        return "新仓0%；已有仓位优先降至0%-10%或卖出复核，风险解除前不重新买入。"
    return "新仓/加仓0%；补齐关键数据前不扩大仓位，已有仓位按低仓位处理。"


def _buy_trigger_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        top_gate = "保持组内Top1" if rank <= 1 else "保持组内Top2"
        return f"{top_gate}，且新闻预警<0.5、财务风险<0.5、同行相对不恶化，才允许试探买入/加仓一档。"
    if grade == "放入观察":
        return "重新进入组内Top2，且新闻预警<0.4、财务风险<0.5、同行/BookSkill至少一项转正，才允许10%试探买入。"
    if grade == "暂时剔除":
        return "风险事件解除并连续两个决策点回到组内Top2后，只能先恢复观察，不直接买入。"
    return "补齐行情、新闻/公告、财报披露日和同行数据后，再重新生成买入/加仓阈值。"


def _sell_trigger_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        return "跌出组内Top2，或新闻预警>=0.6、财务风险>=0.6、同行显著走弱/筹码上压时，停止加仓并降至10%-20%或卖出复核。"
    if grade == "放入观察":
        return "未进Top2且新闻预警>=0.6、财务风险>=0.6或硬反证增加时，已有仓位降至0%-10%或卖出复核。"
    if grade == "暂时剔除":
        return "维持卖出/回避；若负面公告继续扩散或价格/同行同步转弱，不保留观察仓。"
    return "关键数据仍缺失且价格/同行同步转弱，已有仓位降至0%-10%或卖出复核。"


if __name__ == "__main__":
    main()
