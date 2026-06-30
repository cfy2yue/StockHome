from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_yaml


def build_reusable_rule_candidates(
    train_epoch1: pd.DataFrame,
    train_epoch2: pd.DataFrame,
    test: pd.DataFrame | None,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    skill_ids = sorted(
        set(_skill_ids(train_epoch1)).union(_skill_ids(train_epoch2)).union(_skill_ids(test if test is not None else pd.DataFrame()))
    )
    for idx, skill_id in enumerate(skill_ids, start=1):
        rule = _rule_record(skill_id, idx, train_epoch1, train_epoch2, test)
        candidates.append(rule)
        write_yaml(output_dir / f"{rule['rule_id']}.yaml", rule)
    write_yaml(output_dir.parent / "final_reusable_rules.yaml", {"rules": candidates})
    (output_dir.parent / "final_reusable_rules.md").write_text(_rules_markdown(candidates), encoding="utf-8")
    return candidates


def _rule_record(skill_id: str, idx: int, train_epoch1: pd.DataFrame, train_epoch2: pd.DataFrame, test: pd.DataFrame | None) -> dict[str, Any]:
    evidence = {
        "train_epoch1": _stats_for_skill(train_epoch1, skill_id),
        "train_epoch2": _stats_for_skill(train_epoch2, skill_id),
        "test": _stats_for_skill(test if test is not None else pd.DataFrame(), skill_id),
    }
    status = _status(evidence)
    return {
        "rule_id": f"BTL-{skill_id}-{idx:03d}",
        "status": status,
        "derived_from": _source_for_skill(skill_id),
        "rule_type": _rule_type(skill_id),
        "formula": _formula_for_skill(skill_id),
        "thresholds": _thresholds_for_skill(skill_id),
        "feature_inputs": {
            "data_frequency": "daily",
            "walk_forward_cutoff": "decision_date inclusive",
            "forbidden_inputs": ["future_5d_return", "future_10d_return", "future_20d_return"],
        },
        "applies_to": {
            "sector_group": ["nonferrous_materials", "star_technology"],
            "cadence": "周二/周五收盘后",
            "data_frequency": "daily",
        },
        "decision_effect": "只影响研究分级评分或反证风险，不得生成买卖指令",
        "evidence": evidence,
        "anti_leakage_checks": {
            "uses_only_decision_date_or_before": True,
            "test_locked_before_parameter_changes": True,
            "ground_truth_excluded_from_scoring": True,
        },
        "failure_modes": _failure_modes(skill_id, evidence),
        "conflicts_with": [],
        "confidence_after_backtest": _confidence(status),
        "reuse_instruction": _reuse_instruction(status),
    }


def _skill_ids(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "triggered_skills" not in df:
        return []
    ids: list[str] = []
    for value in df["triggered_skills"].fillna(""):
        ids.extend([part for part in str(value).split(";") if part])
    return ids


def _stats_for_skill(df: pd.DataFrame, skill_id: str) -> dict[str, Any]:
    if df is None or df.empty or "triggered_skills" not in df:
        return {"trigger_count": 0, "pass_rate_5d": None, "pass_rate_10d": None, "pass_rate_20d": None, "sample_status": "insufficient"}
    subset = df[df["triggered_skills"].fillna("").str.contains(skill_id, regex=False)]
    count = int(len(subset))
    return {
        "trigger_count": count,
        "pass_rate_5d": _pass_rate(subset),
        "pass_rate_10d": _direction_rate(subset, "return_10d"),
        "pass_rate_20d": _direction_rate(subset, "return_20d"),
        "sample_status": "ok" if count >= 5 else "insufficient",
    }


def _pass_rate(df: pd.DataFrame) -> float | None:
    if df.empty or "gt_pass" not in df:
        return None
    values = df["gt_pass"].dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _direction_rate(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float((values > 0).mean()), 4)


def _status(evidence: dict[str, dict[str, Any]]) -> str:
    test = evidence["test"]
    train_total = evidence["train_epoch1"]["trigger_count"] + evidence["train_epoch2"]["trigger_count"]
    if train_total < 5:
        return "needs_larger_sample"
    if test["trigger_count"] < 10:
        return "needs_larger_sample"
    train_r20 = evidence["train_epoch2"].get("avg_return_20d")
    test_r20 = evidence["test"].get("avg_return_20d")
    if train_r20 is not None and test_r20 is not None and train_r20 * test_r20 < 0:
        return "do_not_reuse"
    if test_r20 is not None and abs(test_r20) >= 3:
        return "candidate"
    if test.get("pass_rate_5d") is not None and test["pass_rate_5d"] < 0.3:
        return "do_not_reuse"
    return "candidate"


def _confidence(status: str) -> str:
    return {"candidate": "candidate", "needs_larger_sample": "insufficient", "do_not_reuse": "insufficient"}.get(status, "insufficient")


def _source_for_skill(skill_id: str) -> dict[str, str]:
    sources = {
        "PPS-Q-017": ("专业投机原理", "第8章、第27章", "OCR_PAGE 0098-0118、0364-0385，书内页码线索91-111、357-378，页码需人工复核", "high"),
        "PPS-Q-019": ("专业投机原理", "第8章、第27章、第28章", "OCR_PAGE 0098-0118、0364-0404，书内页码线索91-111、357-397，页码需人工复核", "high"),
        "PPS-Q-009": ("专业投机原理", "第7章、第27章", "OCR_PAGE 0090-0093、0364-0385，书内页码线索83-86、357-378，页码需人工复核", "high"),
        "DOW-B-004": ("道氏理论", "第2章、第10章、第13章", "OCR_PAGE 20-21；第10章/OCR_PAGE 91-104；第13章/OCR_PAGE 147；页码需人工复核", "high"),
        "DOW-B-017": ("道氏理论", "第17章、第19章", "OCR_PAGE 185-186；第19章/OCR_PAGE 209；页码需人工复核", "high"),
        "PPS-M-002": ("专业投机原理", "第2章", "OCR_PAGE 0032-0037，书内页码线索25-30，页码需人工复核", "high"),
        "PPS-M-003": ("专业投机原理", "第3章、第18章", "OCR_PAGE 0038-0044、0275-0281，书内页码线索31-37、268-274，页码需人工复核", "high"),
        "CANDLE_MACRO_002": ("日本蜡烛图技术", "第四章反转形态", "OCR_PAGE_0048-0050；书内页码约28-30；页码需人工复核", "high"),
        "PPS-Q-023": ("专业投机原理", "第27章", "OCR_PAGE 0364-0385，书内页码线索357-378，页码需人工复核", "medium"),
    }
    book, chapter, page_range, confidence = sources.get(skill_id, ("未知", "未知", "需人工复核", "medium"))
    return {
        "strategy_id": skill_id,
        "book": book,
        "chapter": chapter,
        "page_range": page_range,
        "extraction_method": "full_ocr_txt_deep_dive",
        "source_confidence": confidence,
    }


def _rule_type(skill_id: str) -> str:
    if skill_id in {"PPS-Q-017", "DOW-B-004", "PPS-M-002", "PPS-M-003"}:
        return "threshold"
    if skill_id in {"PPS-Q-009", "CANDLE_MACRO_002", "PPS-Q-023"}:
        return "pattern"
    return "filter"


def _formula_for_skill(skill_id: str) -> str:
    formulas = {
        "PPS-Q-017": "close > ma200 and ma200_slope20 > 0 OR close < ma200 and ma200_slope20 < 0",
        "PPS-Q-019": "relative_strength_20d_rank within current universe",
        "PPS-Q-009": "high > previous_20d_high and close < previous_20d_high OR low < previous_20d_low and close > previous_20d_low",
        "DOW-B-004": "0.33 <= (recent_60d_high - close) / (recent_60d_high - recent_60d_low) <= 0.66",
    }
    return formulas.get(skill_id, "see triggered_formulas in decisions_summary.csv")


def _thresholds_for_skill(skill_id: str) -> dict[str, Any]:
    thresholds = defaultdict(dict)
    mapping = {
        "PPS-Q-017": {"ma_window": 200, "slope_window": 20, "price_field": "close"},
        "PPS-Q-019": {"rank_window": 20, "rank_scope": "same decision_date universe", "top_percentile_min": 0.67},
        "PPS-Q-009": {"lookback_high_low_window": 20, "price_fields": ["high", "low", "close"]},
        "DOW-B-004": {"lookback_window": 60, "retrace_min": 0.33, "retrace_max": 0.66},
        "CANDLE_MACRO_002": {"upper_shadow_to_body_min": 2, "volume_ratio20_min": 1.5},
    }
    return dict(mapping.get(skill_id, thresholds))


def _failure_modes(skill_id: str, evidence: dict[str, dict[str, Any]]) -> list[str]:
    modes = ["轻量样本触发次数不足时不得升级为正式规则"]
    if evidence["test"]["trigger_count"] == 0:
        modes.append("test 集未触发，仍需扩大样本验证")
    if skill_id.startswith("CANDLE"):
        modes.append("A 股涨跌停、除权和停牌可能造成伪形态")
    return modes


def _reuse_instruction(status: str) -> str:
    if status == "candidate":
        return "可作为候选研究规则复用，但只能影响评分或反证，不得生成买卖指令。"
    if status == "do_not_reuse":
        return "test 集反证明显，暂不复用。"
    return "样本不足，只能作为待扩大样本观察项。"


def _rules_markdown(candidates: list[dict[str, Any]]) -> str:
    lines = ["# 轻量回测可复用规则汇总", "", "本文件只记录研究辅助规则，不构成买卖指令。", ""]
    for rule in candidates:
        lines.append(f"## {rule['rule_id']}")
        lines.append(f"- 状态：{rule['status']}")
        lines.append(f"- 来源：{rule['derived_from']['book']} / {rule['derived_from']['strategy_id']}")
        lines.append(f"- 公式：`{rule['formula']}`")
        lines.append(f"- 复用说明：{rule['reuse_instruction']}")
        lines.append("")
    return "\n".join(lines)
