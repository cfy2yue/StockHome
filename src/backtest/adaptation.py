from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io import write_yaml


def build_adaptation_skills(rules_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
    adaptations = []
    for rule in data.get("rules", []):
        if rule.get("status") != "candidate":
            continue
        source = rule.get("derived_from", {})
        strategy_id = source.get("strategy_id", "UNKNOWN")
        adaptations.append(
            {
                "adaptation_id": f"ADAPT-{strategy_id}-{len(adaptations) + 1:03d}",
                "source_skill": strategy_id,
                "name": _name_for(rule),
                "formula": rule.get("formula"),
                "thresholds": rule.get("thresholds", {}),
                "decision_effect": _decision_effect(strategy_id),
                "applicable_context": {
                    "sector_group": rule.get("applies_to", {}).get("sector_group", []),
                    "cadence": rule.get("applies_to", {}).get("cadence", ""),
                    "exclude": [
                        "新闻或公告发生在决策时间窗之后",
                        "财务披露日无法确认但规则需要财务字段",
                        "test 集反证或样本不足",
                    ],
                },
                "evidence": {
                    "source_report": str(Path(rules_path)),
                    "train_epoch1": rule.get("evidence", {}).get("train_epoch1", {}),
                    "train_epoch2": rule.get("evidence", {}).get("train_epoch2", {}),
                    "test": rule.get("evidence", {}).get("test", {}),
                },
                "source": source,
                "anti_leakage_checks": rule.get("anti_leakage_checks", {}),
                "status": "candidate",
                "next_validation": "500股样本复核；通过后再进入50股相似群精调",
                "reuse_instruction": rule.get("reuse_instruction", "只作为研究辅助，不生成买卖指令。"),
            }
        )
    write_yaml(output_path, {"adaptation_skills": adaptations})
    return adaptations


def _name_for(rule: dict[str, Any]) -> str:
    source = rule.get("derived_from", {})
    return f"{source.get('strategy_id', 'UNKNOWN')} 量化适配规则"


def _decision_effect(strategy_id: str) -> str:
    effects = {
        "PPS-Q-017": "作为趋势结构过滤器；200DMA上方且斜率向上时提高继续深挖候选权重，反向时提高反证风险。",
        "PPS-Q-019": "作为相对强弱过滤器；同日横截面20日强弱前1/3时提高研究优先级。",
        "DOW-B-017": "作为市场/行业一致性过滤器；个股强但行业弱时转入分叉复核。",
        "PPS-Q-009": "作为假突破/假跌破分叉信号；只使用决策日前已经完成的反穿。",
        "CANDLE_MACRO_002": "作为K线风险提示；长上影放量后提高反证权重。",
    }
    return effects.get(strategy_id, "只影响研究分级评分或反证风险，不生成买卖指令。")

