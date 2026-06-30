from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    load_ground_truth,
)
from scripts.run_full_channel_ablation_round import GT_SOURCES  # noqa: E402


OUTPUT = ROOT / "reports" / "date_generalization"
CASH_RETURN_20D = (1.03 ** (20 / 252) - 1) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BookSkill strategy-id attribution tables from evidence packs and decision cards.")
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--output-prefix", default="bookskill_attribution_v1")
    parser.add_argument("--with-ground-truth", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)
    evidence = _read_jsonl(args.evidence_pack)
    cards = _read_jsonl(args.decision_ledger)
    details = build_bookskill_attribution_rows(evidence, cards)
    if args.with_ground_truth:
        details = attach_posthoc_returns(details)
    aggregate = aggregate_bookskill_attribution(details)

    detail_path = OUTPUT / f"{prefix}_detail.csv"
    aggregate_path = OUTPUT / f"{prefix}_aggregate.csv"
    report_path = OUTPUT / f"{prefix}_summary.md"
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    write_report(report_path, detail=details, aggregate=aggregate, evidence_path=args.evidence_pack, decision_path=args.decision_ledger)

    print("A股研究Agent")
    print(f"detail_rows={len(details)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"wrote={report_path}")


def build_bookskill_attribution_rows(evidence: list[dict[str, Any]], cards: list[dict[str, Any]]) -> pd.DataFrame:
    cards_by_key = {_card_key(card): card for card in cards}
    rows: list[dict[str, Any]] = []
    for pack in evidence:
        card = cards_by_key.get(_card_key(pack), {})
        skills = pack.get("book_skill_candidates") if isinstance(pack.get("book_skill_candidates"), list) else []
        if not skills:
            skills = [
                {
                    "strategy_id": "NO_BOOKSKILL",
                    "source_status": "missing_or_hidden",
                    "source_book": "",
                    "page_range": "",
                    "confidence": "none",
                }
            ]
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            action = str(card.get("simulated_action") or "")
            weight = _safe_float(card.get("simulated_weight_change"), 0.0)
            active = action == "增加研究暴露" or weight >= 0.5
            rows.append(
                {
                    "agent_policy_version": pack.get("agent_policy_version"),
                    "variant": pack.get("variant"),
                    "task_mode": pack.get("task_mode"),
                    "valid_block": pack.get("valid_block"),
                    "decision_date": pack.get("decision_date"),
                    "code": str(pack.get("code") or "").zfill(6),
                    "name": pack.get("name"),
                    "sample_panel_id": pack.get("sample_panel_id") or "panel_01",
                    "strategy_id": skill.get("strategy_id") or "UNKNOWN",
                    "source_book": skill.get("source_book") or "",
                    "source_status": skill.get("source_status") or "",
                    "page_range": skill.get("page_range") or "",
                    "confidence": skill.get("confidence") or "",
                    "has_grounded_source_detail": bool(skill.get("source_book") and skill.get("page_range") and skill.get("source_status") == "grounded"),
                    "research_grade": card.get("research_grade"),
                    "simulated_action": action,
                    "simulated_weight_change": weight,
                    "active_exposure": active,
                    "final_agent_reasoning_summary": card.get("final_agent_reasoning_summary") or "",
                    "book_skill_evidence": card.get("book_skill_evidence") or "",
                    "counter_evidence": card.get("counter_evidence") or "",
                }
            )
    return pd.DataFrame(rows)


def attach_posthoc_returns(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    )
    gt = frame[["date", "code", "return_20d"]].copy()
    gt["date"] = pd.to_datetime(gt["date"], errors="coerce").dt.date.astype(str)
    gt["code"] = gt["code"].astype(str).str.zfill(6)
    data = details.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"], errors="coerce").dt.date.astype(str)
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.merge(gt.rename(columns={"date": "decision_date", "return_20d": "posthoc_return_20d"}), on=["decision_date", "code"], how="left")
    returns = pd.to_numeric(data["posthoc_return_20d"], errors="coerce")
    weights = pd.to_numeric(data["simulated_weight_change"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    data["posthoc_cash_adjusted_return_20d"] = weights * returns + (1 - weights) * CASH_RETURN_20D
    data.loc[returns.isna(), "posthoc_cash_adjusted_return_20d"] = pd.NA
    data["posthoc_positive_20d"] = returns > 0
    return data


def aggregate_bookskill_attribution(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    rows = []
    keys = ["variant", "task_mode", "strategy_id", "source_status", "source_book"]
    for values, group in details.groupby(keys, dropna=False, sort=True):
        row = {key: value for key, value in zip(keys, values)}
        row["cards_with_skill"] = int(len(group))
        row["unique_codes"] = int(group["code"].nunique())
        row["active_exposure_cards"] = int(pd.Series(group["active_exposure"]).fillna(False).sum())
        row["continue_deep_cards"] = int(group["research_grade"].astype(str).eq("继续深挖").sum())
        row["grounded_source_detail_rate"] = round(float(group["has_grounded_source_detail"].mean()), 4)
        if "posthoc_return_20d" in group:
            returns = pd.to_numeric(group["posthoc_return_20d"], errors="coerce")
            cash = pd.to_numeric(group.get("posthoc_cash_adjusted_return_20d"), errors="coerce")
            row["posthoc_avg_return_20d"] = round(float(returns.mean()), 4) if returns.notna().any() else None
            row["posthoc_positive_20d_rate"] = round(float((returns > 0).mean()), 4) if returns.notna().any() else None
            row["posthoc_cash_adjusted_avg_return_20d"] = round(float(cash.mean()), 4) if cash.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(report_path: Path, *, detail: pd.DataFrame, aggregate: pd.DataFrame, evidence_path: Path, decision_path: Path) -> None:
    active = detail[detail["active_exposure"].fillna(False)] if not detail.empty else pd.DataFrame()
    lines = [
        "# BookSkill Attribution Report",
        "",
        "本报告只用于研究辅助，不构成投资建议，不接券商，不自动交易。",
        "",
        f"- evidence_pack: `{evidence_path}`",
        f"- decision_ledger: `{decision_path}`",
        f"- detail_rows: `{len(detail)}`",
        f"- aggregate_rows: `{len(aggregate)}`",
        f"- active_exposure_rows: `{len(active)}`",
        "",
        "## Aggregate",
        "",
        _table(aggregate.head(80)),
        "",
        "## Active Exposure Detail",
        "",
        _table(active.head(50)),
        "",
        "## Interpretation",
        "",
        "- `NO_BOOKSKILL` 表示该 variant 隐藏了 BookSkill 或 evidence pack 未解析出候选。",
        "- 同一张 card 可能触发多条 BookSkill，因此 strategy-level posthoc 指标会重复计数，不能直接相加。",
        "- 只有跨 panel、跨时间块反复阻止坏主动暴露的具体 strategy_id，才允许提高优先级。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _card_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("agent_policy_version"),
        row.get("variant"),
        row.get("step"),
        row.get("valid_block"),
        row.get("decision_date"),
        str(row.get("code") or "").zfill(6),
        row.get("task_mode"),
        row.get("sample_panel_id") or "panel_01",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(number) else number


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "bookskill_attribution_v1"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()
