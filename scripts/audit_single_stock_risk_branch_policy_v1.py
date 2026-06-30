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

from src.agent_training.risk_branch_policy import build_single_stock_risk_branch_policy


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit single-stock risk branch policy on an existing full-channel round.")
    parser.add_argument("--input-prefix", default="single_stock_opportunity_risk_queue_panel_flash_v1")
    parser.add_argument("--output-prefix", default="single_stock_risk_branch_policy_v1")
    args = parser.parse_args()

    input_prefix = _safe_prefix(args.input_prefix)
    output_prefix = _safe_prefix(args.output_prefix)
    evidence = _read_jsonl(OUTPUT / f"{input_prefix}_evidence_pack.jsonl")
    cards = _read_jsonl(OUTPUT / f"{input_prefix}_decision_ledger.jsonl")
    gt = _load_ground_truth()

    branch_rows = _branch_rows(evidence)
    card_frame = _card_frame(cards, gt)
    detail = _join_branch_and_cards(branch_rows, card_frame)
    branch_summary = _branch_summary(detail)
    paired = _paired_diagnostics(detail)

    detail_path = OUTPUT / f"{output_prefix}_detail.csv"
    summary_path = OUTPUT / f"{output_prefix}_summary.csv"
    paired_path = OUTPUT / f"{output_prefix}_paired.csv"
    report_path = OUTPUT / f"{output_prefix}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    branch_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    _write_report(
        report_path,
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        detail=detail,
        branch_summary=branch_summary,
        paired=paired,
        detail_path=detail_path,
        summary_path=summary_path,
        paired_path=paired_path,
    )
    print("A股研究Agent")
    print(f"risk_branch_audit=True input_prefix={input_prefix} output_prefix={output_prefix}")
    print(f"rows={len(detail)} branches={detail['primary_risk_branch'].nunique() if not detail.empty else 0}")
    print(f"wrote: {report_path}")


def _branch_rows(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pack in evidence:
        if str(pack.get("task_mode") or "") not in {"single_stock", "single_stock_watch"}:
            continue
        risk_row = _risk_tool_row(pack)
        if not risk_row:
            continue
        policy = build_single_stock_risk_branch_policy(pack, risk_row)
        rows.append(
            {
                "variant": str(pack.get("variant") or ""),
                "decision_date": str(pack.get("decision_date") or ""),
                "code": str(pack.get("code") or "").zfill(6),
                "name": pack.get("name"),
                "risk_tier": risk_row.get("risk_tier"),
                "risk_score": risk_row.get("score"),
                "risk_priority": risk_row.get("score_quantile"),
                "tool_grade": risk_row.get("tool_grade"),
                "primary_risk_branch": policy["primary_risk_branch"],
                "risk_branch_labels": ";".join(policy["risk_branch_labels"]),
                "branch_action_hint": policy["branch_action_hint"],
                "branch_false_veto_risk": policy["branch_false_veto_risk"],
            }
        )
    return pd.DataFrame(rows)


def _card_frame(cards: list[dict[str, Any]], gt: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(cards)
    if frame.empty:
        return frame
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["simulated_weight_change"] = pd.to_numeric(frame.get("simulated_weight_change"), errors="coerce").fillna(0.0)
    frame = frame.merge(gt, left_on=["decision_date", "code"], right_on=["date", "code"], how="left")
    frame["return_20d"] = pd.to_numeric(frame.get("return_20d"), errors="coerce")
    frame["cash_adjusted_return_20d"] = frame["simulated_weight_change"] * frame["return_20d"] + (1 - frame["simulated_weight_change"]) * 0.2349
    return frame


def _join_branch_and_cards(branch_rows: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    if branch_rows.empty or cards.empty:
        return pd.DataFrame()
    branch_representative = branch_rows.copy()
    branch_representative["_variant_rank"] = branch_representative["variant"].map(
        {
            "full_agent_with_opportunity_tool": 0,
            "full_agent_with_risk_review_queue": 1,
            "full_agent_without_opportunity_tool": 2,
            "quant_tool_summary_only": 3,
        }
    ).fillna(9)
    branch_representative = branch_representative.sort_values(["decision_date", "code", "_variant_rank"]).drop_duplicates(
        ["decision_date", "code"],
        keep="first",
    )
    keep = [
        "variant",
        "decision_date",
        "code",
        "research_grade",
        "simulated_action",
        "simulated_weight_change",
        "return_20d",
        "cash_adjusted_return_20d",
    ]
    branch_cols = [
        "decision_date",
        "code",
        "name",
        "risk_tier",
        "risk_score",
        "risk_priority",
        "tool_grade",
        "primary_risk_branch",
        "risk_branch_labels",
        "branch_action_hint",
        "branch_false_veto_risk",
    ]
    data = cards[keep].merge(branch_representative[branch_cols], on=["decision_date", "code"], how="left")
    return data.sort_values(["decision_date", "code", "variant"]).reset_index(drop=True)


def _branch_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["primary_risk_branch", "variant"]
    for (branch, variant), group in detail.groupby(keys, sort=True):
        ret = pd.to_numeric(group["return_20d"], errors="coerce")
        weight = pd.to_numeric(group["simulated_weight_change"], errors="coerce").fillna(0.0)
        cash = pd.to_numeric(group["cash_adjusted_return_20d"], errors="coerce")
        rows.append(
            {
                "primary_risk_branch": branch,
                "variant": variant,
                "cards": int(len(group)),
                "positive_20d_rate": round(float((ret > 0).mean()), 4) if ret.notna().any() else None,
                "avg_return_20d": round(float(ret.mean()), 4) if ret.notna().any() else None,
                "avg_weight": round(float(weight.mean()), 4),
                "cash_adjusted_avg20": round(float(cash.mean()), 4) if cash.notna().any() else None,
                "temporary_exclude_cards": int(group["research_grade"].astype(str).eq("暂时剔除").sum()),
                "insufficient_cards": int(group["research_grade"].astype(str).eq("信息不足").sum()),
            }
        )
    return pd.DataFrame(rows)


def _paired_diagnostics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    base_cols = ["decision_date", "code", "primary_risk_branch"]
    dedup = detail.drop_duplicates(["variant", *base_cols])
    weight = dedup.pivot_table(index=base_cols, columns="variant", values="simulated_weight_change", aggfunc="first")
    ret = dedup.drop_duplicates(base_cols).set_index(base_cols)["return_20d"]
    pairs = [
        ("full_agent_with_opportunity_tool", "full_agent_without_opportunity_tool", "opportunity_vs_no_opportunity"),
        ("full_agent_with_risk_review_queue", "full_agent_without_risk_review_queue", "risk_vs_no_risk"),
        ("full_agent_with_opportunity_tool", "full_agent_with_risk_review_queue", "opportunity_vs_risk_queue"),
    ]
    rows: list[dict[str, Any]] = []
    for left, right, label in pairs:
        if left not in weight or right not in weight:
            continue
        delta = (weight[left] - weight[right]).dropna()
        joined = pd.DataFrame({"delta_weight": delta, "return_20d": ret.loc[delta.index]})
        for branch, group in joined.groupby(level="primary_risk_branch", sort=True):
            changed = group[group["delta_weight"] != 0]
            rows.append(
                {
                    "comparison": label,
                    "primary_risk_branch": branch,
                    "pairs": int(len(group)),
                    "changed_pairs": int(len(changed)),
                    "lowered_negative": int(((changed["delta_weight"] < 0) & (changed["return_20d"] < 0)).sum()),
                    "lowered_positive": int(((changed["delta_weight"] < 0) & (changed["return_20d"] > 0)).sum()),
                    "raised_negative": int(((changed["delta_weight"] > 0) & (changed["return_20d"] < 0)).sum()),
                    "raised_positive": int(((changed["delta_weight"] > 0) & (changed["return_20d"] > 0)).sum()),
                    "avg_delta_weight": round(float(group["delta_weight"].mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    input_prefix: str,
    output_prefix: str,
    detail: pd.DataFrame,
    branch_summary: pd.DataFrame,
    paired: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
    paired_path: Path,
) -> None:
    false_veto = paired[(paired.get("lowered_positive", pd.Series(dtype=int)) > 0)] if not paired.empty else pd.DataFrame()
    lines = [
        "# Single-Stock Risk Branch Policy v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- input_prefix: `{input_prefix}`",
        f"- output_prefix: `{output_prefix}`",
        f"- detail_rows: `{len(detail)}`",
        f"- unique_stock_dates: `{detail[['decision_date', 'code']].drop_duplicates().shape[0] if not detail.empty else 0}`",
        f"- branches: `{detail['primary_risk_branch'].nunique() if not detail.empty else 0}`",
        "",
        "## Branch Summary",
        "",
        _table(branch_summary),
        "",
        "## Paired Diagnostics",
        "",
        _table(paired),
        "",
        "## False-Veto Branches",
        "",
        _table(false_veto),
        "",
        "## Decision",
        "",
        "- 分叉规则只使用当前 evidence pack 字段，不读取未来收益；未来收益只用于本离线审计报告。",
        "- `low_hard_counter_with_reversal_support` 不应被风险队列机械压权；它只应阻止上调，并要求明确负面事件/财报风险再降权。",
        "- `overheat_reversal_friction_without_hard_event` 可保持低权重观察和复核，但不得在缺少明确事件时升级为硬剔除或归零。",
        "- explicit hard negative event 可以更强降权，但仍需 source quality、timestamp、财报 available_at 与 BookSkill 条件确认。",
        "",
        "## Outputs",
        "",
        f"- `{detail_path.relative_to(ROOT)}`",
        f"- `{summary_path.relative_to(ROOT)}`",
        f"- `{paired_path.relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _risk_tool_row(pack: dict[str, Any]) -> dict[str, Any] | None:
    rows = pack.get("quant_tool_summaries")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("tool_id") == "single_stock_risk_calibration_v2_review_queue":
            return row
    return None


def _load_ground_truth() -> pd.DataFrame:
    frames = []
    for path in GT_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        keep = [col for col in ["date", "code", "return_20d"] if col in frame]
        frames.append(frame[keep])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "return_20d"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"], keep="last")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "risk_branch_policy_v1"


if __name__ == "__main__":
    main()
