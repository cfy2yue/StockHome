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
DEFAULT_VARIANT_PRIORITY = [
    "full_agent_with_opportunity_tool",
    "full_agent_with_risk_review_queue",
    "full_agent_without_opportunity_tool",
    "quant_tool_summary_only",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit single-stock risk branch purity from dry-run or Flash evidence packs."
    )
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--compare-prefix", default=None, help="Optional older evidence prefix for branch migration audit.")
    args = parser.parse_args()

    input_prefix = _safe_prefix(args.input_prefix)
    output_prefix = _safe_prefix(args.output_prefix)
    compare_prefix = _safe_prefix(args.compare_prefix) if args.compare_prefix else None

    evidence = _read_jsonl(OUTPUT / f"{input_prefix}_evidence_pack.jsonl")
    gt = _load_ground_truth()
    branches = _representative_branch_rows(evidence)
    detail = _join_ground_truth(branches, gt)
    summary = _branch_summary(detail)

    migration = pd.DataFrame()
    if compare_prefix:
        compare_evidence = _read_jsonl(OUTPUT / f"{compare_prefix}_evidence_pack.jsonl")
        old = _representative_branch_rows(compare_evidence).rename(
            columns={
                "primary_risk_branch": "old_primary_risk_branch",
                "risk_branch_labels": "old_risk_branch_labels",
            }
        )
        migration = _branch_migration(old, branches, gt)

    detail_path = OUTPUT / f"{output_prefix}_detail.csv"
    summary_path = OUTPUT / f"{output_prefix}_summary.csv"
    migration_path = OUTPUT / f"{output_prefix}_migration.csv"
    report_path = OUTPUT / f"{output_prefix}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not migration.empty:
        migration.to_csv(migration_path, index=False, encoding="utf-8-sig")
    _write_report(
        report_path,
        input_prefix=input_prefix,
        compare_prefix=compare_prefix,
        detail=detail,
        summary=summary,
        migration=migration,
        detail_path=detail_path,
        summary_path=summary_path,
        migration_path=migration_path if not migration.empty else None,
    )

    print("A股研究Agent")
    print(f"risk_branch_purity_audit=True input_prefix={input_prefix} output_prefix={output_prefix}")
    print(f"rows={len(detail)} branches={detail['primary_risk_branch'].nunique() if not detail.empty else 0}")
    print(f"wrote: {report_path}")


def _representative_branch_rows(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pack in evidence:
        if str(pack.get("task_mode") or "") not in {"single_stock", "single_stock_watch"}:
            continue
        risk_row = _risk_tool_row(pack)
        if not risk_row:
            continue
        policy = build_single_stock_risk_branch_policy(pack, risk_row)
        branch_labels = risk_row.get("risk_branch_labels")
        if isinstance(branch_labels, list):
            branch_labels_text = ";".join(str(label) for label in branch_labels)
        elif branch_labels:
            branch_labels_text = str(branch_labels)
        else:
            branch_labels_text = ";".join(policy["risk_branch_labels"])
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
                "primary_risk_branch": risk_row.get("primary_risk_branch") or policy["primary_risk_branch"],
                "risk_branch_labels": branch_labels_text,
                "computed_primary_risk_branch": policy["primary_risk_branch"],
                "computed_risk_branch_labels": ";".join(policy["risk_branch_labels"]),
                "branch_action_hint": risk_row.get("branch_policy") or policy["branch_action_hint"],
                "branch_false_veto_risk": risk_row.get("known_false_veto_risk") or policy["branch_false_veto_risk"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["_variant_rank"] = frame["variant"].map({v: i for i, v in enumerate(DEFAULT_VARIANT_PRIORITY)}).fillna(99)
    frame = frame.sort_values(["decision_date", "code", "_variant_rank"]).drop_duplicates(
        ["decision_date", "code"], keep="first"
    )
    return frame.drop(columns=["_variant_rank"]).reset_index(drop=True)


def _join_ground_truth(branches: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if branches.empty:
        return branches
    frame = branches.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    out = frame.merge(gt, left_on=["decision_date", "code"], right_on=["date", "code"], how="left")
    out["return_20d"] = pd.to_numeric(out.get("return_20d"), errors="coerce")
    return out.drop(columns=[col for col in ["date"] if col in out.columns])


def _branch_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for branch, group in detail.groupby("primary_risk_branch", sort=True):
        ret = pd.to_numeric(group["return_20d"], errors="coerce")
        rows.append(
            {
                "primary_risk_branch": branch,
                "rows": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "positive_20d_rate": round(float((ret > 0).mean()), 4) if ret.notna().any() else None,
                "avg_return_20d": round(float(ret.mean()), 4) if ret.notna().any() else None,
                "median_return_20d": round(float(ret.median()), 4) if ret.notna().any() else None,
                "loss_gt5_rate": round(float((ret < -5).mean()), 4) if ret.notna().any() else None,
                "missing_gt_rows": int(ret.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _branch_migration(old: pd.DataFrame, new: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if old.empty or new.empty:
        return pd.DataFrame()
    keys = ["decision_date", "code"]
    keep_old = [*keys, "old_primary_risk_branch", "old_risk_branch_labels"]
    keep_new = [*keys, "name", "primary_risk_branch", "risk_branch_labels"]
    merged = old[keep_old].merge(new[keep_new], on=keys, how="inner")
    merged = _join_ground_truth(merged, gt)
    merged["branch_changed"] = merged["old_primary_risk_branch"] != merged["primary_risk_branch"]
    return merged.sort_values(["branch_changed", "decision_date", "code"], ascending=[False, True, True]).reset_index(drop=True)


def _risk_tool_row(pack: dict[str, Any]) -> dict[str, Any] | None:
    for row in pack.get("quant_tool_summaries") or []:
        if isinstance(row, dict) and row.get("tool_id") == "single_stock_risk_calibration_v2_review_queue":
            return row
    return None


def _load_ground_truth() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in GT_SOURCES:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if {"date", "code", "return_20d"} <= set(frame.columns):
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            frames.append(frame[["date", "code", "return_20d"]])
    if not frames:
        return pd.DataFrame(columns=["date", "code", "return_20d"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code"], keep="last")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_number} in {path} is not a JSON object")
            rows.append(obj)
    return rows


def _write_report(
    path: Path,
    *,
    input_prefix: str,
    compare_prefix: str | None,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    migration: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
    migration_path: Path | None,
) -> None:
    changed = migration[migration.get("branch_changed", pd.Series(dtype=bool)) == True] if not migration.empty else pd.DataFrame()
    lines = [
        "# Single-Stock Risk Branch Purity Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Setup",
        "",
        f"- input_prefix: `{input_prefix}`",
        f"- compare_prefix: `{compare_prefix or 'none'}`",
        f"- rows: `{len(detail)}`",
        f"- branches: `{detail['primary_risk_branch'].nunique() if not detail.empty else 0}`",
        "",
        "## Branch Purity",
        "",
        _table(summary),
        "",
    ]
    if not migration.empty:
        lines.extend(
            [
                "## Branch Migration",
                "",
                f"- changed_rows: `{int(changed.shape[0])}`",
                "",
                _table(changed[[
                    "decision_date",
                    "code",
                    "name",
                    "old_primary_risk_branch",
                    "primary_risk_branch",
                    "return_20d",
                ]] if not changed.empty else changed),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Use",
            "",
            "- This is an offline purity audit: future returns are joined only after evidence-pack construction.",
            "- A branch can be promoted only after leakage-safe evidence construction, paired DS behavior, and branch-level bad-exposure/missed-positive diagnostics.",
            "",
            "## Outputs",
            "",
            f"- `{detail_path.relative_to(ROOT)}`",
            f"- `{summary_path.relative_to(ROOT)}`",
        ]
    )
    if migration_path is not None:
        lines.append(f"- `{migration_path.relative_to(ROOT)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def _safe_prefix(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


if __name__ == "__main__":
    main()
