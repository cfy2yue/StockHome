"""Audit promote_context coverage and future-label hygiene."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


FUTURE_NEEDLES = [
    "return_20d",
    "future_return",
    "gt_status",
    "avg20",
    "pos20",
    "pool_excess",
    "loss_gt5",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit promote_context in evidence packs.")
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    rows = _read_jsonl(args.evidence_pack)
    detail = _audit(rows)
    prefix = args.output_prefix or args.evidence_pack.name.replace("_evidence_pack.jsonl", "")
    out_dir = args.evidence_pack.parent
    csv_path = out_dir / f"{prefix}_promote_context_audit.csv"
    md_path = out_dir / f"{prefix}_promote_context_audit.md"
    detail.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_report(md_path, detail, args.evidence_pack)

    print("A股研究Agent")
    print(f"packs: {len(rows)}")
    print(f"context_non_none: {int(detail['context_non_none'].sum()) if not detail.empty else 0}")
    print(f"future_leak_count: {int(detail['future_leak_count'].sum()) if not detail.empty else 0}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _audit(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output = []
    for index, row in enumerate(rows):
        context = str(row.get("promote_context") or "none")
        leaked = [needle for needle in FUTURE_NEEDLES if needle in context]
        output.append(
            {
                "row_index": index,
                "valid_block": row.get("valid_block"),
                "task_mode": row.get("task_mode"),
                "variant": row.get("variant"),
                "code": str(row.get("code", "")).zfill(6),
                "context_non_none": context != "none",
                "walk_forward_prior_only": "walk_forward_prior_only" in context,
                "future_leak_count": len(leaked),
                "future_leak_needles": ",".join(leaked) if leaked else "none",
                "context_preview": context[:500],
            }
        )
    return pd.DataFrame(output)


def _write_report(path: Path, detail: pd.DataFrame, evidence_path: Path) -> None:
    if detail.empty:
        path.write_text("# Promote Context Audit\n\n_无数据_\n", encoding="utf-8")
        return

    block_counts = (
        detail.groupby(["valid_block", "task_mode", "variant"], dropna=False)
        .agg(
            packs=("row_index", "count"),
            context_non_none=("context_non_none", "sum"),
            walk_forward_prior_only=("walk_forward_prior_only", "sum"),
            future_leak_count=("future_leak_count", "sum"),
        )
        .reset_index()
    )
    status = Counter(detail["future_leak_count"].gt(0).map({True: "leak", False: "clean"}))
    lines = [
        "# Promote Context Audit",
        "",
        "本报告只审计 evidence pack 中的正向升级先验上下文，不构成投资建议。",
        "",
        f"- evidence_pack: `{evidence_path}`",
        f"- packs: {len(detail)}",
        f"- context_non_none: {int(detail['context_non_none'].sum())}",
        f"- future_leak_count: {int(detail['future_leak_count'].sum())}",
        f"- status_counts: {dict(status)}",
        "",
        "## By Block / Variant",
        "",
        block_counts.to_markdown(index=False),
        "",
        "## Rules",
        "",
        "- `promote_context` 必须包含 `walk_forward_prior_only` 才能给 Agent 使用。",
        "- 上下文只能包含 `rule_status/agent_use/active_promote_rules`，不得包含收益、胜率、GT 或后验指标。",
        "- 上下文不得包含 `return_20d/future_return/gt_status/avg20/pos20/pool_excess/loss_gt5` 等未来标签或后验指标。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
