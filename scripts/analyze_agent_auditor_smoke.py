"""Analyze DeepSeek Agent Auditor smoke outputs with exposure diagnostics."""
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
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    load_ground_truth,
)


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
FUTURE_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Agent Auditor smoke decision ledger.")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    prefix = _safe_prefix(args.prefix)

    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    out_csv = OUTPUT / f"{prefix}_agent_auditor_diagnostics.csv"
    out_md = OUTPUT / f"{prefix}_agent_auditor_diagnostics.md"
    cards = _read_jsonl(decision_path)
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    detail = _join_returns(cards, frame)
    detail.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _write_report(out_md, detail, cards)
    print("A股研究Agent")
    print(f"cards: {len(cards)}")
    print(f"future_leak_count: {sum(len(_find_future_keys(card)) for card in cards)}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_md}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _join_returns(cards: list[dict[str, Any]], frame: pd.DataFrame) -> pd.DataFrame:
    if not cards:
        return pd.DataFrame()
    source = frame[["date", "code", "return_20d"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    source["code"] = source["code"].astype(str).str.zfill(6)
    card_frame = pd.DataFrame(cards)
    card_frame["date"] = pd.to_datetime(card_frame["decision_date"], errors="coerce").dt.date.astype(str)
    card_frame["code"] = card_frame["code"].astype(str).str.zfill(6)
    merged = card_frame.merge(source, on=["date", "code"], how="left")
    merged["simulated_weight_change"] = pd.to_numeric(merged.get("simulated_weight_change"), errors="coerce")
    merged["is_exposure"] = merged.get("simulated_action", pd.Series(dtype=str)).astype(str).eq("增加研究暴露")
    merged["bad_exposure_negative"] = merged["is_exposure"] & (pd.to_numeric(merged["return_20d"], errors="coerce") < 0)
    merged["bad_exposure_loss_gt5"] = merged["is_exposure"] & (pd.to_numeric(merged["return_20d"], errors="coerce") <= -5)
    merged["quant_or_chip_cited"] = merged.apply(_quant_or_chip_cited, axis=1)
    keep = [
        "variant",
        "task_mode",
        "valid_block",
        "decision_date",
        "code",
        "research_grade",
        "simulated_action",
        "simulated_weight_change",
        "return_20d",
        "is_exposure",
        "bad_exposure_negative",
        "bad_exposure_loss_gt5",
        "quant_or_chip_cited",
        "final_agent_reasoning_summary",
        "research_only",
        "not_investment_instruction",
    ]
    return merged[[col for col in keep if col in merged.columns]].copy()


def _quant_or_chip_cited(row: pd.Series) -> bool:
    text = " ".join(
        str(row.get(field) or "")
        for field in [
            "final_agent_reasoning_summary",
            "counter_evidence",
            "memory_experience_used",
        ]
    )
    needles = ["量化", "筹码", "chip", "ranker", "rev_plus", "rev+chip"]
    return any(item in text for item in needles)


def _write_report(path: Path, detail: pd.DataFrame, cards: list[dict[str, Any]]) -> None:
    leaked = sum(len(_find_future_keys(card)) for card in cards)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby(["variant", "task_mode"], dropna=False)
            .agg(
                cards=("code", "count"),
                exposure_cards=("is_exposure", "sum"),
                bad_exposure_negative=("bad_exposure_negative", "sum"),
                bad_exposure_loss_gt5=("bad_exposure_loss_gt5", "sum"),
                exposure_avg20=("return_20d", lambda s: pd.to_numeric(s[detail.loc[s.index, "is_exposure"]], errors="coerce").mean()),
                quant_or_chip_cited_rate=("quant_or_chip_cited", "mean"),
            )
            .reset_index()
        )
    lines = [
        f"# Agent Auditor Smoke Diagnostics: {path.stem.replace('_agent_auditor_diagnostics', '')}",
        "",
        "本报告只用于研究辅助，不构成投资建议。",
        "",
        f"- cards: {len(cards)}",
        f"- future_leak_count: {leaked}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_无数据_",
        "",
        "## Interpretation Rules",
        "",
        "- `bad_exposure_negative`：研究暴露后 20 日收益为负。",
        "- `bad_exposure_loss_gt5`：研究暴露后 20 日收益不高于 -5%。",
        "- `quant_or_chip_cited_rate`：Agent 理由中是否显式引用量化/筹码/ranker 证据。",
        "- 样本小于 100 张卡时只可视为 smoke，不可 promotion。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_future_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        leaked = {str(key) for key in value if str(key) in FUTURE_FIELDS}
        for child in value.values():
            leaked.update(_find_future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked: set[str] = set()
        for child in value:
            leaked.update(_find_future_keys(child))
        return leaked
    return set()


def _safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in value]
    return "".join(chars).strip("_") or "agent_auditor_smoke"


if __name__ == "__main__":
    main()
