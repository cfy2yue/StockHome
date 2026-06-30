"""Build a low-cost latest rolling risk register for P0/P1 delivery.

This script does not call external APIs, does not read secrets, and does not
create trading instructions. It turns already-completed latest rolling smokes
into a machine-checkable product risk register so the project can avoid
overclaiming final generalization after small or defensive latest-block tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "latest_rolling_product_risk_register_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--p0-latest-metrics",
        type=Path,
        default=REPORT_DIR / "p0_latest_rolling_micro_flash_smoke_v3_metrics.csv",
    )
    parser.add_argument(
        "--p1-latest-metrics",
        type=Path,
        default=REPORT_DIR / "p1_rolling_cross_sector_anchor_flash_smoke_v3_postcheck_metrics.csv",
    )
    parser.add_argument(
        "--rolling-preflight-gates",
        type=Path,
        default=REPORT_DIR / "rolling_confirmation_preflight_v1_gates.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p0 = summarize_p0_latest(args.p0_latest_metrics)
    p1 = summarize_p1_latest(args.p1_latest_metrics)
    preflight = summarize_preflight(args.rolling_preflight_gates)
    gates = build_gate_table(p0, p1, preflight)
    write_outputs(args.output_prefix, p0, p1, preflight, gates)
    print(f"wrote: {REPORT_DIR / f'{safe_prefix(args.output_prefix)}.md'}")
    print(gates.to_string(index=False))


def summarize_p0_latest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "artifact": display_path(path),
            "status": "missing",
            "cards": 0,
            "invalid_outputs": 0,
            "exposure_cards": 0,
            "cash_pos20": float("nan"),
            "cash_avg20": float("nan"),
            "active_exposure": float("nan"),
            "data_missing_cards": 0,
            "reason": "latest P0 Flash smoke metrics missing",
        }
    frame = pd.read_csv(path, low_memory=False)
    cards = _sum(frame, "decision_cards")
    invalid = _sum(frame, "invalid_outputs")
    exposure_cards = _sum(frame, "exposure_cards")
    cash_pos20 = _mean(frame, "cash_adjusted_positive_20d_rate")
    cash_avg20 = _mean(frame, "cash_adjusted_avg_return_20d")
    active_exposure = _mean(frame, "active_exposure")
    data_missing = _sum(frame, "data_missing_flag_cards")
    if cards <= 0:
        status = "missing"
        reason = "latest P0 smoke has no decision cards"
    elif invalid > 0:
        status = "not_confirmed_invalid_outputs"
        reason = "latest P0 smoke has invalid model outputs"
    elif exposure_cards == 0:
        status = "not_confirmed_zero_exposure"
        reason = "latest P0 smoke produced no active buy/add exposure; this is defensive behavior, not stock-picking confirmation"
    elif cards < 20:
        status = "not_confirmed_tiny_sample"
        reason = "latest P0 smoke has too few cards for a product confirmation"
    elif cash_pos20 < 0.60 or cash_avg20 <= 0:
        status = "not_confirmed_low_latest_metric"
        reason = "latest P0 smoke does not clear the positive-rate/mean-return risk threshold"
    else:
        status = "candidate_confirmation_needs_ablation"
        reason = "latest P0 smoke is directionally positive but still needs paired ablation and panel confirmation"
    return {
        "artifact": display_path(path),
        "status": status,
        "cards": cards,
        "invalid_outputs": invalid,
        "exposure_cards": exposure_cards,
        "cash_pos20": cash_pos20,
        "cash_avg20": cash_avg20,
        "active_exposure": active_exposure,
        "data_missing_cards": data_missing,
        "reason": reason,
    }


def summarize_p1_latest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "artifact": display_path(path),
            "status": "missing",
            "cards": 0,
            "top1_excess": float("nan"),
            "top2_excess": float("nan"),
            "top1_positive": float("nan"),
            "top1_worst": float("nan"),
            "anchor_match": float("nan"),
            "top2_overlap": float("nan"),
            "reason": "latest P1 Flash smoke metrics missing",
        }
    frame = pd.read_csv(path, low_memory=False)
    cards = int(len(frame))
    top1_excess = _mean(frame, "top1_excess_20d")
    top2_excess = _mean(frame, "top2_excess_20d")
    top1_positive = _bool_mean(frame, "top1_positive")
    top1_worst = _bool_mean(frame, "top1_is_worst")
    anchor_match = _bool_mean(frame, "agent_top1_matches_default_top1")
    top2_overlap = _mean(frame, "agent_top2_overlap_default_top2")
    if cards <= 0:
        status = "missing"
        reason = "latest P1 smoke has no valid cards"
    elif cards < 4:
        status = "partial_sorting_smoke_not_confirmation"
        reason = "latest P1 smoke is smaller than the planned bounded panel; treat as partial sorting evidence only"
    elif top2_excess <= 0 or top1_worst > 0.15 or anchor_match < 0.90:
        status = "not_confirmed_metric_gate_failed"
        reason = "latest P1 smoke fails at least one sorting/anchor risk gate"
    elif cards < 12:
        status = "partial_positive_not_confirmation"
        reason = "latest P1 smoke is positive but still too small for rolling confirmation"
    else:
        status = "candidate_confirmation_needs_fresh_panel"
        reason = "latest P1 smoke is positive and anchor-safe but still needs larger fresh-panel validation"
    return {
        "artifact": display_path(path),
        "status": status,
        "cards": cards,
        "top1_excess": top1_excess,
        "top2_excess": top2_excess,
        "top1_positive": top1_positive,
        "top1_worst": top1_worst,
        "anchor_match": anchor_match,
        "top2_overlap": top2_overlap,
        "reason": reason,
    }


def summarize_preflight(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"artifact": display_path(path), "status": "missing", "next_step": "missing"}
    frame = pd.read_csv(path, low_memory=False)
    if "gate" not in frame or "status" not in frame:
        return {"artifact": display_path(path), "status": "malformed", "next_step": "malformed"}
    by_gate = {str(row["gate"]): str(row["status"]) for _, row in frame.iterrows()}
    return {
        "artifact": display_path(path),
        "status": by_gate.get("rolling_confirmation_next_step", "missing"),
        "next_step": by_gate.get("rolling_confirmation_next_step", "missing"),
        "p0_sample_plan": by_gate.get("P0_latest_sample_plan", "missing"),
        "p0_dryrun": by_gate.get("P0_latest_dryrun_evidence", "missing"),
        "p1_preflight": by_gate.get("P1_rolling_newdata_preflight", "missing"),
    }


def build_gate_table(p0: dict[str, Any], p1: dict[str, Any], preflight: dict[str, Any]) -> pd.DataFrame:
    p0_confirmed = str(p0["status"]).startswith("candidate_confirmation")
    p1_confirmed = str(p1["status"]).startswith("candidate_confirmation")
    risk_logged = (
        p0["status"] != "missing"
        and p1["status"] != "missing"
        and preflight.get("next_step") in {"ready_for_bounded_flash", "not_ready", "missing"}
    )
    if p0_confirmed and p1_confirmed:
        overall = "candidate_needs_final_ablation"
        next_action = "Run paired ablation/fresh panel before any Pro or final-completion claim."
    else:
        overall = "logged_not_complete"
        next_action = "Do not expand DS broadly. Keep P0/P1 strong-yellow MVP; use latest rolling as risk disclosure until larger bounded panels pass."
    rows = [
        {
            "gate": "P0_latest_flash_confirmation",
            "status": p0["status"],
            "evidence": (
                f"cards={p0['cards']}, invalid={p0['invalid_outputs']}, exposure_cards={p0['exposure_cards']}, "
                f"cash_pos20={_fmt(p0['cash_pos20'])}, cash_avg20={_fmt(p0['cash_avg20'])}, "
                f"data_missing_cards={p0['data_missing_cards']}"
            ),
            "next_action": p0["reason"],
        },
        {
            "gate": "P1_latest_flash_confirmation",
            "status": p1["status"],
            "evidence": (
                f"cards={p1['cards']}, top1_excess={_fmt(p1['top1_excess'])}, top2_excess={_fmt(p1['top2_excess'])}, "
                f"top1_positive={_fmt(p1['top1_positive'])}, top1_worst={_fmt(p1['top1_worst'])}, "
                f"anchor_match={_fmt(p1['anchor_match'])}, top2_overlap={_fmt(p1['top2_overlap'])}"
            ),
            "next_action": p1["reason"],
        },
        {
            "gate": "rolling_preflight_context",
            "status": str(preflight.get("next_step", "missing")),
            "evidence": (
                f"p0_sample_plan={preflight.get('p0_sample_plan', '')}, p0_dryrun={preflight.get('p0_dryrun', '')}, "
                f"p1_preflight={preflight.get('p1_preflight', '')}"
            ),
            "next_action": "Preflight can allow bounded Flash, but completed latest smokes still control product overclaim risk.",
        },
        {
            "gate": "latest_rolling_product_risk_register",
            "status": overall if risk_logged else "missing_or_malformed",
            "evidence": f"p0_confirmed={p0_confirmed}, p1_confirmed={p1_confirmed}, risk_logged={risk_logged}",
            "next_action": next_action,
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(prefix: str, p0: dict[str, Any], p1: dict[str, Any], preflight: dict[str, Any], gates: pd.DataFrame) -> None:
    safe = safe_prefix(prefix)
    gates_path = REPORT_DIR / f"{safe}_gates.csv"
    summary_path = REPORT_DIR / f"{safe}_summary.csv"
    report_path = REPORT_DIR / f"{safe}.md"
    summary = pd.DataFrame(
        [
            {"section": "P0_latest", **p0},
            {"section": "P1_latest", **p1},
            {"section": "rolling_preflight", **preflight},
        ]
    )
    gates.to_csv(gates_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(gates, summary, gates_path, summary_path), encoding="utf-8")


def render_report(gates: pd.DataFrame, summary: pd.DataFrame, gates_path: Path, summary_path: Path) -> str:
    return "\n".join(
        [
            "# Latest Rolling Product Risk Register v1",
            "",
            "本报告不调用 DeepSeek、不读取密钥、不生成交易指令。它把已完成的 latest rolling smoke 转成产品风险登记，防止把小样本或防守型结果误写成最终泛化成功。",
            "",
            "## Verdict",
            "",
            "- P0 latest micro smoke 当前只能作为风险确认：若出现 0 暴露或样本太小，不得宣称单支盯盘已经完成最新块泛化。",
            "- P1 latest cross-sector smoke 若排序超额为正但样本过小，只能说明 ranker-anchor 方向仍可用，不能替代 rolling new-data confirmation。",
            "- 产品交付继续保留 P0/P1 强黄灯 MVP；broad active-buy 和最终绿灯仍不得宣称完成。",
            "",
            "## Gates",
            "",
            gates.to_markdown(index=False),
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Artifacts",
            "",
            f"- `{gates_path}`",
            f"- `{summary_path}`",
        ]
    ) + "\n"


def _sum(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _bool_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].astype(bool).mean())


def _fmt(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(numeric) else f"{float(numeric):.6f}"


def safe_prefix(prefix: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in prefix)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
