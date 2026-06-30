from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_INPUT_PREFIX = "p0_transfer_analog_rag_v1_panel36_preflight"
DEFAULT_OUTPUT_PREFIX = "p0_transfer_analog_rag_panel36_preflight_gate_v1"


KEYS = ["frequency", "variant", "analog_id", "gate_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fresh-panel preflight gates for P0 transfer analog/RAG candidates.")
    parser.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()

    input_prefix = safe_prefix(args.input_prefix)
    output_prefix = safe_prefix(args.output_prefix)
    summary_path = REPORT_DIR / f"{input_prefix}_summary.csv"
    panel_path = REPORT_DIR / f"{input_prefix}_h2026_panel_detail.csv"
    summary = pd.read_csv(summary_path)
    panels = pd.read_csv(panel_path)

    candidate_summary = summary[
        summary["promotion_status"]
        .astype(str)
        .str.contains("green|yellow|time_support_insufficient", regex=True, na=False)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in candidate_summary.iterrows():
        panel_slice = panels.copy()
        for key in KEYS:
            panel_slice = panel_slice[panel_slice[key].astype(str).eq(str(row.get(key)))]
        if panel_slice.empty:
            continue
        metrics = panel_metrics(panel_slice)
        out = {
            key: row.get(key)
            for key in KEYS
        }
        out.update(
            {
                "local_promotion_status": row.get("promotion_status"),
                "prior_blocks": int_or_none(row.get("prior_blocks")),
                "prior_evaluable_blocks": int_or_none(row.get("prior_evaluable_blocks")),
                "prior_selected_rows_mean": num(row.get("prior_selected_rows_mean")),
                "prior_evaluable_selected_rows_mean": num(row.get("prior_evaluable_selected_rows_mean")),
                "prior_delta_pos_hit": num(row.get("prior_delta_pos_hit")),
                "prior_delta_avg_hit": num(row.get("prior_delta_avg_hit")),
                "prior_evaluable_delta_pos_hit": num(row.get("prior_evaluable_delta_pos_hit")),
                "prior_evaluable_delta_avg_hit": num(row.get("prior_evaluable_delta_avg_hit")),
                "h2026_transfer_rows": int_or_none(row.get("h2026_transfer_rows")),
                "h2026_selected_rows": int_or_none(row.get("h2026_selected_rows")),
                "h2026_selected_pos20": num(row.get("h2026_selected_pos20")),
                "h2026_selected_avg20_pp": num(row.get("h2026_selected_avg20")),
                "h2026_selected_loss_gt5": num(row.get("h2026_selected_loss_gt5")),
                "h2026_delta_pos_vs_transfer": num(row.get("h2026_delta_pos_vs_transfer")),
                "h2026_delta_avg_pp_vs_transfer": num(row.get("h2026_delta_avg_vs_transfer")),
                "rank_score": num(row.get("rank_score")),
            }
        )
        out.update(metrics)
        out["flash_preflight_status"] = flash_preflight_status(row, metrics)
        out["pro_status"] = "not_ready_until_flash_paired_lift_and_fresh_panel_pass"
        out["main_read"] = main_read(out)
        rows.append(out)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["flash_preflight_status", "panel_selected_pos20_mean", "panel_delta_pos_vs_transfer_mean"],
            ascending=[True, False, False],
        )
    status_summary = build_status_summary(result)
    shortlist = result[result["flash_preflight_status"].eq("flash_candidate_strong_not_pro_ready")].copy()

    outputs = {
        "gate": REPORT_DIR / f"{output_prefix}.csv",
        "status": REPORT_DIR / f"{output_prefix}_status_summary.csv",
        "shortlist": REPORT_DIR / f"{output_prefix}_flash_shortlist.csv",
        "report": REPORT_DIR / f"{output_prefix}.md",
    }
    result.to_csv(outputs["gate"], index=False, encoding="utf-8-sig")
    status_summary.to_csv(outputs["status"], index=False, encoding="utf-8-sig")
    shortlist.to_csv(outputs["shortlist"], index=False, encoding="utf-8-sig")
    outputs["report"].write_text(render_report(result, status_summary, shortlist, inputs=[summary_path, panel_path], outputs=outputs), encoding="utf-8")

    print("A股研究Agent")
    print(f"candidate_rows={len(result)} strong_flash_candidates={len(shortlist)}")
    print(f"report={outputs['report']}")


def panel_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {"panels": int(frame["panel_seed"].nunique()) if "panel_seed" in frame else len(frame)}
    for column in [
        "selected_rows",
        "selected_pos20",
        "selected_avg20",
        "selected_loss_gt5",
        "delta_pos_vs_transfer",
        "delta_avg_vs_transfer",
    ]:
        values = pd.to_numeric(frame.get(column), errors="coerce").dropna()
        prefix = f"panel_{column}"
        if values.empty:
            metrics[f"{prefix}_mean"] = None
            metrics[f"{prefix}_std"] = None
            metrics[f"{prefix}_min"] = None
            metrics[f"{prefix}_p10"] = None
            metrics[f"{prefix}_p90"] = None
            metrics[f"{prefix}_max"] = None
            continue
        metrics[f"{prefix}_mean"] = round(float(values.mean()), 6)
        metrics[f"{prefix}_std"] = round(float(values.std() if len(values) > 1 else 0.0), 6)
        metrics[f"{prefix}_min"] = round(float(values.min()), 6)
        metrics[f"{prefix}_p10"] = round(float(values.quantile(0.10)), 6)
        metrics[f"{prefix}_p90"] = round(float(values.quantile(0.90)), 6)
        metrics[f"{prefix}_max"] = round(float(values.max()), 6)
    return metrics


def flash_preflight_status(row: pd.Series, metrics: dict[str, Any]) -> str:
    local_status = str(row.get("promotion_status"))
    prior_evaluable_blocks = safe_float(row.get("prior_evaluable_blocks"))
    prior_eval_pos_hit = safe_float(row.get("prior_evaluable_delta_pos_hit"))
    prior_eval_avg_hit = safe_float(row.get("prior_evaluable_delta_avg_hit"))
    panels = safe_float(metrics.get("panels"))
    selected_rows = safe_float(metrics.get("panel_selected_rows_mean"))
    pos_mean = safe_float(metrics.get("panel_selected_pos20_mean"))
    pos_p10 = safe_float(metrics.get("panel_selected_pos20_p10"))
    avg_mean = safe_float(metrics.get("panel_selected_avg20_mean"))
    avg_p10 = safe_float(metrics.get("panel_selected_avg20_p10"))
    loss_mean = safe_float(metrics.get("panel_selected_loss_gt5_mean"))
    loss_p90 = safe_float(metrics.get("panel_selected_loss_gt5_p90"))
    delta_pos_mean = safe_float(metrics.get("panel_delta_pos_vs_transfer_mean"))
    delta_pos_p10 = safe_float(metrics.get("panel_delta_pos_vs_transfer_p10"))
    delta_avg_p10 = safe_float(metrics.get("panel_delta_avg_vs_transfer_p10"))

    if (
        "green" in local_status
        and prior_evaluable_blocks >= 2
        and prior_eval_pos_hit >= 0.75
        and prior_eval_avg_hit >= 0.75
        and panels >= 36
        and selected_rows >= 30
        and pos_mean >= 0.75
        and pos_p10 >= 0.75
        and avg_mean >= 5.0
        and avg_p10 >= 4.0
        and loss_mean <= 0.10
        and loss_p90 <= 0.08
        and delta_pos_mean >= 0.03
        and delta_pos_p10 > 0
        and delta_avg_p10 > 0
    ):
        return "flash_candidate_strong_not_pro_ready"
    if "time_support_insufficient" in local_status:
        return "hold_before_flash_time_generalization_insufficient"
    if "green" in local_status or "yellow" in local_status:
        if prior_evaluable_blocks < 2:
            return "hold_before_flash_time_generalization_insufficient"
        if panels >= 36 and selected_rows >= 25 and pos_mean >= 0.70 and avg_mean >= 4.0 and delta_pos_mean > 0:
            return "flash_candidate_observe_needs_tighter_panel_or_prior"
        return "reject_before_flash_panel_weak_or_sparse"
    return "not_a_promoted_candidate"


def build_status_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["flash_preflight_status", "rows", "pos20_mean", "avg20_mean", "delta_pos_mean"])
    grouped = frame.groupby("flash_preflight_status", dropna=False)
    rows = []
    for status, group in grouped:
        rows.append(
            {
                "flash_preflight_status": status,
                "rows": len(group),
                "panel_pos20_mean": mean(group.get("panel_selected_pos20_mean")),
                "panel_avg20_pp_mean": mean(group.get("panel_selected_avg20_mean")),
                "panel_delta_pos_vs_transfer_mean": mean(group.get("panel_delta_pos_vs_transfer_mean")),
                "min_panel_pos20_p10": min_value(group.get("panel_selected_pos20_p10")),
                "max_panel_loss_gt5_p90": max_value(group.get("panel_selected_loss_gt5_p90")),
            }
        )
    return pd.DataFrame(rows).sort_values("flash_preflight_status").round(6)


def render_report(
    result: pd.DataFrame,
    status_summary: pd.DataFrame,
    shortlist: pd.DataFrame,
    *,
    inputs: list[Path],
    outputs: dict[str, Path],
) -> str:
    lines = [
        "# P0 Transfer Analog/RAG 36-Panel Preflight Gate v1",
        "",
        "本报告是本地回测预检，不调用 DeepSeek，不读取或输出 API key/token。它用于判断最新 `analog/RAG + K线量化工具` 候选是否值得进入 Flash/Pro 真模型消融。",
        "",
        "## Main Verdict",
        "",
    ]
    if shortlist.empty:
        lines.append("- 没有候选通过强 Flash 预检；在 DS 恢复前不应排队 Flash/Pro。")
    else:
        lines.append(
            f"- 有 {len(shortlist)} 个候选通过 `flash_candidate_strong_not_pro_ready`：可以在 DS 恢复后优先跑 Flash paired on/off。"
        )
        lines.append("- 这些候选仍不是 Pro-ready，也不能上线；只有 Flash paired lift 和 fresh panel 继续通过后，才进入 Pro 同样本确认。")
    lines.extend(
        [
            "- 预检强项要求：36 panels、panel pos20 mean/p10 过线、平均收益过线、尾部损失低、相对 transfer reference 的 delta_pos/delta_avg 在低分位仍为正。",
            "- 任何 `observe` 或 `yellow` 候选都只能作为解释/候选池材料，不能直接消耗 Pro token。",
            "",
            "## Status Summary",
            "",
            markdown_table(status_summary),
            "",
            "## Flash Shortlist",
            "",
            markdown_table(
                shortlist,
                [
                    "frequency",
                    "variant",
                    "analog_id",
                    "gate_id",
                "local_promotion_status",
                "prior_delta_pos_hit",
                "prior_evaluable_blocks",
                "prior_evaluable_delta_pos_hit",
                "panel_selected_rows_mean",
                    "panel_selected_pos20_mean",
                    "panel_selected_pos20_p10",
                    "panel_selected_avg20_mean",
                    "panel_selected_avg20_p10",
                    "panel_selected_loss_gt5_mean",
                    "panel_selected_loss_gt5_p90",
                    "panel_delta_pos_vs_transfer_mean",
                    "panel_delta_pos_vs_transfer_p10",
                    "panel_delta_avg_vs_transfer_p10",
                    "flash_preflight_status",
                    "main_read",
                ],
            ),
            "",
            "## Candidate Gate Rows",
            "",
            markdown_table(result.head(80)),
            "",
            "## Inputs",
            "",
            *[f"- `{path}`" for path in inputs],
            "",
            "## Outputs",
            "",
            *[f"- `{path}`" for path in outputs.values()],
            "",
            "## Next Action",
            "",
            "1. DS 恢复后先跑 Flash：`full_agent / no_analogue_case_context / no_quant_tools / quant_tool_summary_only / no_chip_context / no_news / no_financial_report / no_peer / no_bookskill / python_only`。",
            "2. Flash 通过后再做 fresh panel 和 Pro 同样本确认；Flash 未通过则停止 Pro。",
            "3. 报告必须继续标注 baseline、三次采样均值/std、bad_raise、missed_positive、active exposure、future leak、secret scan 和 token 成本。",
        ]
    )
    return "\n".join(lines)


def main_read(row: dict[str, Any]) -> str:
    status = str(row.get("flash_preflight_status"))
    if status == "flash_candidate_strong_not_pro_ready":
        return "local 36-panel preflight is strong; run Flash first, Pro only after paired Flash lift"
    if status == "hold_before_flash_time_generalization_insufficient":
        return "latest block/panel metrics are bright but prior evaluable blocks are insufficient; hold DS tokens and seek more time support"
    if status == "flash_candidate_observe_needs_tighter_panel_or_prior":
        return "positive but not enough for automatic Flash priority; keep as backup or diagnostic"
    return "do not spend DS tokens before improving prior support, coverage, or panel stability"


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_empty_"
    work = frame.copy()
    if columns is not None:
        work = work[[col for col in columns if col in work.columns]]
    work = work.fillna("")
    cols = list(work.columns)
    rows = work.astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_")


def num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return round(parsed, 6)


def int_or_none(value: Any) -> int | None:
    parsed = num(value)
    if parsed is None:
        return None
    return int(parsed)


def safe_float(value: Any) -> float:
    parsed = num(value)
    return -1e9 if parsed is None else parsed


def mean(values: Any) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.mean()), 6)


def min_value(values: Any) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.min()), 6)


def max_value(values: Any) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.max()), 6)


if __name__ == "__main__":
    main()
