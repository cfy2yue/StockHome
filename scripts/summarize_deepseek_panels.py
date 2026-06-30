from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT = ROOT / "reports" / "date_generalization"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DeepSeek dual-mode panel metrics.")
    parser.add_argument("--prefixes", nargs="+", default=["deepseek_dual_mode", "deepseek_dual_mode_panel1", "deepseek_dual_mode_panel2"])
    args = parser.parse_args()

    metrics, steps, usage = _load(args.prefixes)
    metrics.to_csv(OUTPUT / "deepseek_dual_mode_panel_metrics.csv", index=False, encoding="utf-8-sig")
    steps.to_csv(OUTPUT / "deepseek_dual_mode_panel_step_metrics.csv", index=False, encoding="utf-8-sig")
    usage.to_csv(OUTPUT / "deepseek_dual_mode_panel_usage.csv", index=False, encoding="utf-8-sig")
    aggregate = _aggregate(metrics)
    aggregate.to_csv(OUTPUT / "deepseek_dual_mode_panel_aggregate.csv", index=False, encoding="utf-8-sig")
    diagnostics = _diagnostics(steps)
    diagnostics.to_csv(OUTPUT / "deepseek_dual_mode_panel_diagnostics.csv", index=False, encoding="utf-8-sig")
    _write_report(aggregate, diagnostics, usage, OUTPUT / "deepseek_dual_mode_panel_summary.md")
    print("A股研究Agent")
    print(f"panels: {metrics['panel'].nunique() if not metrics.empty else 0}")
    print(f"decision_cards: {int(metrics['decision_cards'].sum()) if 'decision_cards' in metrics else 0}")
    print(f"wrote: {OUTPUT / 'deepseek_dual_mode_panel_summary.md'}")


def _load(prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    step_rows = []
    usage_rows = []
    for panel, prefix in enumerate(prefixes):
        metric_path = OUTPUT / f"{prefix}_metrics.csv"
        step_path = OUTPUT / f"{prefix}_step_metrics.csv"
        usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
        if metric_path.exists():
            frame = pd.read_csv(metric_path)
            frame.insert(0, "panel", panel)
            frame.insert(1, "prefix", prefix)
            metric_rows.append(frame)
        if step_path.exists():
            frame = pd.read_csv(step_path)
            frame.insert(0, "panel", panel)
            frame.insert(1, "prefix", prefix)
            step_rows.append(frame)
        if usage_path.exists():
            frame = pd.read_csv(usage_path)
            frame.insert(0, "panel", panel)
            frame.insert(1, "prefix", prefix)
            usage_rows.append(frame)
    return (
        pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame(),
        pd.concat(step_rows, ignore_index=True) if step_rows else pd.DataFrame(),
        pd.concat(usage_rows, ignore_index=True) if usage_rows else pd.DataFrame(),
    )


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby("task_mode")
        .agg(
            panels=("panel", "nunique"),
            decision_cards=("decision_cards", "sum"),
            invalid_outputs=("invalid_outputs", "sum"),
            avg_return_mean=("avg_return_20d_exposure", "mean"),
            avg_return_std=("avg_return_20d_exposure", "std"),
            pos_rate_mean=("positive_20d_rate_exposure", "mean"),
            pos_rate_std=("positive_20d_rate_exposure", "std"),
            cash_avg_mean=("cash_adjusted_avg_return_20d", "mean"),
            cash_avg_std=("cash_adjusted_avg_return_20d", "std"),
            cash_pos_mean=("cash_adjusted_positive_20d_rate", "mean"),
            cash_pos_std=("cash_adjusted_positive_20d_rate", "std"),
        )
        .reset_index()
    )


def _diagnostics(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    rows = []
    for (task_mode, valid_block), group in steps.groupby(["task_mode", "valid_block"], sort=True):
        rows.append(
            {
                "task_mode": task_mode,
                "valid_block": valid_block,
                "panels": int(group["panel"].nunique()),
                "cash_adjusted_avg_mean": round(float(pd.to_numeric(group["cash_adjusted_avg_return_20d"], errors="coerce").mean()), 4),
                "cash_adjusted_positive_mean": round(float(pd.to_numeric(group["cash_adjusted_positive_20d_rate"], errors="coerce").mean()), 4),
                "raw_positive_mean": round(float(pd.to_numeric(group["positive_20d_rate_exposure"], errors="coerce").mean()), 4),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["needs_attention"] = (result["cash_adjusted_positive_mean"] < 0.60) | (result["cash_adjusted_avg_mean"] < 0)
    return result


def _write_report(aggregate: pd.DataFrame, diagnostics: pd.DataFrame, usage: pd.DataFrame, path: Path) -> None:
    token_total = int(pd.to_numeric(usage.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not usage.empty else 0
    model_counts = usage.get("model", pd.Series(dtype=str)).value_counts().to_dict() if not usage.empty and "model" in usage else {}
    lines = [
        "# DeepSeek 双模式多 Panel 汇总",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 总览",
        "",
        f"- total_tokens: {token_total}",
        f"- model_counts: {model_counts}",
        "",
        "## 三组 Panel 聚合",
        "",
        _table(aggregate),
        "",
        "## 时间块诊断",
        "",
        _table(diagnostics),
        "",
        "## 下一轮优化方向",
        "",
        "- 组合模式在三组 panel 上 cash-adjusted 均值仍偏弱，应优先优化候选池排序、候选数量和决策频率。",
        "- 单支模式 cash-adjusted 为正且跨 panel 更稳定，可先扩大样本验证盯盘/排雷价值。",
        "- 训练阶段继续使用 deepseek-v4-flash；最终验收或正式用户推理再使用 deepseek-v4-pro。",
        "- 下一轮建议做系统性实验：候选池 TopN、过热 gate、新闻缺失 gate、周二/周五 vs 每周一次决策频率。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")


if __name__ == "__main__":
    main()
