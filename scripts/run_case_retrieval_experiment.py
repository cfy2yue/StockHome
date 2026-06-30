"""Time-safe analogue case retrieval experiment: build library, evaluate RankIC, ablation, regime decay.

Labels are offline-only; never enter evidence pack. Zero network / zero DeepSeek.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_supervised_ranker_experiment import (  # noqa: E402
    BLOCKS,
    FINAL_OOT_BLOCK,
    load_merged_frame,
    per_date_rank_ic,
    summarize_ic,
)
from src.agent_training.analogue_case_retriever import (  # noqa: E402
    DEFAULT_K,
    DEFAULT_RECENT_WINDOW_TD,
    attach_ledger_skill_hints,
    build_case_library,
    retrieve_analogues_for_index,
    run_leakage_self_check,
    score_analogue_features,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402

REPORT_DIR = ROOT / "reports" / "date_generalization"
DESIGN_DOC = REPORT_DIR / "rag_case_retrieval_design.md"
REPORT_PATH = REPORT_DIR / "rag_case_retrieval_experiment_v1.md"
METRICS_CSV = REPORT_DIR / "rag_case_retrieval_block_metrics_v1.csv"
MIN_DAILY_N = 20


def combined_score(df: pd.DataFrame) -> pd.Series:
    rev = df.groupby("date")["reversal_composite"].rank(pct=True, method="average")
    ana = df.groupby("date")["analogue_base_rate"].rank(pct=True, method="average")
    return (rev + ana) / 2.0


def block_rank_ic_table(df: pd.DataFrame, score_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        sub = df[df["time_block"] == block]
        if sub.empty:
            continue
        ic = per_date_rank_ic(sub, score_col)
        sm = summarize_ic(ic)
        rows.append({"time_block": block, "score": score_col, **sm})
    return rows


def regime_decay_warning_eval(df: pd.DataFrame) -> dict[str, Any]:
    work = df.dropna(subset=["regime_decay_signal", "fwd_ret_20d"]).copy()
    if work.empty:
        return {"note": "no regime_decay rows"}
    daily = (
        work.groupby("date", sort=False)
        .agg(
            regime_decay_median=("regime_decay_signal", "median"),
            reversal_ic=("reversal_composite", lambda s: np.nan),
            mean_fwd=("fwd_ret_20d", "mean"),
        )
        .reset_index()
    )
    rev_ic = per_date_rank_ic(work, "reversal_composite")
    daily = daily.merge(
        rev_ic.rename("reversal_ic").reset_index().rename(columns={"index": "date"}),
        on="date",
        how="left",
    )
    work = work.merge(daily[["date", "regime_decay_median"]], on="date", how="left")
    warning = work["regime_decay_signal"] < -2.0
    rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        sub = work[work["time_block"] == block]
        if sub.empty:
            continue
        warn_rate = float(warning.loc[sub.index].mean())
        warn_fwd = float(sub.loc[warning.loc[sub.index], "fwd_ret_20d"].mean()) if warning.loc[sub.index].any() else np.nan
        normal_fwd = float(sub.loc[~warning.loc[sub.index], "fwd_ret_20d"].mean()) if (~warning.loc[sub.index]).any() else np.nan
        rev_ic_block = summarize_ic(rev_ic.reindex(sub["date"].unique()))
        rows.append(
            {
                "time_block": block,
                "warning_rate": round(warn_rate, 4),
                "mean_fwd_when_warning": round(warn_fwd, 4) if pd.notna(warn_fwd) else np.nan,
                "mean_fwd_when_normal": round(normal_fwd, 4) if pd.notna(normal_fwd) else np.nan,
                "fwd_delta_warning_minus_normal": round(warn_fwd - normal_fwd, 4)
                if pd.notna(warn_fwd) and pd.notna(normal_fwd)
                else np.nan,
                "reversal_mean_rank_ic": rev_ic_block["mean_rank_ic"],
            }
        )
    early_h2026 = work[(work["time_block"] == "H2025_2") | (work["time_block"] == FINAL_OOT_BLOCK)]
    h2026_only = work[work["time_block"] == FINAL_OOT_BLOCK]
    return {
        "warning_threshold": -2.0,
        "by_block": rows,
        "H2025_2_warning_rate": round(float(warning.loc[early_h2026.index].mean()), 4) if not early_h2026.empty else np.nan,
        "H2026_1_warning_rate": round(float(warning.loc[h2026_only.index].mean()), 4) if not h2026_only.empty else np.nan,
        "H2026_1_fwd_delta_warning_minus_normal": next(
            (r["fwd_delta_warning_minus_normal"] for r in rows if r["time_block"] == FINAL_OOT_BLOCK),
            np.nan,
        ),
    }


def skill_tag_eval(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in sorted(df["dominant_skill_tag"].dropna().unique()):
        if not tag:
            continue
        sub = df[df["dominant_skill_tag"] == tag]
        rows.append(
            {
                "skill_tag": tag,
                "n_rows": int(len(sub)),
                "mean_fwd_ret_20d": round(float(sub["fwd_ret_20d"].mean()), 4),
                "pos_rate": round(float((sub["fwd_ret_20d"] > 0).mean()), 4),
                "H2026_mean_fwd": round(
                    float(sub.loc[sub["time_block"] == FINAL_OOT_BLOCK, "fwd_ret_20d"].mean()), 4
                )
                if (sub["time_block"] == FINAL_OOT_BLOCK).any()
                else np.nan,
            }
        )
    return sorted(rows, key=lambda r: r["n_rows"], reverse=True)


def coverage_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_dates": int(df["date"].nunique()),
        "analogue_coverage_rate": round(float(df["analogue_base_rate"].notna().mean()), 4),
        "mean_n_candidates": round(float(df["n_candidates"].mean()), 1),
        "blocks": df["time_block"].value_counts().to_dict(),
    }


def render_report(
    *,
    leakage: dict[str, Any],
    coverage: dict[str, Any],
    block_rows: list[dict[str, Any]],
    ablation: dict[str, Any],
    regime: dict[str, Any],
    skills: list[dict[str, Any]],
    sample_retrieval: dict[str, Any],
    verdict: str,
    verdict_reason: str,
) -> str:
    lines = [
        "# RAG Case Retrieval Experiment v1",
        "",
        "研究辅助；标签仅离线评估；不构成投资建议；零 DeepSeek / 零网络。",
        "",
        "## 定位",
        "",
        "时间安全 kNN 案例检索层：为 Agent 审计提供历史类比 base-rate、regime 衰减与 skill 启发，**不是主预测器**。",
        "",
        "## Leakage 自检",
        "",
        f"- 抽检 queries: `{leakage.get('n_checked')}`",
        f"- 情境特征数: `{leakage.get('context_feature_count')}`",
        f"- 未来字段泄漏: `{leakage.get('forbidden_fields_in_context') or 'none'}`",
        f"- 时间安全断言: `{leakage.get('time_safe_assertions_passed')}`",
        "",
        "## 覆盖",
        "",
        f"- 行数: `{coverage['n_rows']}`",
        f"- 决策日: `{coverage['n_dates']}`",
        f"- analogue 有效覆盖率: `{coverage['analogue_coverage_rate']}`",
        f"- 平均候选库大小: `{coverage['mean_n_candidates']}`",
        "",
        "## RankIC（逐块 Spearman，score vs fwd_ret_20d）",
        "",
        "| time_block | score | mean_rank_ic | ic_pos_rate | icir | n_days |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in block_rows:
        lines.append(
            f"| {row['time_block']} | {row['score']} | {row['mean_rank_ic']} | {row['ic_positive_rate']} | {row['icir']} | {row['n_days']} |"
        )

    lines.extend(
        [
            "",
            "## Ablation（reversal vs reversal+analogue）",
            "",
            f"- reversal_composite OOS mean RankIC（不含 H2026）: `{ablation.get('reversal_oos_mean_ic')}`",
            f"- analogue_base_rate OOS mean RankIC（不含 H2026）: `{ablation.get('analogue_oos_mean_ic')}`",
            f"- combined OOS mean RankIC（不含 H2026）: `{ablation.get('combined_oos_mean_ic')}`",
            f"- combined − reversal OOS 增量: `{ablation.get('combined_minus_reversal_oos')}`",
            f"- reversal H2026 mean RankIC: `{ablation.get('reversal_h2026_ic')}`",
            f"- analogue H2026 mean RankIC: `{ablation.get('analogue_h2026_ic')}`",
            f"- combined H2026 mean RankIC: `{ablation.get('combined_h2026_ic')}`",
            f"- combined − reversal H2026 增量: `{ablation.get('combined_minus_reversal_h2026')}`",
            "",
            "## Regime 衰减示警",
            "",
            f"- 示警阈值 regime_decay_signal < `{regime.get('warning_threshold')}`（%/20d，最近类比均值 − 更早类比均值）",
            f"- H2025_2 示警占比: `{regime.get('H2025_2_warning_rate')}`",
            f"- H2026_1 示警占比: `{regime.get('H2026_1_warning_rate')}`",
            f"- H2026_1 示警 vs 正常 fwd_ret_20d 差: `{regime.get('H2026_1_fwd_delta_warning_minus_normal')}`",
            "",
            "| time_block | warning_rate | mean_fwd_warning | mean_fwd_normal | delta | reversal_ic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regime.get("by_block", []):
        lines.append(
            f"| {row['time_block']} | {row['warning_rate']} | {row['mean_fwd_when_warning']} | {row['mean_fwd_when_normal']} | {row['fwd_delta_warning_minus_normal']} | {row['reversal_mean_rank_ic']} |"
        )

    lines.extend(["", "## Skill 启发（dominant_skill_tag vs 后续表现）", ""])
    if skills:
        lines.append("| skill_tag | n_rows | mean_fwd | pos_rate | H2026_mean_fwd |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in skills[:10]:
            lines.append(
                f"| {row['skill_tag']} | {row['n_rows']} | {row['mean_fwd_ret_20d']} | {row['pos_rate']} | {row['H2026_mean_fwd']} |"
            )
    else:
        lines.append("- 无足够 skill tag 样本。")

    lines.extend(
        [
            "",
            "## 样例检索（Agent 层 skill 启发 + ledger）",
            "",
            f"- query: `{sample_retrieval.get('query')}`",
            f"- analogue_mean_fwd_ret_20d: `{sample_retrieval.get('analogue_mean')}`",
            f"- regime_decay_signal: `{sample_retrieval.get('regime_decay')}`",
            f"- skill_tag_counts: `{sample_retrieval.get('skill_tags')}`",
            f"- ledger_skill_hints: `{sample_retrieval.get('ledger_hints')}`",
            "",
            "## 判定",
            "",
            f"**{verdict}** — {verdict_reason}",
            "",
            "## 复现",
            "",
            "```bash",
            "/data/cyx/1030/stock/.conda/stock-agent/bin/python scripts/run_case_retrieval_experiment.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def compute_ablation(df: pd.DataFrame) -> dict[str, Any]:
    df = df.copy()
    df["combined_score"] = combined_score(df)
    oos_blocks = [b for b in BLOCKS if b != FINAL_OOT_BLOCK and b != BLOCKS[0]]
    h2026 = df[df["time_block"] == FINAL_OOT_BLOCK]

    def mean_ic(sub: pd.DataFrame, col: str) -> float:
        ics = [summarize_ic(per_date_rank_ic(sub[sub["time_block"] == b], col))["mean_rank_ic"] for b in oos_blocks]
        vals = [v for v in ics if pd.notna(v)]
        return round(float(np.mean(vals)), 4) if vals else np.nan

    rev_oos = mean_ic(df, "reversal_composite")
    ana_oos = mean_ic(df, "analogue_base_rate")
    comb_oos = mean_ic(df, "combined_score")
    rev_h = summarize_ic(per_date_rank_ic(h2026, "reversal_composite"))["mean_rank_ic"]
    ana_h = summarize_ic(per_date_rank_ic(h2026, "analogue_base_rate"))["mean_rank_ic"]
    comb_h = summarize_ic(per_date_rank_ic(h2026, "combined_score"))["mean_rank_ic"]
    return {
        "reversal_oos_mean_ic": rev_oos,
        "analogue_oos_mean_ic": ana_oos,
        "combined_oos_mean_ic": comb_oos,
        "combined_minus_reversal_oos": round(comb_oos - rev_oos, 4) if pd.notna(comb_oos) and pd.notna(rev_oos) else np.nan,
        "reversal_h2026_ic": rev_h,
        "analogue_h2026_ic": ana_h,
        "combined_h2026_ic": comb_h,
        "combined_minus_reversal_h2026": round(comb_h - rev_h, 4) if pd.notna(comb_h) and pd.notna(rev_h) else np.nan,
    }


def write_design_doc() -> None:
    content = """# RAG Case Retrieval Design（时间安全案例检索）

研究辅助；标签仅离线评估；不构成投资建议。

## 1. 定位

RAG / case-based reasoning **不是主预测器**，而是 Agent 审计层的辅助：

| 输出 | 用途 |
|---|---|
| (a) 相似簇已兑现 base-rate | 历史类比胜率/均值/离散，供反证与 base-rate 校准 |
| (b) book skill / 策略启发 | 类比簇内 skill tag 频率 + ledger 文本检索 |
| (c) regime 衰减信号 | 近 3 月类比 vs 更早类比的表现差，侦测模式失效 |

与 ML / Agent / gate 分工：

- **portfolio_ranker (ML)**：截面排序主预测
- **date_regime_gate**：全市场广度/波动，决定是否降暴露
- **analogue RAG**：个股情境下的历史类比 base-rate + 衰减示警 + skill 清单
- **Agent**：解释/反证/冲突，不猜阈值

## 2. 情境向量（决策时点已知）

来源：`daily_kline_multiscale_features` + `corr_peer_kline_features` + `tushare_industry_region_peer_features`。

| 特征 | 说明 |
|---|---|
| reversal_composite | −mean(zscore[kline_return_20d, kline_return_60d, corr_peer_avg_return_20d]) |
| kline_return_20d/60d | 先按日 zscore，负 IC 特征符号翻转 |
| kline_drawdown_20d/60d | 回撤 |
| kline_volatility_ratio_20_60 | 波动 |
| kline_range_position_60d | 区间位置 |
| corr_peer_relative_return_20d | 同行相对 |
| tushare_industry_relative_return_20d | 行业相对 |
| tushare_industry top-12 one-hot | 行业分桶 |

**降级**：本地无 `daily_basic`，价值/规模未纳入（已记录）。

**禁用黑名单**：`fwd_ret_*`, `return_*d`, `top_decile_flag`, `rank_pct_*`, `gt_*` 等（见 `analogue_case_retriever.FUTURE_FIELD_BLACKLIST`）。

## 3. 时间安全规则

1. **候选库**：查询日 T 只含 `maturity_date = decision_date + 20 交易日 ≤ T` 的历史案例（结果已兑现）。
2. **检索特征**：仅情境向量；标签 `fwd_ret_20d` 只用于计算类比簇 base-rate，不进入 query 向量。
3. **自检**：`assert_retrieval_time_safe` + `assert_context_columns_safe`；实验脚本抽检 200 点。
4. **walk-forward**：k=20、recent_window=63 交易日固定，不用 OOT 调参。

## 4. 检索方法

- 透明 **kNN cosine**（L2 归一化后的情境向量）。
- **行业分桶优先**：同 `tushare_industry` 已兑现案例；不足 k 时回退全局池。
- 排除自身 (date, code)。
- skill tag：规则化启发（reversal_pullback / deep_drawdown / weak_peer / industry_laggard / high_volatility）。
- ledger skill：`case_memory_retriever.retrieve_cases` 文本检索，供 Agent prompt，不进 ML 特征。

## 5. 评估口径

- **RankIC**：analogue_base_rate vs fwd_ret_20d，逐日 Spearman，按 time_block 汇总。
- **Ablation**：reversal_composite vs combined(reversal rank + analogue rank)。
- **Regime 衰减**：regime_decay_signal = recent_mean − older_mean；阈值 −2%/20d 示警。
- **Skill 稳定性**：dominant_skill_tag 条件均值 fwd_ret_20d。

## 6. 实现

- 核心：`src/agent_training/analogue_case_retriever.py`
- 实验：`scripts/run_case_retrieval_experiment.py`
- 复用：`case_memory_retriever.py`（ledger 文本）、`memory_context.py`（compact memory 边界）

## 7. 与现有 memory 关系

- **strategy_experience_ledger 等**：记录假设/反证/下一步（文本 RAG，已有）。
- **本层**：市场情境 kNN + 已兑现 outcome base-rate（数值 RAG，新增）。
- 二者互补：数值类比给 base-rate；ledger 给可解释 checklist。
"""
    DESIGN_DOC.parent.mkdir(parents=True, exist_ok=True)
    if DESIGN_DOC.exists():
        backup = DESIGN_DOC.with_suffix(".md.bak")
        if not backup.exists():
            DESIGN_DOC.replace(backup)
    DESIGN_DOC.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-safe analogue case retrieval experiment.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--recent-window-td", type=int, default=DEFAULT_RECENT_WINDOW_TD)
    args = parser.parse_args()

    write_design_doc()

    merged = load_merged_frame()
    library = build_case_library(merged)
    leakage = run_leakage_self_check(library)
    scored = score_analogue_features(library, k=args.k, recent_window_td=args.recent_window_td)
    scored["combined_score"] = combined_score(scored)

    block_rows: list[dict[str, Any]] = []
    for score_col in ["analogue_base_rate", "reversal_composite", "combined_score"]:
        block_rows.extend(block_rank_ic_table(scored, score_col))

    ablation = compute_ablation(scored)
    regime = regime_decay_warning_eval(scored)
    skills = skill_tag_eval(scored)
    coverage = coverage_stats(scored)

    # sample retrieval for report
    sample_sub = scored[scored["time_block"] == FINAL_OOT_BLOCK]
    if sample_sub.empty:
        sample_sub = scored.tail(1)
    row = sample_sub.iloc[0]
    lib_match = library.frame.index[
        (library.frame["date"].astype(str) == str(row["date"]))
        & (library.frame["code"].astype(str) == str(row["code"]))
    ]
    lib_idx = int(lib_match[0]) if len(lib_match) else 0
    sample_res = attach_ledger_skill_hints(retrieve_analogues_for_index(library, lib_idx))
    sample_retrieval = {
        "query": f"{sample_res.query_code}@{sample_res.query_date}",
        "analogue_mean": round(sample_res.analogue_mean_fwd_ret_20d, 4),
        "regime_decay": round(sample_res.regime_decay_signal, 4) if pd.notna(sample_res.regime_decay_signal) else None,
        "skill_tags": sample_res.skill_tag_counts,
        "ledger_hints": sample_res.ledger_skill_hints,
    }

    ana_oos = ablation["analogue_oos_mean_ic"]
    ana_h = ablation["analogue_h2026_ic"]
    comb_delta_oos = ablation["combined_minus_reversal_oos"]
    comb_delta_h = ablation["combined_minus_reversal_h2026"]
    h2026_warn_delta = regime.get("H2026_1_fwd_delta_warning_minus_normal")

    if pd.notna(comb_delta_oos) and comb_delta_oos >= 0.01 and pd.notna(ana_oos) and ana_oos > 0.02:
        verdict = "🟢 有增量值得接 Agent"
        verdict_reason = "OOS analogue RankIC 为正且 combined 对 reversal 有正交增量。"
    elif pd.notna(ana_oos) and ana_oos > 0 and (pd.isna(comb_delta_oos) or comb_delta_oos >= 0):
        verdict = "🟡 仅作上下文"
        verdict_reason = "analogue 有一定 RankIC 但增量有限；regime/skill 可作审计上下文，不升权 ML。"
    else:
        verdict = "🔴 无用"
        verdict_reason = "analogue base-rate RankIC 弱或与 reversal 高度同源且无 OOT 增量。"

    if pd.notna(ana_h) and ana_h <= 0 and pd.notna(h2026_warn_delta) and h2026_warn_delta < 0:
        verdict_reason += " H2026 analogue IC≤0 但 regime 示警与更差 fwd 一致，可作弃权参考。"

    report = render_report(
        leakage=leakage,
        coverage=coverage,
        block_rows=block_rows,
        ablation=ablation,
        regime=regime,
        skills=skills,
        sample_retrieval=sample_retrieval,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        backup = REPORT_PATH.with_suffix(".md.bak")
        if not backup.exists():
            REPORT_PATH.replace(backup)
    REPORT_PATH.write_text(report, encoding="utf-8")
    pd.DataFrame(block_rows).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    print("A股研究Agent")
    print(f"rows={len(scored)} leakage_checked={leakage['n_checked']}")
    print(f"analogue_oos_ic={ablation['analogue_oos_mean_ic']} analogue_h2026_ic={ablation['analogue_h2026_ic']}")
    print(f"combined_minus_reversal_oos={ablation['combined_minus_reversal_oos']} combined_minus_reversal_h2026={ablation['combined_minus_reversal_h2026']}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
