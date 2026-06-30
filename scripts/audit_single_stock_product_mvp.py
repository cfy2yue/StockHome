"""Audit single-stock product MVP: rev+chip_core tier grading + enhanced risk flags.

Opportunity: threshold tiers (强/中/弱), not cross-section TopK.
Risk: rule flags baseline vs enhanced (chip loosen / chase-high etc.).
Labels offline-only; never enter evidence.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lightweight_ml_channel_experiment import _rolling_split  # noqa: E402
from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402
from src.agent_training.single_stock_review import (  # noqa: E402
    FINAL_OOT_BLOCK,
    OPPORTUNITY_TIERS,
    POLICY_VERSION,
    block_base_metrics,
    build_review_frame,
    calibrate_tier_thresholds,
    compute_risk_flags,
    example_cards_for_block,
    review_row,
    risk_recall_precision,
    select_risk_flagged,
    side_metrics,
)

REPORT_PATH = ROOT / "reports" / "date_generalization" / "single_stock_product_mvp_v1.md"
CSV_PATH = ROOT / "reports" / "date_generalization" / "single_stock_product_mvp_v1.csv"
BLOCKS = list(TIME_BLOCKS.keys())
TARGET_BLOCKS = BLOCKS[1:]
MIN_TRAIN_ROWS = 500
MIN_VALID_ROWS = 200
MIN_TARGET_ROWS = 200


def traffic_light_opp(h2026_delta: float, oos_positive_frac: float) -> str:
    if pd.notna(h2026_delta) and h2026_delta >= 0.03 and oos_positive_frac >= 0.75:
        return "🟢"
    if pd.notna(h2026_delta) and h2026_delta > 0:
        return "🟡"
    return "🔴"


def traffic_light_risk(h2026_recall: float, h2026_delta_loss: float, improved: bool) -> str:
    if pd.notna(h2026_recall) and h2026_recall >= 0.35 and pd.notna(h2026_delta_loss) and h2026_delta_loss >= 0:
        return "🟢"
    if improved or (pd.notna(h2026_recall) and h2026_recall >= 0.20):
        return "🟡"
    return "🔴"


def product_verdict(opp_light: str, risk_light: str) -> str:
    if opp_light == "🟢" and risk_light in ("🟢", "🟡"):
        return "🟢 单支 MVP 可作为 H2026 主交付（机会分级稳 + 排雷可用）"
    if opp_light in ("🟢", "🟡") and risk_light == "🟡":
        return "🟡 机会侧可交付；排雷作复核助手，recall 待继续抬升"
    if opp_light == "🟢":
        return "🟡 仅机会侧过线，排雷未达标"
    return "🔴 单支 MVP 未达可交付线"


def render_report(
    rows: pd.DataFrame,
    examples: list[dict[str, Any]],
    notes: list[str],
    anomalies: list[str],
    opp_light: str,
    risk_light: str,
    product: str,
    before_h2026: dict[str, float],
    after_h2026: dict[str, float],
    h2026_delta: float,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 单支盯盘/排雷产品 MVP v1",
        "",
        f"> 生成时间：{ts} | policy={POLICY_VERSION} | Final OOT = `{FINAL_OOT_BLOCK}`",
        f"> 信号：rev+chip_core 等权 z 分级阈值（非截面 TopK）",
        "",
        "研究辅助，不构成投资建议。标签仅离线评估，不进 evidence。",
        "",
        "## 1. 方法",
        "",
        "- **机会侧**：`rev+chip_core` 分数 → 验证集分位阈值 → 强/中/弱 分级（非 TopK）。",
        "- **排雷侧**：筹码/价量规则旗标集；`flag_count≥2` 或关键旗标 → 暂时剔除倾向。",
        "- **弃权**：数据缺失 / regime 衰减（验证集信号均值过低）→ 信息不足或放入观察。",
        "- **H2026_1**：仅 target，不参与 train/valid/阈值校准。",
        "",
        f"缓存备注：{' ; '.join(notes)}",
        "",
        "## 2. 机会分级命中（各级 vs base）",
        "",
        "| block | tier | n | sel_pos | base_pos | Δpos | sel_mean | Δmean | active |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    opp = rows[rows["side"].str.startswith("opp_tier_")].sort_values(["target_block", "tier"])
    for _, r in opp.iterrows():
        lines.append(
            f"| {r['target_block']} | {r['tier']} | {int(r['sample_count'])} | "
            f"{r['positive_20d_rate']:.4f} | {r['base_pos']:.4f} | {r['delta_pos_vs_base']:+.4f} | "
            f"{r['avg_return_20d']:.4f} | {r['delta_mean_vs_base']:+.4f} | {r.get('active_exposure', np.nan):.4f} |"
        )

    strong_h = opp[(opp["target_block"] == FINAL_OOT_BLOCK) & (opp["tier"] == "强")]
    if not strong_h.empty:
        r = strong_h.iloc[0]
        lines.extend([
            "",
            f"**H2026 强分级 Δpos**：{r['delta_pos_vs_base']:+.4f}（base_pos={r['base_pos']:.4f}）",
        ])

    combined = rows[rows["side"] == "opp_score_threshold"]
    if not combined.empty:
        lines.extend(["", "### 强阈值池（验证集择优阈值，非 TopK）", ""])
        lines.append("| block | n | sel_pos | Δpos | sel_mean | Δmean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in combined.sort_values("target_block").iterrows():
            lines.append(
                f"| {r['target_block']} | {int(r['sample_count'])} | {r['positive_20d_rate']:.4f} | "
                f"{r['delta_pos_vs_base']:+.4f} | {r['avg_return_20d']:.4f} | {r['delta_mean_vs_base']:+.4f} |"
            )

    lines.extend(["", f"**机会侧判定**：{opp_light}", ""])
    lines.extend([
        "### 与 v1 additive_bin 对照（参考，非本产品信号）",
        "",
        "v1 `opportunity_model`（additive_bin ML）H2026 Δpos=**+5.25pp**；本产品 rev+chip_core 强阈值池 H2026 Δpos="
        f"**{h2026_delta:+.4f}**（chip 交集宇宙，信号已切换为 rev+chip_core）。",
        "",
    ])

    lines.extend([
        "## 3. 排雷 recall/precision（H2026 重点，改善前后）",
        "",
        "| variant | block | recall | precision | sel_loss | Δloss | n_flagged |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    risk = rows[rows["side"].str.startswith("risk_")].sort_values(["variant", "target_block"])
    for _, r in risk.iterrows():
        lines.append(
            f"| {r['variant']} | {r['target_block']} | {r.get('risk_recall', np.nan):.4f} | "
            f"{r.get('risk_precision', np.nan):.4f} | {r['loss_gt5_rate']:.4f} | "
            f"{r['delta_loss_vs_base']:+.4f} | {int(r['sample_count'])} |"
        )

    lines.extend([
        "",
        "### H2026 排雷对比摘要",
        f"- **改善前（baseline exclude）**：recall={before_h2026.get('risk_recall', 'NA')} precision={before_h2026.get('risk_precision', 'NA')}",
        f"- **改善后（enhanced review flag≥1）**：recall={after_h2026.get('risk_recall', 'NA')} precision={after_h2026.get('risk_precision', 'NA')}",
        "",
        f"**排雷侧判定**：{risk_light}",
        "",
        "## 4. 示例决策卡（脱敏，时间安全）",
        "",
        "仅展示四选一研究分级；不含未来收益字段。",
        "",
    ])
    for i, card in enumerate(examples, 1):
        lines.extend([
            f"### 示例 {i}：{card.get('research_grade')}",
            "",
            f"- 决策日：{card.get('decision_date')} | 代码：{card.get('code')} | 名称：{card.get('name')}",
            f"- 机会分级：{card.get('opportunity_tier')} | rev+chip_core={card.get('rev_chip_core_score')}",
            f"- 排雷旗标（{card.get('risk_flag_count')}）：{card.get('counter_evidence')}",
            f"- **研究分级**：{card.get('research_grade')} | 模拟动作：{card.get('simulated_action')} | 权重={card.get('simulated_weight_change')}",
            f"- 置信度：{card.get('confidence_level')} | 摘要：{card.get('final_agent_reasoning_summary')}",
            "",
        ])

    lines.extend(["## 5. 产品可交付判定", "", f"**综合**：{product}", ""])
    if anomalies:
        lines.extend(["## 6. 异常与降级", ""])
        for a in anomalies:
            lines.append(f"- {a}")
        lines.append("")
    lines.extend([
        "## 引用",
        "- `src/agent_training/single_stock_review.py`",
        "- `scripts/audit_single_stock_review_quality.py`",
        "- 标签：`data/date_generalization_cache/market_5000/task_labels_v1.csv`",
    ])
    return "\n".join(lines)


def main() -> None:
    print("A股研究Agent")
    anomalies: list[str] = []
    merged, notes = build_review_frame()
    detail_rows: list[dict[str, Any]] = []
    example_cards: list[dict[str, Any]] = []

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(merged, target_block)
        if (
            len(train_base) < MIN_TRAIN_ROWS
            or len(validation) < MIN_VALID_ROWS
            or len(target) < MIN_TARGET_ROWS
        ):
            anomalies.append(
                f"skip {target_block}: train={len(train_base)} valid={len(validation)} target={len(target)}"
            )
            continue

        base = block_base_metrics(target)
        tier_thr = calibrate_tier_thresholds(validation)
        train_blocks = "+".join([b for b in BLOCKS if b != target_block and b != validation["time_block"].iloc[0]])

        target = target.copy()
        target["opportunity_tier"] = target["rev_chip_core_score"].apply(
            lambda s: tier_thr.tier_for_score(float(s) if pd.notna(s) else float("nan"))
        )

        pool_n = max(len(target), 1)
        for tier in OPPORTUNITY_TIERS:
            if tier == "无":
                continue
            sel = target[target["opportunity_tier"] == tier]
            m = side_metrics(sel, base)
            detail_rows.append({
                "target_block": target_block,
                "side": f"opp_tier_{tier}",
                "tier": tier,
                "variant": "rev_chip_core_threshold",
                **base,
                **m,
                "active_exposure": round(len(sel) / pool_n, 4),
            })

        sm = target[target["opportunity_tier"].isin(["强", "中"])]
        m_sm = side_metrics(sm, base)
        detail_rows.append({
            "target_block": target_block,
            "side": "opp_tier_strong_plus_medium",
            "tier": "强+中",
            "variant": "rev_chip_core_threshold",
            **base,
            **m_sm,
            "active_exposure": round(len(sm) / pool_n, 4),
        })

        thr_sel = target[target["rev_chip_core_score"] >= tier_thr.strong]
        m_thr = side_metrics(thr_sel, base)
        detail_rows.append({
            "target_block": target_block,
            "side": "opp_score_threshold",
            "tier": "强阈值池",
            "variant": "rev_chip_core_validation_threshold",
            "threshold": round(tier_thr.strong, 6),
            **base,
            **m_thr,
            "active_exposure": round(len(thr_sel) / pool_n, 4),
        })

        # risk baseline vs enhanced on same target frame
        tgt_base_flags = compute_risk_flags(target, enhanced=False)
        tgt_enh = compute_risk_flags(target, enhanced=True)

        for variant, frame, enhanced, mode in (
            ("baseline_flags", tgt_base_flags, False, "exclude"),
            ("enhanced_review", tgt_enh, True, "review"),
            ("enhanced_exclude", tgt_enh, True, "exclude"),
        ):
            flagged = select_risk_flagged(frame, enhanced=enhanced, mode=mode)
            m = side_metrics(flagged, base)
            rp = risk_recall_precision(target, flagged)
            detail_rows.append({
                "target_block": target_block,
                "side": f"risk_{variant}",
                "variant": variant,
                "tier": "",
                **base,
                **m,
                **rp,
            })

        if target_block == FINAL_OOT_BLOCK:
            leak = set(train_base["time_block"].unique()) | set(validation["time_block"].unique())
            if FINAL_OOT_BLOCK in leak:
                anomalies.append(f"LEAK: {FINAL_OOT_BLOCK} in train/valid {leak}")
            example_cards = example_cards_for_block(
                target, tier_thr, valid_block=validation["time_block"].iloc[0], train_blocks=train_blocks,
            )

    rows = pd.DataFrame(detail_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(CSV_PATH, index=False)

    opp_combined = rows[rows["side"] == "opp_score_threshold"]
    h2026_opp = opp_combined[opp_combined["target_block"] == FINAL_OOT_BLOCK]
    h2026_delta = float(h2026_opp.iloc[0]["delta_pos_vs_base"]) if not h2026_opp.empty else np.nan
    oos_deltas = opp_combined[opp_combined["target_block"] != FINAL_OOT_BLOCK]["delta_pos_vs_base"].dropna()
    oos_hit = float((oos_deltas > 0).mean()) if len(oos_deltas) else 0.0
    opp_light = traffic_light_opp(h2026_delta, oos_hit)

    before_row = rows[(rows["side"] == "risk_baseline_flags") & (rows["target_block"] == FINAL_OOT_BLOCK)]
    after_row = rows[(rows["side"] == "risk_enhanced_review") & (rows["target_block"] == FINAL_OOT_BLOCK)]
    before_h2026 = before_row.iloc[0].to_dict() if not before_row.empty else {}
    after_h2026 = after_row.iloc[0].to_dict() if not after_row.empty else {}
    recall_before = before_h2026.get("risk_recall", np.nan)
    recall_after = after_h2026.get("risk_recall", np.nan)
    improved = pd.notna(recall_before) and pd.notna(recall_after) and recall_after > recall_before
    risk_light = traffic_light_risk(
        after_h2026.get("risk_recall", np.nan),
        after_h2026.get("delta_loss_vs_base", np.nan),
        improved,
    )
    product = product_verdict(opp_light, risk_light)

    h2026_delta = float(h2026_opp.iloc[0]["delta_pos_vs_base"]) if not h2026_opp.empty else np.nan

    report = render_report(
        rows=rows,
        examples=example_cards,
        notes=notes,
        anomalies=anomalies,
        opp_light=opp_light,
        risk_light=risk_light,
        product=product,
        before_h2026=before_h2026,
        after_h2026=after_h2026,
        h2026_delta=h2026_delta,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"rows={len(rows)} report={REPORT_PATH}")
    print(f"opportunity: {opp_light} | risk: {risk_light} | product: {product}")
    if not h2026_opp.empty:
        print(f"H2026 strong-threshold Δpos={h2026_opp.iloc[0]['delta_pos_vs_base']:+.4f}")
    print(f"H2026 risk recall before={recall_before} after={recall_after}")


if __name__ == "__main__":
    main()
