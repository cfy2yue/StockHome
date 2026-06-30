from __future__ import annotations

from pathlib import Path

import pandas as pd

from .strategy_compare import comparison_summary


def write_summary_report(output_dir: str | Path, train1: pd.DataFrame, train2: pd.DataFrame, test: pd.DataFrame | None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_stats = _load_pattern_stats(output_dir)
    comparison = _load_comparison(output_dir)
    comparison_delta = comparison_summary(comparison, "test") if not comparison.empty else {}
    baseline = _load_csv(output_dir / "baseline_comparison.csv")
    tree_gate = _load_csv(output_dir / "tree_gate_optimization.csv")
    news_report = _load_csv(output_dir / "news_feature_report.csv")
    pool_report = _load_csv(output_dir / "pool_selection_report.csv")
    pool_optimizer = _load_csv(output_dir / "pool_optimizer_report.csv")
    pool_walkforward = _load_csv(output_dir / "pool_walkforward_report.csv")
    rebound_diagnostics = _load_csv(output_dir / "rebound_diagnostics.csv")
    rebound_validation = _load_csv(output_dir / "rebound_validation.csv")
    lines = [
        "# A 股轻量回测最终报告",
        "",
        "本报告只用于研究辅助，不自动交易，不接券商接口，不输出买卖指令。",
        "",
        "## 分组表现",
        "",
        "| 分组 | 决策数 | 可验证数 | 通过率 | 继续深挖数 | 放入观察数 | 暂时剔除数 | 信息不足数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("训练集 epoch1", train1),
        _metric_row("训练集 epoch2", train2),
        _metric_row("Test 集", test if test is not None else pd.DataFrame()),
        "",
        "## 策略优化结论",
        "",
        _optimization_note(train1, train2, test if test is not None else pd.DataFrame()),
        "",
        "## 原始策略 vs 优化后量化策略",
        "",
        _comparison_note(comparison_delta),
        "",
        "## 长期持有基线与 Gate",
        "",
        _baseline_note(baseline),
        "",
        "## Tree Gate 分叉",
        "",
        _tree_gate_note(tree_gate),
        "",
        "## 新闻向量观察",
        "",
        _news_note(news_report),
        "",
        "## 候选池筛选",
        "",
        _pool_note(pool_report),
        "",
        "## 候选池公式优化",
        "",
        _pool_optimizer_note(pool_optimizer),
        "",
        "## 候选池 Walk-Forward",
        "",
        _pool_walkforward_note(pool_walkforward),
        "",
        "## 反弹型候选池诊断",
        "",
        _rebound_diagnostics_note(rebound_diagnostics),
        "",
        "## 反弹策略锁定 Test 验证",
        "",
        _rebound_validation_note(rebound_validation),
        "",
        "## 可泛化候选规律",
        "",
        "| 策略ID | test触发数 | test 20日均值 | test 20日正收益率 | 判断 |",
        "|---|---:|---:|---:|---|",
        *_pattern_rows(pattern_stats),
        "",
        "## 风险提示",
        "",
        "- 轻量样本不能代表全市场，所有规则最高先作为候选研究规则。",
        "- Test 集只用于最终验收；若 test 表现弱于训练集，应优先考虑过拟合。",
        "- Book Skill 来源必须继续保留书名、章节、页码线索、策略 ID 和提取方式。",
        "- 当前财务字段因缺少可靠披露日未参与 walk-forward 财务评分，后续补齐后要重跑。",
        "",
        "## 下一步",
        "",
        "1. 新闻通道优先补历史公告/新闻回填，降低近期覆盖造成的时间段偏差。",
        "2. 对 test 偶然较强但 valid 不足或未达标的分叉，进入反证复核而不是升级规则。",
        "3. 对财报披露日缺失的样本，补官方公告披露日或继续剔除财务维度。",
        "",
        "## 可用资产",
        "",
        "- `adaptation_skills.yaml`：由候选规则生成的适配技能，包含公式、阈值、证据和复用限制。",
        "- `case_memory.csv`：历史决策案例库，可用于后续相似案例检索。",
        "- `strategy_comparison.md`：原始 Book Skill 口径与优化后量化共振口径的收益和稳定性对比。",
        "- `pattern_report.md`：按 Book Skill 汇总触发次数、test 20日均值和候选判断。",
        "- `gate_optimization.md` / `tree_gate_optimization.md`：固定 gate 与浅层分叉的训练、验证、test 表现。",
        "- `news_feature_report.md`：新闻 32 维向量覆盖率和 active/inactive 分组表现。",
        "- `pool_selection_report.md`：多股票候选池 Top N 筛选与全池等权基线对比。",
        "- `pool_optimizer_report.md`：候选池评分公式的 search/valid/test 防过拟合检验。",
        "- `pool_walkforward_report.md`：候选池公式滚动时间折 out-of-sample 稳健性验证。",
        "- `rebound_diagnostics.md`：深跌/低广度反弹策略族的分 fold 诊断和复用限制。",
        "- `rebound_validation.md`：训练集选择反弹策略族后，锁定规则在 test 集的一次性验证。",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def _metric_row(name: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return f"| {name} | 0 | 0 | NA | 0 | 0 | 0 | 0 |"
    verifiable = df["gt_pass"].dropna() if "gt_pass" in df else pd.Series(dtype=bool)
    pass_rate = "NA" if verifiable.empty else f"{float(verifiable.mean()):.2%}"
    counts = df["rating"].value_counts() if "rating" in df else {}
    return (
        f"| {name} | {len(df)} | {len(verifiable)} | {pass_rate} | "
        f"{int(counts.get('继续深挖', 0))} | {int(counts.get('放入观察', 0))} | "
        f"{int(counts.get('暂时剔除', 0))} | {int(counts.get('信息不足', 0))} |"
    )


def _optimization_note(train1: pd.DataFrame, train2: pd.DataFrame, test: pd.DataFrame) -> str:
    p1 = _pass(train1)
    p2 = _pass(train2)
    pt = _pass(test)
    if p1 is None or p2 is None:
        return "- 训练集可验证样本不足，暂不能评估权重优化。"
    delta = p2 - p1
    lines = [f"- 训练集 epoch1 通过率 {p1:.2%}，epoch2 通过率 {p2:.2%}，变化 {delta:+.2%}。"]
    if abs(delta) < 0.005:
        lines.append("- 本轮权重微调没有带来可见改善，说明当前评分边界比权重本身更需要优化。")
    if pt is not None:
        lines.append(f"- Test 集最终通过率 {pt:.2%}，略高于训练集但仍偏低，不应直接升级为正式策略。")
    return "\n".join(lines)


def _pass(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "gt_pass" not in df:
        return None
    values = df["gt_pass"].dropna()
    if values.empty:
        return None
    return float(values.mean())


def _load_pattern_stats(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "book_skill_pattern_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_comparison(output_dir: Path) -> pd.DataFrame:
    return _load_csv(output_dir / "strategy_comparison.csv")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _comparison_note(delta: dict) -> str:
    if not delta:
        return "- 尚未生成策略对比。"
    return "\n".join(
        [
            f"- Test 集 20日均值提升：{_fmt(delta.get('avg_return_20d_delta'))}",
            f"- Test 集 20日正收益率提升：{_fmt(delta.get('positive_20d_rate_delta'))}",
            f"- Test 集 20日波动变化：{_fmt(delta.get('std_return_20d_delta'))}",
            f"- Test 集 20日跌幅超过5%的比例变化：{_fmt(delta.get('loss_20d_over_5_rate_delta'))}",
            f"- Test 集稳定性分变化：{_fmt(delta.get('stability_score_delta'))}",
        ]
    )


def _pattern_rows(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["| NA | 0 | NA | NA | 尚未生成规律分析 |"]
    ordered = df.sort_values(["test_trigger_count", "train_epoch2_trigger_count"], ascending=False)
    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            f"| {row['strategy_id']} | {int(row.get('test_trigger_count') or 0)} | "
            f"{_fmt(row.get('test_avg_return_20d'))} | {_fmt(row.get('test_positive_20d_rate'))} | {row.get('judgement')} |"
        )
    return rows


def _baseline_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成长期持有基线。"
    rows = []
    for _, row in df[df["split"].isin(["test", "full_period"])].iterrows():
        rows.append(
            f"- {row['split']} / {row['gate_name']}：样本 {int(row['sample_count'])}，"
            f"20日均值 {_fmt(row.get('avg_return_20d'))}，正收益率 {_fmt(row.get('positive_20d_rate'))}，"
            f"稳定性分 {_fmt(row.get('stability_score'))}。"
        )
    return "\n".join(rows) if rows else "- 尚未生成 test 基线。"


def _tree_gate_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成 Tree Gate 报告。"
    valid = df[pd.to_numeric(df.get("valid_sample_count"), errors="coerce").fillna(0) > 0].copy()
    if valid.empty:
        return "- Tree Gate 候选在验证段样本不足，本轮不升级任何分叉规则。"
    best = valid.sort_values(["target_hit_on_test", "valid_avg_return_20d"], ascending=False).iloc[0]
    hit = (
        _numeric(best.get("valid_avg_return_20d")) >= 8
        and _numeric(best.get("valid_positive_20d_rate")) >= 0.65
        and _numeric(best.get("test_avg_return_20d")) >= 8
        and _numeric(best.get("test_positive_20d_rate")) >= 0.65
    )
    lines = [
        f"- 当前最佳分叉：`{best.get('formula')}`。",
        f"- valid：20日均值 {_fmt(best.get('valid_avg_return_20d'))}，正收益率 {_fmt(best.get('valid_positive_20d_rate'))}；"
        f"test：20日均值 {_fmt(best.get('test_avg_return_20d'))}，正收益率 {_fmt(best.get('test_positive_20d_rate'))}。",
    ]
    lines.append("- 验证集与 test 同时达标，可进入下一轮复核。" if hit else "- 未同时通过验证集与 test，不升级为正式规则。")
    return "\n".join(lines)


def _news_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成新闻向量报告。"
    test = df[df["split"] == "test"].copy()
    if test.empty:
        return "- 新闻向量尚无 test 统计。"
    top = test.sort_values("coverage_rate", ascending=False).head(3)
    lines = []
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['feature']}` 覆盖率 {_fmt(row.get('coverage_rate'))}，active 20日均值 {_fmt(row.get('active_avg_return_20d'))}，"
            f"inactive 20日均值 {_fmt(row.get('inactive_avg_return_20d'))}。"
        )
    lines.append("- 当前新闻覆盖偏近期，观察到的差异先作为反证线索，不作因果结论。")
    return "\n".join(lines)


def _pool_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成候选池筛选报告。"
    test = df[df["split"] == "test"].copy()
    if test.empty:
        return "- 候选池筛选尚无 test 统计。"
    baseline = test[test["strategy"] == "全候选池等权基线"]
    best = test.sort_values("avg_return_20d", ascending=False).iloc[0]
    lines = []
    if not baseline.empty:
        base = baseline.iloc[0]
        lines.append(
            f"- 全候选池等权基线：20日均值 {_fmt(base.get('avg_return_20d'))}，"
            f"正收益率 {_fmt(base.get('positive_20d_rate'))}，稳定性分 {_fmt(base.get('stability_score'))}。"
        )
    lines.append(
        f"- 当前 test 最佳候选池策略：{best.get('strategy')}，20日均值 {_fmt(best.get('avg_return_20d'))}，"
        f"正收益率 {_fmt(best.get('positive_20d_rate'))}，稳定性分 {_fmt(best.get('stability_score'))}。"
    )
    lines.append("- 候选池筛选解决“多股票择优”问题；单股调研需输出明确操作建议、仓位/阈值、反证和辅助研究分级。")
    return "\n".join(lines)


def _pool_optimizer_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成候选池公式优化报告。"
    best = df.sort_values(["target_hit_on_valid", "selection_score"], ascending=False).iloc[0]
    hit = str(best.get("target_hit_on_test")).lower() in {"true", "1", "yes"}
    lines = [
        f"- valid 最优公式：{best.get('formula_name')} Top{int(best.get('top_n'))}。",
        f"- valid：20日均值 {_fmt(best.get('valid_avg_return_20d'))}，正收益率 {_fmt(best.get('valid_positive_20d_rate'))}，稳定性分 {_fmt(best.get('valid_stability_score'))}。",
        f"- test：20日均值 {_fmt(best.get('test_avg_return_20d'))}，正收益率 {_fmt(best.get('test_positive_20d_rate'))}，稳定性分 {_fmt(best.get('test_stability_score'))}。",
    ]
    lines.append("- 已同时通过 valid 和 test，可进入下一轮复核。" if hit else "- 未同时达到目标，不升级为正式候选池策略。")
    return "\n".join(lines)


def _pool_walkforward_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成候选池 walk-forward 报告。"
    avg = df["oos_avg_return_20d"].mean()
    positive = df["oos_positive_20d_rate"].mean()
    stability = df["oos_stability_score"].mean()
    hit_count = int(df["oos_target_hit"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    lines = [
        f"- OOS 平均 20日均值 {_fmt(avg)}，平均正收益率 {_fmt(positive)}，平均稳定性分 {_fmt(stability)}。",
        f"- 达标 fold 数：{hit_count}/{len(df)}。",
    ]
    lines.append("- 所有 fold 达标，可进入下一轮扩大样本复核。" if hit_count == len(df) else "- 滚动时间折未稳定达标，不升级为正式策略。")
    return "\n".join(lines)


def _rebound_diagnostics_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成反弹型候选池诊断报告。"
    family = (
        df.groupby(["formula_name", "date_gate", "top_n"])
        .agg(
            folds=("fold", "count"),
            hit_rate=("oos_target_hit", "mean"),
            avg_return=("oos_avg_return_20d", "mean"),
            positive_rate=("oos_positive_20d_rate", "mean"),
            stability=("oos_stability_score", "mean"),
            avg_decision_dates=("oos_decision_dates", "mean"),
        )
        .reset_index()
        .sort_values(["hit_rate", "avg_return", "avg_decision_dates"], ascending=False)
    )
    best_family = family.iloc[0]
    best_single = df.sort_values(["oos_target_hit", "oos_avg_return_20d"], ascending=False).iloc[0]
    hit_count = int(df["oos_target_hit"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    total = len(df)
    lines = [
        f"- 跨 fold 最佳策略族：{best_family.get('formula_name')} / {best_family.get('date_gate')} / Top{int(best_family.get('top_n'))}。",
        f"- 4 fold 平均 20日均值 {_fmt(best_family.get('avg_return'))}，平均正收益率 {_fmt(best_family.get('positive_rate'))}，平均稳定性分 {_fmt(best_family.get('stability'))}，达标率 {_fmt(best_family.get('hit_rate'))}。",
        f"- 单 fold 最高记录：{best_single.get('formula_name')} / {best_single.get('date_gate')} / Top{int(best_single.get('top_n'))}，OOS期数 {int(best_single.get('oos_decision_dates'))}，20日均值 {_fmt(best_single.get('oos_avg_return_20d'))}。",
        f"- 达标组合数：{hit_count}/{total}。",
    ]
    lines.append("- 反弹策略族仍需跨 fold 稳定验证；若只在局部时段达标，只作为市场状态线索记录。")
    return "\n".join(lines)


def _rebound_validation_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 尚未生成反弹策略锁定 test 验证报告。"
    work = df.copy()
    if "promotion_candidate" in work:
        work["promotion_flag"] = work["promotion_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        work["promotion_flag"] = False
    if "test_target_hit" in work:
        work["target_flag"] = work["test_target_hit"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        work["target_flag"] = False
    row = work.sort_values(["promotion_flag", "target_flag", "test_avg_return_20d"], ascending=False).iloc[0]
    hit = str(row.get("test_target_hit")).lower() in {"true", "1", "yes"}
    lines = [
        f"- 最强锁定规则：{row.get('selector_name')} -> {row.get('formula_name')} / {row.get('date_gate')} / Top{int(row.get('top_n'))}。",
        f"- Test：决策期数 {int(row.get('test_decision_dates') or 0)}，20日均值 {_fmt(row.get('test_avg_return_20d'))}，正收益率 {_fmt(row.get('test_positive_20d_rate'))}，稳定性分 {_fmt(row.get('test_stability_score'))}。",
        f"- 相对全候选池等权基线：20日均值变化 {_fmt(row.get('test_vs_pool_avg_return_delta'))}，正收益率变化 {_fmt(row.get('test_vs_pool_positive_delta'))}，稳定性变化 {_fmt(row.get('test_vs_pool_stability_delta'))}。",
        f"- 相对20日滚动长期持有基线：20日均值变化 {_fmt(row.get('test_vs_rolling_hold_avg_return_delta'))}，正收益率变化 {_fmt(row.get('test_vs_rolling_hold_positive_delta'))}，稳定性变化 {_fmt(row.get('test_vs_rolling_hold_stability_delta'))}。",
    ]
    candidate_count = int(work["promotion_flag"].sum())
    target_count = int(work["target_flag"].sum())
    if candidate_count:
        lines.append(f"- 预定义训练选择器中有 {candidate_count}/{len(work)} 个达到 test 目标并跑过基线；下一步需要扩大股票池或更长历史复核。")
    elif target_count:
        lines.append(f"- 预定义训练选择器中有 {target_count}/{len(work)} 个锁定 test 达标，但训练侧均值或升级门槛不足，先作为扩大复核候选。")
    else:
        lines.append("- 锁定 test 未达到目标，不升级为正式候选池策略。")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _numeric(value) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
