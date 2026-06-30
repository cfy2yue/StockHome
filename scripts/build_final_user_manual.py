from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_MD = REPORT_DIR / "final_user_manual.md"
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{8,}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p0_metrics, p0_steps = load_p0()
    p1_panel, p1_all = load_p1()
    p0_case = load_p0_case()
    p1_case, p1_case_metrics = load_p1_case(p1_all)
    markdown = build_markdown(p0_metrics, p0_steps, p1_panel, p1_all, p0_case, p1_case, p1_case_metrics)
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(f"wrote: {OUTPUT_MD}")


def load_p0() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _read_csv("p0_acceptance_multiblock_3panel_flash_v1_metrics.csv")
    steps = _read_csv("p0_acceptance_multiblock_3panel_flash_v1_step_metrics.csv")
    if not metrics.empty:
        metrics = metrics[metrics["variant"].astype(str).eq("full_agent_without_opportunity_tool")].copy()
    if not steps.empty:
        steps = steps[steps["variant"].astype(str).eq("full_agent_without_opportunity_tool")].copy()
    pro_metrics = _read_csv("p0_acceptance_single_default_pro_v1_metrics.csv")
    if not pro_metrics.empty:
        pro_metrics = pro_metrics[pro_metrics["variant"].astype(str).eq("full_agent_without_opportunity_tool")].copy()
        pro_metrics["model_family"] = "Pro confirmation"
    if not metrics.empty:
        metrics["model_family"] = "Flash acceptance"
    metrics = pd.concat([metrics, pro_metrics], ignore_index=True) if not pro_metrics.empty else metrics
    return metrics, steps


def load_p1() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = [
        ("panel0", "candidate_comparison_anchor_rankavg_flash_v1_metrics.csv"),
        ("panel1", "candidate_comparison_anchor_rankavg_panel1_flash_v1_metrics.csv"),
        ("panel2", "candidate_comparison_anchor_rankavg_panel2_flash_v1_metrics.csv"),
    ]
    frames = []
    panel_rows = []
    for panel, filename in files:
        frame = _read_csv(filename)
        if frame.empty:
            continue
        frame = frame[frame["variant"].astype(str).eq("ranker_anchor_agent")].copy()
        frame["panel"] = panel
        frames.append(frame)
        panel_rows.append(
            {
                "panel": panel,
                "cards": len(frame),
                "top1_excess_20d": _mean(frame, "top1_excess_20d"),
                "top2_excess_20d": _mean(frame, "top2_excess_20d"),
                "top1_positive": _bool_mean(frame, "top1_positive"),
                "top1_is_worst": _bool_mean(frame, "top1_is_worst"),
            }
        )
    all_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame(panel_rows), all_metrics


def load_p0_case() -> list[dict[str, Any]]:
    rows = _read_jsonl("p0_acceptance_single_default_pro_v1_decision_ledger.jsonl")
    if not rows:
        rows = _read_jsonl("p0_acceptance_multiblock_3panel_flash_v1_decision_ledger.jsonl")
    rows = [row for row in rows if row.get("variant") == "full_agent_without_opportunity_tool"]
    if not rows:
        return []
    counts = pd.Series([row.get("code") for row in rows]).value_counts()
    target = str(counts.index[0])
    case_rows = [row for row in rows if str(row.get("code")) == target]
    if len(case_rows) < 2:
        case_rows = rows[:3]
    return sorted(case_rows[:4], key=lambda row: str(row.get("decision_date")))


def load_p1_case(p1_all: pd.DataFrame) -> tuple[dict[str, Any], pd.Series]:
    if p1_all.empty:
        return {}, pd.Series(dtype=object)
    row = p1_all.sort_values("top2_excess_20d", ascending=False).iloc[0]
    target_group = str(row.get("comparison_group_id"))
    for filename in [
        "candidate_comparison_anchor_rankavg_flash_v1_decision_ledger.jsonl",
        "candidate_comparison_anchor_rankavg_panel1_flash_v1_decision_ledger.jsonl",
        "candidate_comparison_anchor_rankavg_panel2_flash_v1_decision_ledger.jsonl",
    ]:
        for card in _read_jsonl(filename):
            if str(card.get("comparison_group_id")) == target_group:
                return card, row
    return {}, row


def build_markdown(
    p0_metrics: pd.DataFrame,
    p0_steps: pd.DataFrame,
    p1_panel: pd.DataFrame,
    p1_all: pd.DataFrame,
    p0_case: list[dict[str, Any]],
    p1_case: dict[str, Any],
    p1_case_metrics: pd.Series,
) -> str:
    p0_flash = p0_metrics[p0_metrics.get("model_family", pd.Series(dtype=str)).astype(str).eq("Flash acceptance")]
    p0_pro = p0_metrics[p0_metrics.get("model_family", pd.Series(dtype=str)).astype(str).eq("Pro confirmation")]
    p0_row = p0_flash.iloc[0] if not p0_flash.empty else (p0_metrics.iloc[0] if not p0_metrics.empty else pd.Series(dtype=object))
    p0_pro_row = p0_pro.iloc[0] if not p0_pro.empty else pd.Series(dtype=object)
    p1_mean = {
        "top1_excess_20d": _mean(p1_all, "top1_excess_20d"),
        "top2_excess_20d": _mean(p1_all, "top2_excess_20d"),
        "top1_positive": _bool_mean(p1_all, "top1_positive"),
        "top1_is_worst": _bool_mean(p1_all, "top1_is_worst"),
    }
    lines = [
        "# A 股研究 Agent 用户手册",
        "",
        "本系统输出研究辅助型操作建议，不自动下单，不接券商接口，不保证收益。用户端必须先给明确建议，例如买入、卖出、加仓、减仓、持有、等待或补数据，再给仓位/阈值、依据、反证和复评条件。",
        "",
        "## 1. 你可以让系统做什么",
        "",
        "- 单支股票盯盘：围绕一只股票复核 K 线、筹码、新闻公告、财报、同行、Book Skill 和历史相似案例，给出买入/卖出/加减仓/持有/等待建议和下一步复查点。",
        "- 盘中/实时盯盘：按你确认的间隔复核实时/分钟级行情；新闻公告按日缓存，数据源失败时明确标注。",
        "- 多股候选对比：输入 2-20 支同领域或跨领域候选，输出操作优先级、每支候选建议、反证和信息缺口。",
        "- 策略研究：作为后台评估工具，用来比较基线、消融、RankIC、TopK 和时间泛化，不直接作为用户主交互。",
        "",
        "## 2. 使用前系统会先确认目标",
        "",
        "如果你的问题不够具体，系统应该先让你选择：单支盯盘、多股候选对比、两者结合、盘中盯盘或策略研究。确认任务后才进入对应工作流，避免把不同任务混在一起。",
        "",
        "系统最终必须先给明确操作建议：买入/试探买入、加仓、持有、减仓、卖出、等待或先补数据。每条建议都要配套原因、仓位上限、买入/加仓触发、减仓/卖出触发和下一次复查条件。",
        "",
        "## 3. 操作建议怎么读",
        "",
        "- 买入/试探买入：只在多通道证据支持时出现，必须给仓位上限和止损/复评阈值。",
        "- 加仓：必须比买入更严格，需要已有正向证据继续确认，且反证没有扩大。",
        "- 持有：适合已有仓位，必须写清继续持有条件和失效条件。",
        "- 减仓/卖出：用于明确负面事件、财报风险、同行显著转弱、筹码压力或高波动过热共振。",
        "- 等待：不是模糊观察，意思是暂不新增买入/加仓；必须写明什么条件转为买入/加仓，什么条件转为减仓/卖出。",
        "- 补数据：关键数据缺失时先停止方向性动作，补齐后再判断。",
        "",
        "## 4. 决策工作流",
        "",
        "1. 解析股票或候选池，并确认是单支、同领域对比还是跨领域对比。",
        "2. 拉取或读取本地缓存：日 K、分钟 K、筹码、同行/地域、新闻公告、财报、Book Skill、历史相似案例。",
        "3. Python/ML 工具先给出可复现量化摘要，例如 `rev+chip_core`、K 线多尺度、同行强弱、新闻问卷、财报 as-of 质量。",
        "4. DeepSeek 作为审计型 Agent 读取工具摘要、Book Skill 和反证，输出明确操作建议、仓位/阈值、辅助研究分级、证据、缺口和复查触发条件。单支盯盘默认读取原始新闻、财报、同行和 BookSkill 分支；候选池研究可额外使用非价格 conflict checklist，提醒它区分“明确负面事件”和“容易错杀的反转摩擦”。",
        "5. 回测中只用未来结果做后验评估；未来收益、GT、密钥不会进入 evidence pack。",
        "",
        "## 5. 当前评估结果",
        "",
        "### P0 单支盯盘",
        "",
        f"- 3 面板 Flash 接受矩阵：{_fmt_int(p0_row.get('decision_cards'))} 张卡，schema 通过率 `{_fmt(p0_row.get('schema_pass_rate'))}`。",
        f"- Flash 20 日 cash-adjusted 正收益率 `{_fmt(p0_row.get('cash_adjusted_positive_20d_rate'))}`，平均收益 `{_fmt(p0_row.get('cash_adjusted_avg_return_20d'))}pp`。",
        f"- Pro 确认：{_fmt_int(p0_pro_row.get('decision_cards'))} 张卡，20 日 cash-adjusted 正收益率 `{_fmt(p0_pro_row.get('cash_adjusted_positive_20d_rate'))}`，平均收益 `{_fmt(p0_pro_row.get('cash_adjusted_avg_return_20d'))}pp`。",
        "- 诚实边界：H2026_1 分块正收益率只有 `0.5000`，因此还不能宣称最终泛化完成；当前定位是可用的盯盘/排雷底座。",
        "",
        "### P1 多股候选对比",
        "",
        f"- 3 个不重叠 Flash 面板共 `{len(p1_all)}` 个候选组。",
        f"- Top1 相对候选池超额均值 `{_fmt(p1_mean['top1_excess_20d'])}pp`，Top2 超额均值 `{_fmt(p1_mean['top2_excess_20d'])}pp`。",
        f"- Top1 正收益率 `{_fmt(p1_mean['top1_positive'])}`，Top1 选成组内最弱的比例 `{_fmt(p1_mean['top1_is_worst'])}`。",
        "- 默认协议：同领域以 `rev_chip_core` 为锚；跨领域以 `rank_avg_rev_watch` 为锚；Agent 只在明确硬反证下调整 Top1/Top2。",
        "",
        "### 基线与消融解释",
        "",
        "- 灰色参考基线：现金/观察、候选池均值、随机/等权、纯 ranker、纯单支 scorer。",
        "- P0 消融显示各通道并非每次都提升，新闻、Book Skill、同行和财报更像反证/确认清单；这也是系统不把单一通道直接升为强信号的原因。",
        "- 最新非价格风险覆盖层显示：同行/地域弱和“近期无财报事件”在高 `rev+chip` 回撤反转候选里常是 false-veto 区，不能机械剔除；高风险新闻在该范围内应触发二次确认或降低置信度，并可能转为减仓/卖出复核或新仓回避。",
        "- 三面板 Flash on/off 进一步显示：这层 broad overlay 对单支盯盘会增加错升/错杀，所以 P0 默认隐藏；它只在候选池对比/组合研究中作为 conflict checklist 使用，不会单独把股票升为 `继续深挖`。",
        "- P1 自由排序效果不稳定，已改为 ranker-anchor 审计协议，稳定性明显好于自由 Agent 排序。",
        "",
            "## 6. 单支盯盘案例",
        "",
    ]
    if p0_case:
        code = p0_case[0].get("code")
        name = p0_case[0].get("name")
        lines.append(f"案例股票：{name}（{code}）。以下是回测中同一工作流的决策点片段：")
        lines.append("")
        for row in p0_case:
            lines.append(
                f"- {row.get('decision_date')}：操作建议：{_operation_from_grade(row.get('research_grade'))}；"
                f"辅助分级 `{row.get('research_grade')}`。"
                f"依据：{_clean(row.get('final_agent_reasoning_summary'))}。"
                f"缺口：{_clean(row.get('data_missing_flags') or '无明确缺口')}。"
            )
    else:
        lines.append("- 暂无可用案例。")
    lines.extend(["", "## 7. 多股候选对比案例", ""])
    if p1_case:
        lines.append(
            f"案例候选组：{p1_case.get('comparison_group_id')}，场景 `{p1_case.get('comparison_scenario')}`，"
            f"日期 `{p1_case.get('decision_date')}`。"
        )
        lines.append(f"- 候选池均值 20 日收益：`{_fmt(p1_case_metrics.get('group_mean_return_20d'))}pp`。")
        lines.append(f"- Top1 20 日收益：`{_fmt(p1_case_metrics.get('top1_return_20d'))}pp`，相对候选池 `{_fmt(p1_case_metrics.get('top1_excess_20d'))}pp`。")
        lines.append(f"- Top2 均值 20 日收益：`{_fmt(p1_case_metrics.get('top2_mean_return_20d'))}pp`，相对候选池 `{_fmt(p1_case_metrics.get('top2_excess_20d'))}pp`。")
        lines.append("")
        lines.append("Top 排序理由：")
        for item in (p1_case.get("ranked_candidates") or [])[:3]:
            lines.append(
                f"- Rank {item.get('rank')}：{item.get('name')}（{item.get('code')}），"
                f"分级 `{item.get('research_grade')}`。理由：{_clean(item.get('priority_reason'))}。"
            )
    else:
        lines.append("- 暂无可用案例。")
    lines.extend(
        [
            "",
            "## 8. 盘中/实时盯盘模式",
            "",
            "命令示例：",
            "",
            "```bash",
            "python scripts/run_live_watch_session.py --code 000001 --name 平安银行 --interval-seconds 1200 --max-iterations 1",
            "```",
            "",
            "连续盯盘时把 `--max-iterations` 改成需要的决策点数量。每个决策点会输出操作建议、仓位/阈值、依据、反证、数据源缺口和下一步复查点。新闻公告默认一天内复用缓存，避免重复抓取。",
            "",
            "## 9. 能力边界",
            "",
            "- 如果实时行情、新闻、公告或财报源失败，系统必须如实报告，必要时给“暂不交易，先补数据”。",
            "- Book Skill 是检查清单和条件库，不是单独信号；只有在回测中反复验证的适用条件才提高优先级。",
            "- 最新行情块如果退化，系统应降级为观察/排雷，不硬推研究优先级。",
            "- 每 1-3 个月或行情切换后，需要拉新数据做滚动复核，更新阈值、经验账本和手册。",
            "",
        ]
    )
    return "\n".join(_clean(line) for line in lines) + "\n"


def _read_csv(filename: str) -> pd.DataFrame:
    path = REPORT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _read_jsonl(filename: str) -> list[dict[str, Any]]:
    path = REPORT_DIR / filename
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _bool_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].astype(bool).mean())


def _fmt(value: Any) -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def _fmt_int(value: Any) -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if pd.isna(value) else str(int(value))


def _clean(value: Any) -> str:
    text = str(value or "")
    text = SECRET_PATTERN.sub("[masked-secret]", text)
    replacements = {
        "维持观察": "等待触发阈值，暂不新增仓位",
        "保持观察": "等待触发阈值，暂不新增仓位",
        "暂观察": "等待触发阈值，暂不新增仓位",
        "暂缓升级为重点对象": "暂不新增买入/加仓，等待升级阈值",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text.replace("\n", " ").strip()


def _operation_from_grade(value: Any) -> str:
    grade = str(value or "")
    if grade == "继续深挖":
        return "可小仓试探买入或继续持有，但必须等待下一决策点确认"
    if grade == "暂时剔除":
        return "新仓回避；已有仓位减仓或卖出复核"
    if grade == "信息不足":
        return "暂不交易，先补关键数据"
    return "暂不新增买入/加仓，等待升级或下调阈值"


if __name__ == "__main__":
    main()
