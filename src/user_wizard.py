from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from src import APP_PREFIX, DISCLAIMER
from src.data.akshare_adapter import AKShareAdapter


MODE_GUIDE = """正式执行研究工作流前，请先确认任务模式：

1. 单支股票盯盘/调研：围绕一只股票做信息流、Book Skill、新闻/公告、反证和买入/卖出/加减仓/等待建议。
2. 多股候选对比：用户已给 2-20 支候选，系统做同领域或跨领域优先级排序，并给每支候选的操作建议和风险阈值。
3. 策略研究/组合回测：把较大候选池作为策略研究对象，报告 TopK、等权基线、RankIC、消融和成本。
4. 两者结合：先做候选对比缩小范围，再对入选股票做单股深度调研。
5. 盘中/实时盯盘：按用户确认的间隔复核行情、新闻公告和数据缺口。"""

MENU = f"""{APP_PREFIX}

我可以帮你做 A 股研究辅助。请先选择任务模式或工具：

1. 单支股票盯盘/调研
2. 多股候选对比
3. 策略研究/组合回测
4. 两者结合
5. 更新候选池新闻/公告
6. 用书籍策略检查一只股票
7. 回测某条书籍策略
8. 财务排雷专项分析
9. 趋势/技术结构专项分析
10. 查看系统能做什么和不能做什么
11. 盘中/实时盯盘模式
12. 退出
请输入数字："""

HELP = f"""{APP_PREFIX}

用法：
  python -m src.user_wizard

这个向导会先区分任务模式：单支盯盘、多股候选对比、策略研究/组合回测、两者结合。若用户没有说清楚，应先用选择题确认，避免把单股结论、候选对比和组合研究混用。

{DISCLAIMER}
"""


WORKFLOW_OPTIONS = [
    "单支股票调研/盯盘",
    "已关注股票的风险复核",
    "多股候选对比",
    "盘中/实时盯盘",
    "策略研究/组合回测",
]


@dataclass(frozen=True)
class WorkflowRoute:
    workflow_id: str
    workflow_label: str
    confidence: str
    should_ask_user: bool
    reason: str
    questions: list[str] = field(default_factory=list)
    extracted_codes: list[str] = field(default_factory=list)


def route_user_request(text: str) -> WorkflowRoute:
    """Route a free-form user request before running the research workflow.

    This is intentionally deterministic and conservative. Ambiguous requests
    should trigger a choice question instead of forcing P0/P1/P2 incorrectly.
    """

    query = (text or "").strip()
    codes = sorted(set(re.findall(r"(?<!\d)(?:[036]\d{5}|00\d{4}|30\d{4}|60\d{4})(?!\d)", query)))
    lowered = query.lower()

    live_terms = ["盘中", "实时", "盯盘", "20min", "20分钟", "分钟", "interval", "每隔", "开盘后"]
    compare_terms = ["候选", "比较", "对比", "哪个", "哪只", "选", "挑", "top", "top1", "top2", "组合", "配置", "分配"]
    strategy_terms = ["回测", "策略研究", "组合回测", "rankic", "topk", "消融", "baseline", "基线", "收益率", "正收益率"]
    single_terms = ["这只", "这支", "单支", "个股", "能不能买", "要不要买", "卖不卖", "加仓", "减仓", "持有", "买入", "卖出"]
    risk_terms = ["风险", "排雷", "暴雷", "财报", "公告", "负面", "止损", "复核"]

    has_live = any(term in lowered or term in query for term in live_terms)
    has_compare = any(term in lowered or term in query for term in compare_terms)
    has_strategy = any(term in lowered or term in query for term in strategy_terms)
    has_single = any(term in lowered or term in query for term in single_terms) or len(codes) == 1
    has_risk = any(term in lowered or term in query for term in risk_terms)

    if not query:
        return _clarify_route("empty_request", "用户问题为空，需要先选择任务模式。", codes)

    if has_live and (len(codes) == 1 or has_single) and not has_compare:
        return WorkflowRoute(
            workflow_id="live_watch",
            workflow_label="盘中/实时盯盘",
            confidence="high",
            should_ask_user=False,
            reason="用户提到盘中/实时/分钟级复核，并且目标接近单支股票。",
            questions=[],
            extracted_codes=codes,
        )

    if len(codes) >= 2 or (has_compare and not has_strategy):
        return WorkflowRoute(
            workflow_id="candidate_comparison",
            workflow_label="多股候选对比",
            confidence="high" if len(codes) >= 2 else "medium",
            should_ask_user=False if len(codes) >= 2 else True,
            reason="用户给出多支候选或明确要求比较/择优，应先走 P1 ranker-anchor 候选对比。",
            questions=[] if len(codes) >= 2 else _choice_questions(),
            extracted_codes=codes,
        )

    if has_strategy and not has_single:
        return WorkflowRoute(
            workflow_id="strategy_research",
            workflow_label="策略研究/组合回测",
            confidence="medium",
            should_ask_user=False,
            reason="用户关注回测、基线、RankIC、TopK 或策略研究，不应套用单支盯盘口径。",
            questions=[],
            extracted_codes=codes,
        )

    if has_single and has_risk:
        return WorkflowRoute(
            workflow_id="single_stock_risk_review",
            workflow_label="已关注股票的风险复核",
            confidence="high" if len(codes) == 1 else "medium",
            should_ask_user=False,
            reason="用户关注单支股票，并强调风险/公告/财报/止损等复核问题。",
            questions=[],
            extracted_codes=codes,
        )

    if has_single:
        return WorkflowRoute(
            workflow_id="single_stock_watch",
            workflow_label="单支股票调研/盯盘",
            confidence="high" if len(codes) == 1 else "medium",
            should_ask_user=False if len(codes) == 1 else True,
            reason="用户问题更像围绕一只股票做买卖/持有/等待判断。",
            questions=[] if len(codes) == 1 else _choice_questions(),
            extracted_codes=codes,
        )

    return _clarify_route("ambiguous_request", "无法稳定判断用户要单支盯盘、候选对比还是策略研究。", codes)


def render_route_guidance(route: WorkflowRoute) -> str:
    lines = [
        APP_PREFIX,
        "",
        DISCLAIMER,
        "",
        f"建议工作流：{route.workflow_label}",
        f"置信度：{route.confidence}",
        f"原因：{route.reason}",
    ]
    if route.extracted_codes:
        lines.append("识别到的股票代码：" + "、".join(route.extracted_codes))
    if route.should_ask_user:
        lines += ["", "请先确认你要做哪一种："]
        lines += [f"{idx}. {choice}" for idx, choice in enumerate(route.questions or _choice_questions(), start=1)]
    else:
        lines += ["", "下一步：按该工作流生成 evidence pack，再输出明确操作建议、仓位/阈值、证据、反证和复查条件。"]
    return "\n".join(lines) + "\n"


def _choice_questions() -> list[str]:
    return WORKFLOW_OPTIONS.copy()


def _clarify_route(workflow_id: str, reason: str, codes: list[str]) -> WorkflowRoute:
    return WorkflowRoute(
        workflow_id=workflow_id,
        workflow_label="需要用户确认",
        confidence="low",
        should_ask_user=True,
        reason=reason,
        questions=_choice_questions(),
        extracted_codes=codes,
    )


def print_boundary() -> None:
    print(APP_PREFIX)
    print()
    print(MODE_GUIDE)
    print()
    print("我能做：单支盯盘、多股候选对比、盘中/实时盯盘、策略研究/组合回测、新闻公告更新、书籍策略检查、财务排雷、趋势结构分析，并输出明确操作建议、仓位/阈值和风险条件。")
    print("我不能做：收益保证、自动下单、连接券商、泄露或记录 API key/token。")
    print("可使用用户已合法授权的付费/会员/标准化数据源，但必须标注来源、保护凭证并优先走本地缓存。")
    print(DISCLAIMER)


def resolve_interactive(query: str) -> dict | None:
    adapter = AKShareAdapter(dry_run=True)
    result = adapter.resolve_stock(query)
    candidates = result.data or []
    if not candidates:
        print(APP_PREFIX)
        print()
        print(f"你输入的是“{query}”，我没有找到可靠 A 股名称。请重新输入。")
        return None
    first = candidates[0]
    if query not in {first.get("名称"), first.get("代码")}:
        print(APP_PREFIX)
        print()
        print(f"你输入的是“{query}”，我没有找到完全匹配的 A 股名称。")
        print("可能候选包括：")
        for i, item in enumerate(candidates, start=1):
            print(f"{i}. {item.get('名称')}，{item.get('代码')}")
        print(f"{len(candidates) + 1}. 重新输入")
        choice = input("请选择：").strip()
        if choice == str(len(candidates) + 1):
            return None
    return {"code": first.get("代码"), "name": first.get("名称")}


def _confirm_single_stock_task(label: str) -> None:
    query = input(f"{APP_PREFIX}\n\n{label}需要先确认股票。请输入股票名称或代码：").strip()
    stock = resolve_interactive(query)
    if not stock:
        return
    print(APP_PREFIX)
    print()
    print(DISCLAIMER)
    print(f"已确认候选：{stock['name']}（{stock['code']}）。")
    print("下一步可运行 pipeline 生成完整研究报告。")


def _confirm_candidate_comparison_task(with_single_review: bool = False) -> None:
    print(APP_PREFIX)
    print()
    print(DISCLAIMER)
    print("请提供 2-20 支候选，并说明是同领域比较还是跨领域比较。系统会输出候选优先级、每支股票的操作建议、仓位/阈值、反证和信息缺口。")
    if with_single_review:
        print("两者结合模式会先做候选对比，再对入选股票做单支深度盯盘。")


def _confirm_strategy_research_task() -> None:
    print(APP_PREFIX)
    print()
    print(DISCLAIMER)
    print("请在 config/candidates.yaml 中维护候选池。策略研究会输出 TopK、等权/随机/旧规则基线、RankIC、成本、消融和失败块。")


def _confirm_live_watch_task() -> None:
    query = input(f"{APP_PREFIX}\n\n盘中/实时盯盘需要先确认股票。请输入股票名称或代码：").strip()
    stock = resolve_interactive(query)
    if not stock:
        return
    interval = input("请输入复核间隔秒数，直接回车默认 1200 秒：").strip() or "1200"
    print(APP_PREFIX)
    print()
    print(DISCLAIMER)
    print(f"已确认：{stock['name']}（{stock['code']}），复核间隔 {interval} 秒。")
    print("可运行：")
    print(f"python scripts/run_live_watch_session.py --code {stock['code']} --name {stock['name']} --interval-seconds {interval} --max-iterations 1")
    print("若要连续盯盘，把 --max-iterations 改成需要的决策点数量。新闻公告按日缓存，数据源失败会明确标注信息不足。")


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP)
        return
    while True:
        choice = input(MENU).strip()
        if choice == "12":
            print(APP_PREFIX)
            print()
            print("已退出。")
            return
        if choice == "10":
            print_boundary()
            continue
        if choice == "1":
            _confirm_single_stock_task("单支股票盯盘/调研")
            continue
        if choice == "2":
            _confirm_candidate_comparison_task()
            continue
        if choice == "3":
            _confirm_strategy_research_task()
            continue
        if choice == "4":
            _confirm_candidate_comparison_task(with_single_review=True)
            continue
        if choice == "5":
            print(APP_PREFIX)
            print()
            print("请在 config/candidates.yaml 中维护候选池，然后运行新闻/公告更新流程。")
            continue
        if choice in {"6", "7", "8", "9"}:
            labels = {
                "6": "书籍策略检查",
                "7": "书籍策略回测",
                "8": "财务排雷专项分析",
                "9": "趋势/技术结构专项分析",
            }
            _confirm_single_stock_task(labels[choice])
            continue
        if choice == "11":
            _confirm_live_watch_task()
            continue
        print(APP_PREFIX)
        print()
        print("输入无效，请输入 1-12。")


if __name__ == "__main__":
    main()
