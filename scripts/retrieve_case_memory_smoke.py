from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.case_memory_retriever import format_retrieved_cases, retrieve_cases  # noqa: E402


DEFAULT_REPORT = ROOT / "reports" / "date_generalization" / "case_memory_retrieval_smoke.md"
DEFAULT_QUERIES = [
    "no_news Python relative strength news financial Book Skill missing bad active exposure",
    "financial_report_only without ordinary news peer Book Skill confirmation upgrade risk",
    "announcement row cap partial coverage official source count not alpha",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test lightweight local case-memory retrieval without model calls.")
    parser.add_argument("--query", action="append", default=[], help="Query text. Can be repeated. Defaults to three current workflow probes.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(queries, top_k=args.top_k), encoding="utf-8")

    print("A股研究Agent")
    print(f"queries={len(queries)}")
    print(f"top_k={args.top_k}")
    print(f"report={report_path}")


def render_report(queries: list[str], *, top_k: int) -> str:
    lines = [
        "# Case Memory Retrieval Smoke",
        "",
        "本报告只用于研究辅助和工作流验证，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "验证轻量本地案例检索是否能从已有 memory 中找回相似失败模式、反证规则和下一步动作。该 smoke 不调用 DeepSeek，不读取 API key/token，不向决策 evidence pack 注入未来收益或 GT 字段。",
        "",
        "## Configuration",
        "",
        f"- top_k: `{top_k}`",
        "- retriever: `src/agent_training/case_memory_retriever.py`",
        "- source_ledgers: strategy/book_skill/news_world_model/ablation/failure_case",
        "",
    ]
    for index, query in enumerate(queries, start=1):
        cases = retrieve_cases(ROOT, query, top_k=top_k)
        lines.extend(
            [
                f"## Query {index}",
                "",
                f"`{query}`",
                "",
                "```text",
                format_retrieved_cases(cases, max_chars=1800),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- 如果检索能稳定找回对应失败模式和下一步动作，可进入下一轮 dry-run evidence pack 的候选输入。",
            "- 如果检索只返回宽泛旧结论或没有命中，应保持为离线辅助，不进入默认 Agent 决策。",
            "- 接入 DeepSeek 前必须比较 `no_rag`、`memory_compact_only`、`retrieved_cases_v1`，并跑 evidence leakage audit。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
