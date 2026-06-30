"""Smoke-test deterministic user intake routing for product handoff.

No external APIs are called. This checks that ambiguous user questions trigger
choice-style clarification, while clear P0/P1/live/strategy prompts route to
the right workflow before any evidence pack or agent decision is generated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.user_wizard import render_route_guidance, route_user_request


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "user_intake_router_audit_v1"

CASES = [
    {
        "case_id": "single_buy",
        "prompt": "000001 现在能不能买，要不要先试探买入？",
        "expected_workflow": "single_stock_watch",
        "expected_ask": False,
    },
    {
        "case_id": "single_risk",
        "prompt": "我已经持有000001，最近财报和公告有没有风险，要不要减仓？",
        "expected_workflow": "single_stock_risk_review",
        "expected_ask": False,
    },
    {
        "case_id": "multi_compare",
        "prompt": "000001 000002 600000 三只里面我只想选1只，哪个更值得买？",
        "expected_workflow": "candidate_comparison",
        "expected_ask": False,
    },
    {
        "case_id": "live_watch",
        "prompt": "开盘后帮我盯盘000001，每20分钟复核一次。",
        "expected_workflow": "live_watch",
        "expected_ask": False,
    },
    {
        "case_id": "strategy_research",
        "prompt": "我想看这个策略的回测、RankIC、baseline和消融。",
        "expected_workflow": "strategy_research",
        "expected_ask": False,
    },
    {
        "case_id": "ambiguous",
        "prompt": "帮我看看股票机会。",
        "expected_workflow": "ambiguous_request",
        "expected_ask": True,
    },
]


def run_audit() -> pd.DataFrame:
    rows = []
    for case in CASES:
        route = route_user_request(case["prompt"])
        guidance = render_route_guidance(route)
        rows.append(
            {
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "expected_workflow": case["expected_workflow"],
                "actual_workflow": route.workflow_id,
                "expected_ask": case["expected_ask"],
                "actual_ask": route.should_ask_user,
                "confidence": route.confidence,
                "codes": ";".join(route.extracted_codes),
                "status": "pass"
                if route.workflow_id == case["expected_workflow"] and route.should_ask_user == case["expected_ask"]
                else "fail",
                "guidance_preview": guidance[:500].replace("\n", "\\n"),
            }
        )
    return pd.DataFrame(rows)


def write_report(prefix: str, detail: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORT_DIR / f"{prefix}_detail.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    detail.to_csv(detail_path, index=False)
    fail_rows = int(detail["status"].ne("pass").sum()) if not detail.empty else 0
    lines = [
        "# User Intake Router Audit",
        "",
        "本地烟测，不调用外部 API、不读取密钥。目标是确认模糊用户问题先走选择题澄清，明确问题进入 P0/P1/盘中/策略研究对应工作流。",
        "",
        "## Verdict",
        "",
        f"- status: `{'pass' if fail_rows == 0 else 'fail'}`",
        f"- rows: `{len(detail)}`",
        f"- fail_rows: `{fail_rows}`",
        "",
        "## Detail",
        "",
        detail.to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        f"- `{detail_path}`",
        f"- `{report_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detail = run_audit()
    report_path = write_report(args.output_prefix, detail)
    print(f"wrote: {report_path}")
    print(detail.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
