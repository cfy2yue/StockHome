from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multisource_adapter import MultiSourceDataAdapter, results_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="A股研究Agent 多源数据输入流 smoke test")
    parser.add_argument("--code", default="600888")
    parser.add_argument("--output", default="reports/latest/multisource_data_smoke.md")
    args = parser.parse_args()
    adapter = MultiSourceDataAdapter()
    results = adapter.smoke_all(args.code)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results_to_markdown(results), encoding="utf-8")
    print("A股研究Agent")
    print()
    print(f"多源数据 smoke test 已生成：{out}")


if __name__ == "__main__":
    main()
