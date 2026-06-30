from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.preflight import run_preflight, write_preflight_reports


def main() -> None:
    report = run_preflight(ROOT)
    md_path, json_path = write_preflight_reports(report, ROOT / "reports" / "date_generalization")
    print("A股研究Agent")
    print(f"preflight ok: {report['ok']}")
    print(f"wrote: {md_path}")
    print(f"wrote: {json_path}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
