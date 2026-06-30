from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.gt_maturity import MaturityConfig, maturity_report


GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gt_pending/is_provisional maturity report.")
    parser.add_argument("--current-date", default="2026-06-25")
    parser.add_argument("--output", default=str(ROOT / "reports" / "date_generalization" / "gt_maturity_report.csv"))
    args = parser.parse_args()
    frames = [pd.read_csv(path, low_memory=False) for path in GT_SOURCES if path.exists()]
    if not frames:
        raise FileNotFoundError("missing backtest_scale_500 ground_truth sources")
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    report = maturity_report(frame, MaturityConfig(current_date=args.current_date))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print("A股研究Agent")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
