from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.backtest.adaptation import build_adaptation_skills
from src.backtest.case_memory import build_case_memory
from src.backtest.gate_optimizer import write_gate_optimization_report
from src.backtest.io import load_universe, load_weights, write_yaml
from src.backtest.news_report import write_news_feature_report
from src.backtest.pattern_analysis import write_pattern_report
from src.backtest.pool_optimizer import write_pool_optimizer_report
from src.backtest.pool_selection import write_pool_selection_report
from src.backtest.pool_walkforward import write_pool_walkforward_report
from src.backtest.rebound_diagnostics import write_rebound_diagnostics_report
from src.backtest.rebound_validation import write_rebound_validation_report
from src.backtest.reporting import write_summary_report
from src.backtest.reusable_rules import build_reusable_rule_candidates
from src.backtest.scoring import DEFAULT_WEIGHTS
from src.backtest.strategy_compare import write_strategy_comparison
from src.backtest.tree_gate import write_tree_gate_report
from src.backtest.weight_optimizer import optimize_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight A-share research backtest.")
    parser.add_argument("--universe", default="config/backtest_light_universe.yaml")
    parser.add_argument("--data-dir", default="data/backtest_light")
    parser.add_argument("--weights", default="config/weights.yaml")
    parser.add_argument("--output", default="reports/backtest_light")
    parser.add_argument("--keep-details", action="store_true", help="Write per-stock decision and ground-truth files.")
    args = parser.parse_args()

    universe = load_universe(args.universe)
    train = [{**item, "set": "train"} for item in universe["train"]]
    test = [{**item, "set": "test"} for item in universe["test"]]
    output = Path(args.output)
    weights = _load_weights_or_default(args.weights)

    epoch1 = run_backtest(train, args.data_dir, weights, output, "epoch1", keep_details=args.keep_details)
    weights_epoch1 = optimize_weights(epoch1["ground_truth"], weights)
    write_yaml(output / "weights_epoch1.yaml", weights_epoch1)

    epoch2 = run_backtest(train, args.data_dir, weights_epoch1, output, "epoch2", keep_details=args.keep_details)
    final_weights = optimize_weights(epoch2["ground_truth"], weights_epoch1)
    write_yaml(output / "final_weights.yaml", final_weights)

    test_result = run_backtest(test, args.data_dir, final_weights, output, "test", keep_details=args.keep_details) if test else {"ground_truth": None}
    build_reusable_rule_candidates(
        epoch1["ground_truth"],
        epoch2["ground_truth"],
        test_result["ground_truth"],
        output / "reusable_rules",
    )
    build_adaptation_skills(output / "final_reusable_rules.yaml", output / "adaptation_skills.yaml")
    build_case_memory(output / "epoch2" / "ground_truth.csv", output / "case_memory.csv")
    write_pattern_report(output)
    write_strategy_comparison(output)
    write_gate_optimization_report(output, args.data_dir)
    write_tree_gate_report(output)
    write_news_feature_report(output)
    write_pool_selection_report(output)
    write_pool_optimizer_report(output)
    write_pool_walkforward_report(output)
    write_rebound_diagnostics_report(output)
    write_rebound_validation_report(output)
    write_summary_report(output, epoch1["ground_truth"], epoch2["ground_truth"], test_result["ground_truth"])
    print("A股研究Agent")
    print(f"轻量回测完成，报告目录：{output}")


def _load_weights_or_default(path: str) -> dict[str, float]:
    target = Path(path)
    if not target.exists():
        return DEFAULT_WEIGHTS.copy()
    return load_weights(target)


if __name__ == "__main__":
    main()
