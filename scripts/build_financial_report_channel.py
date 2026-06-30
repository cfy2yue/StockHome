from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tushare_pro_adapter import DEFAULT_CACHE_DIR  # noqa: E402
from src.world_model.financial_report_channel import build_financial_report_outputs, merge_financial_report_features_asof  # noqa: E402


DEFAULT_MARKET_DIR = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_REPORT_DIR = ROOT / "reports" / "date_generalization"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disclosure-date-safe financial report event channel from local offline caches.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--market-dir", default=str(DEFAULT_MARKET_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--skip-asof-audit", action="store_true")
    args = parser.parse_args()

    events, features = build_financial_report_outputs(args.cache_dir, market_dir=args.market_dir, report_dir=args.report_dir)
    asof_path = None if args.skip_asof_audit else _write_asof_audit(Path(args.market_dir), Path(args.report_dir), features)
    _update_market_manifest(Path(args.market_dir))
    print("A股研究Agent")
    print(f"financial_report_events={len(events)}")
    print(f"financial_report_features={len(features)}")
    print(f"wrote={Path(args.market_dir) / 'financial_report_events.csv'}")
    print(f"report={Path(args.report_dir) / 'financial_report_channel_coverage.md'}")
    if asof_path:
        print(f"asof_audit={asof_path}")


def _write_asof_audit(market_dir: Path, report_dir: Path, features: pd.DataFrame) -> Path | None:
    joined_path = market_dir / "joined_ground_truth_combined_news.csv"
    if not joined_path.exists() or features.empty:
        return None
    gt = pd.read_csv(joined_path, dtype={"code": str}, low_memory=False, usecols=lambda column: column in {"date", "code"})
    gt["code"] = gt["code"].astype(str).str.zfill(6)
    merged = merge_financial_report_features_asof(gt, features, window_days=90)
    matched = merged["financial_report_join_status"].astype(str).eq("event_window_matched")
    merged["_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    blocks = {
        "H2023_1": ("2023-01-01", "2023-06-30"),
        "H2023_2": ("2023-07-01", "2023-12-31"),
        "H2024_1": ("2024-01-01", "2024-06-30"),
        "H2024_2": ("2024-07-01", "2024-12-31"),
        "H2025_1": ("2025-01-01", "2025-06-30"),
        "H2025_2": ("2025-07-01", "2025-12-31"),
        "H2026_1": ("2026-01-01", "2026-06-30"),
    }
    rows = []
    for block, (start, end) in blocks.items():
        selector = (merged["_dt"] >= pd.Timestamp(start)) & (merged["_dt"] <= pd.Timestamp(end))
        rows.append(
            {
                "block": block,
                "rows": int(selector.sum()),
                "matched": int((selector & matched).sum()),
                "matched_rate": round(float((selector & matched).sum() / selector.sum()), 4) if selector.sum() else 0.0,
                "matched_stocks": int(merged.loc[selector & matched, "code"].nunique()),
            }
        )
    audit = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "financial_report_channel_asof90_coverage.csv"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path = report_dir / "financial_report_channel_coverage.md"
    if md_path.exists():
        total_line = f"- asof90_matched_rows: `{int(matched.sum())}` / `{len(merged)}` = `{round(float(matched.mean()), 4)}`"
        with md_path.open("a", encoding="utf-8") as handle:
            handle.write("\n## 90-Day As-Of GT Coverage\n\n")
            handle.write(total_line + "\n\n")
            handle.write(audit.to_markdown(index=False) + "\n")
            handle.write("\n该覆盖率表示当前本地缓存能在决策日前 90 天窗口内看到财报事件的 GT 行比例；它不是全市场财务覆盖率。\n")
    return csv_path


def _update_market_manifest(market_dir: Path) -> None:
    manifest_path = market_dir / "cache_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    files = manifest.setdefault("files", {})
    for name in ["financial_report_events.csv", "financial_report_features.csv", "financial_report_schema.csv"]:
        path = market_dir / name
        if path.exists():
            try:
                files[name] = max(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1, 0)
            except OSError:
                pass
    manifest["financial_report_channel_available"] = True
    manifest["financial_report_time_policy"] = "date-only disclosures become available at next natural day 00:00:00"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
