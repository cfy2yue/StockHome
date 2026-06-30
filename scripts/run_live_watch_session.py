from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_watch import LiveWatchConfig, LiveWatchSession, render_live_watch_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cache-aware single-stock live watch research loop.")
    parser.add_argument("--code", required=True, help="Six-digit A-share code.")
    parser.add_argument("--name", default="", help="Optional stock name for display.")
    parser.add_argument("--interval-seconds", type=int, default=1200, help="Decision interval. Default: 1200 seconds.")
    parser.add_argument("--max-iterations", type=int, default=1, help="Number of decision points to run.")
    parser.add_argument("--news-cache-ttl-hours", type=int, default=24, help="Refresh news/announcement context at most once per TTL.")
    parser.add_argument("--intraday-frequency", default="5m", choices=["1m", "5m", "15m", "30m", "60m"])
    parser.add_argument("--intraday-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LiveWatchConfig(
        code=args.code,
        name=args.name,
        interval_seconds=args.interval_seconds,
        max_iterations=args.max_iterations,
        news_cache_ttl_hours=args.news_cache_ttl_hours,
        intraday_frequency=args.intraday_frequency,
        intraday_limit=args.intraday_limit,
    )
    session = LiveWatchSession(config)
    for decision in session.run():
        print(render_live_watch_markdown(decision))


if __name__ == "__main__":
    main()
