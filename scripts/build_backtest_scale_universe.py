from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible broad A-share backtest universe from free sources.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--output", default="config/backtest_scale_universe.yaml")
    parser.add_argument("--cache", default="data/backtest_scale/a_share_codes.csv")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    stocks = _load_a_share_codes(args.limit, Path(args.cache), args.refresh_cache)
    if not stocks:
        raise SystemExit("未获取到股票代码，无法生成 universe。")
    test_count = max(1, int(len(stocks) * args.test_ratio))
    train = stocks[:-test_count]
    test = stocks[-test_count:]
    data = {
        "meta": {
            "source": "AkShare stock_info_a_code_name 免费接口",
            "limit": len(stocks),
            "test_ratio": args.test_ratio,
            "split_rule": "过滤后按代码稳定排序，尾部作为 test；test 不参与调权或 gate 选择",
        },
        "train": train,
        "test": test,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("A股研究Agent")
    print(f"已生成宽股票池：{target}，train={len(train)}，test={len(test)}")


def _load_a_share_codes(limit: int, cache_path: Path | None = None, refresh_cache: bool = False) -> list[dict[str, str]]:
    import akshare as ak

    if cache_path and cache_path.exists() and not refresh_cache:
        df = pd.read_csv(cache_path, dtype={"code": str})
    else:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                df = ak.stock_info_a_code_name()
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
                break
            except Exception as exc:  # pragma: no cover - depends on public network behavior.
                last_error = exc
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"AkShare 股票列表获取失败：{last_error}") from last_error
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str)
    df = df[df["code"].map(_is_supported_code)]
    df = df[~df["name"].str.contains("ST|退|B", regex=True, na=False)]
    df = df.drop_duplicates("code").sort_values("code").head(limit)
    rows = []
    for _, row in df.iterrows():
        code = str(row["code"])
        rows.append(
            {
                "code": code,
                "name": str(row["name"]),
                "board": _board(code),
                "sector_group": _sector_group(code),
                "industry": "unknown",
                "selected_reason": "宽样本自动生成；行业字段待免费源补充",
            }
        )
    return rows


def _is_supported_code(code: str) -> bool:
    return code.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688"))


def _board(code: str) -> str:
    if code.startswith("688"):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main"


def _sector_group(code: str) -> str:
    if code.startswith("688"):
        return "star_technology"
    if code.startswith(("300", "301")):
        return "growth_technology"
    return "broad_a_share"


if __name__ == "__main__":
    main()
