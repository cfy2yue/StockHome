from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.io import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Check lightweight backtest local cache layout.")
    parser.add_argument("--universe", default="config/backtest_light_universe.yaml")
    parser.add_argument("--data-dir", default="data/backtest_light")
    parser.add_argument("--write-template", action="store_true", help="Write a universe template if --universe does not exist.")
    parser.add_argument("--fetch", action="store_true", help="Fetch BaoStock daily data into the local cache.")
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default="2026-06-24")
    args = parser.parse_args()
    universe_path = Path(args.universe)
    if args.write_template and not universe_path.exists():
        _write_template(universe_path)
        print("A股研究Agent")
        print(f"已生成股票池模板：{universe_path}")
        print("请补足 15 支训练股和 5 支 test 股后再运行预检。")
        return
    if not universe_path.exists():
        print("A股研究Agent")
        print(f"未找到股票池文件：{universe_path}")
        print("你可以先运行：")
        print(f"  .\\.venv\\Scripts\\python.exe scripts/prepare_backtest_light_data.py --universe {universe_path} --write-template")
        return
    universe = load_universe(args.universe)
    data_dir = Path(args.data_dir)
    missing = []
    warnings = []
    train_count = len(universe["train"])
    test_count = len(universe["test"])
    if train_count != 15:
        warnings.append(f"训练集当前 {train_count} 支，计划要求 15 支")
    if test_count != 5:
        warnings.append(f"test 集当前 {test_count} 支，计划要求 5 支")
    if args.fetch:
        _fetch_daily_cache(universe["train"] + universe["test"], data_dir, args.start_date, args.end_date)

    for item in universe["train"] + universe["test"]:
        code = str(item["code"])
        for filename in ["daily.csv", "financial.json"]:
            path = data_dir / code / filename
            if not path.exists():
                missing.append(str(path))
    print("A股研究Agent")
    if missing:
        print("以下本地缓存缺失，请先用白名单免费源准备数据，接口失败时跳过并报告：")
        for path in missing:
            print(f"- {path}")
    else:
        print("本地缓存结构检查通过，可以运行 scripts/run_backtest_light.py")
    if warnings:
        print("配置提醒：")
        for item in warnings:
            print(f"- {item}")


def _write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "train": [
            {
                "code": "600888",
                "name": "新疆众和",
                "board": "main",
                "sector_group": "nonferrous_materials",
                "industry": "铝",
                "selected_reason": "有色/材料训练样本示例；实际运行前需确认数据不少于 240 个交易日",
            }
        ],
        "test": [
            {
                "code": "688001",
                "name": "科创测试样本",
                "board": "star",
                "sector_group": "star_technology",
                "industry": "半导体/硬科技",
                "selected_reason": "test 集占位，只用于最终验收，请替换为真实股票",
            }
        ],
    }
    path.write_text(yaml.safe_dump(template, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _fetch_daily_cache(stocks: list[dict], data_dir: Path, start_date: str, end_date: str) -> None:
    try:
        import baostock as bs
    except Exception as exc:
        print(f"BaoStock 未安装或不可用，跳过拉取：{exc}")
        return
    login = bs.login()
    if login.error_code != "0":
        print(f"BaoStock 登录失败，跳过拉取：{login.error_msg}")
        return
    try:
        for stock in stocks:
            code = str(stock["code"])
            stock_dir = data_dir / code
            stock_dir.mkdir(parents=True, exist_ok=True)
            daily_path = stock_dir / "daily.csv"
            financial_path = stock_dir / "financial.json"
            try:
                df = _fetch_baostock_daily(bs, code, start_date, end_date)
                if df.empty:
                    print(f"- {code} 未获取到日线数据")
                    continue
                df.to_csv(daily_path, index=False, encoding="utf-8")
                if not financial_path.exists():
                    financial_path.write_text("[]\n", encoding="utf-8")
                metadata = {
                    "code": code,
                    "name": stock.get("name", ""),
                    "source": "BaoStock historical_structured 日K线",
                    "start_date": start_date,
                    "end_date": end_date,
                    "rows": int(len(df)),
                    "financial_note": "本脚本不猜测财报披露日；financial.json 默认为空，财务维度在 walk-forward 中标记缺失。",
                }
                (stock_dir / "metadata.json").write_text(yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"- {code} 日线缓存完成：{len(df)} 行")
            except Exception as exc:
                print(f"- {code} 拉取失败，已跳过：{exc}")
    finally:
        bs.logout()


def _fetch_baostock_daily(bs, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    market = "sh" if code.startswith(("5", "6", "9")) else "sz"
    fields = "date,code,open,high,low,close,volume,amount,pctChg"
    rs = bs.query_history_k_data_plus(
        f"{market}.{code}",
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(dict(zip(fields.split(","), rs.get_row_data())))
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"pctChg": "pct_chg"})
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
    return df.dropna(subset=["date", "close"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
