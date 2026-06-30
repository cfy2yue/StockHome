from __future__ import annotations

from src.data.schemas import FetchResult, now_text


def collect_market_regime(dry_run: bool = False) -> FetchResult:
    if dry_run:
        return FetchResult(True, "dry-run 市场环境样例", {"summary": "dry-run 未判断真实市场环境", "risk": "市场状态信息不足"}, fetched_at=now_text())
    # 优先用 mootdx 获取上证指数实时快照
    try:
        from mootdx.quotes import Quotes
        q = Quotes.factory(market="std")
        try:
            df = q.quotes(symbol=["000001"])
            if not df.empty:
                row = df.iloc[0]
                data = {
                    "指数": "上证指数",
                    "最新价": float(row.get("price", 0)),
                    "昨收": float(row.get("last_close", 0)),
                    "开盘": float(row.get("open", 0)),
                    "最高": float(row.get("high", 0)),
                    "最低": float(row.get("low", 0)),
                    "成交量": int(row.get("vol", 0)),
                }
                return FetchResult(True, "mootdx 上证指数", data, fetched_at=now_text())
        finally:
            try:
                q.close()
            except Exception:
                pass
    except Exception as exc:
        pass
    # 回退 AKShare
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        return FetchResult(True, "AKShare 上证指数", df.tail(120).to_dict("records"), fetched_at=now_text())
    except Exception as exc:
        return FetchResult(True, "AKShare 上证指数", {"summary": "市场环境接口失败", "risk": "市场状态信息不足"}, error=str(exc), warning="市场环境接口失败，已降级", fetched_at=now_text())
