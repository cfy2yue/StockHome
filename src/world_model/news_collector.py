from __future__ import annotations

from src.data.schemas import FetchResult, now_text


def collect_stock_news(code: str, name: str, dry_run: bool = False) -> FetchResult:
    if dry_run:
        return FetchResult(
            True,
            "dry-run 新闻样例",
            [
                {
                    "事件标题": f"{name} 新闻模块 dry-run 样例",
                    "来源": "本地样例",
                    "时间": now_text(),
                    "关联股票": name,
                    "关联行业": "有色金属",
                    "事件类型": "不确定",
                    "影响层级": "公司",
                    "是否改变原逻辑": "不确定",
                    "需要人工验证的信息": "正式运行时需拉取公开新闻接口并核对来源。",
                }
            ],
            fetched_at=now_text(),
        )
    try:
        import akshare as ak

        df = ak.stock_news_em(symbol=code)
        rows = df.head(10).to_dict("records")
        return FetchResult(True, "AKShare stock_news_em", rows, fetched_at=now_text())
    except Exception as exc:
        return FetchResult(False, "AKShare stock_news_em", [], error=str(exc), warning="新闻模块：当前公开接口获取失败，本轮新闻信息不足；已跳过，不影响财务和行情分析。", fetched_at=now_text())
