from __future__ import annotations

from src.data.schemas import FetchResult, now_text


def collect_sector_events(sector: str | None = None, dry_run: bool = False) -> FetchResult:
    if dry_run:
        return FetchResult(True, "dry-run 行业事件样例", [], warning="dry-run 未拉取行业新闻", fetched_at=now_text())
    try:
        import akshare as ak
        # 使用正确的 CNInfo 行业市盈率接口
        df = ak.stock_industry_pe_ratio_cninfo()
        rows = df.head(20).to_dict("records") if df is not None and not df.empty else []
        return FetchResult(True, "AKShare stock_industry_pe_ratio_cninfo", rows, fetched_at=now_text())
    except Exception as exc:
        return FetchResult(True, "AKShare stock_industry_pe_ratio_cninfo", [], error=str(exc), warning="行业事件接口暂不可用，已降级。", fetched_at=now_text())
