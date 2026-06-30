from __future__ import annotations

from src.data.schemas import FetchResult, now_text


def collect_announcements(code: str, name: str, dry_run: bool = False) -> FetchResult:
    if dry_run:
        return FetchResult(True, "dry-run 公告样例", [], warning="dry-run 未拉取公告原文", fetched_at=now_text())
    try:
        import akshare as ak
        # 修正：AKShare 1.18+ 参数签名为 (security, symbol='全部', begin_date, end_date)
        df = ak.stock_individual_notice_report(security=code, symbol="全部")
        return FetchResult(True, "AKShare stock_individual_notice_report", df.head(10).to_dict("records"), fetched_at=now_text())
    except TypeError as exc:
        # 回退旧版参数签名
        try:
            import akshare as ak
            df = ak.stock_individual_notice_report(symbol=code)
            return FetchResult(True, "AKShare stock_individual_notice_report (legacy)", df.head(10).to_dict("records"), fetched_at=now_text())
        except Exception as exc2:
            return FetchResult(True, "AKShare stock_individual_notice_report", [], error=str(exc2), warning="公告模块接口参数兼容失败，需人工核对交易所公告。", fetched_at=now_text())
    except Exception as exc:
        return FetchResult(True, "AKShare stock_individual_notice_report", [], error=str(exc), warning="公告模块超时或失败，已降级，需人工核对交易所公告。", fetched_at=now_text())
