from src.data.multisource_adapter import results_to_markdown
from src.data.schemas import FetchResult
import yaml


def test_multisource_markdown_mentions_source_tiers():
    text = results_to_markdown(
        {
            "quote_realtime": FetchResult(True, "mootdx quote_protocol 通达信行情", data=[{"price": 1}], fetched_at="2026-06-23"),
            "stock_news": FetchResult(True, "AKShare public_aggregator 个股新闻", data=[{"title": "x"}], fetched_at="2026-06-23"),
        }
    )
    assert text.startswith("A股研究Agent")
    assert "quote_protocol" in text
    assert "public_aggregator" in text
    assert "不是交易所直连原始逐笔数据" in text


def test_source_tiers_use_expected_names():
    with open("config/source_tiers.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tiers = data["tiers"]
    for name in [
        "quote_protocol",
        "historical_structured",
        "public_aggregator",
        "paid_standardized",
        "official_disclosure",
        "model_estimate",
        "cache",
    ]:
        assert name in tiers
