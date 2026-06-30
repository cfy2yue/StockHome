from src.data.akshare_adapter import AKShareAdapter


def test_resolve_xinjiang_fixture():
    adapter = AKShareAdapter(dry_run=True)
    result = adapter.resolve_stock("新疆合众")
    assert result.ok
    assert result.data[0]["代码"] == "600888"
