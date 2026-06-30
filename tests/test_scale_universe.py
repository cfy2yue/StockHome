from scripts.build_backtest_scale_universe import _board, _is_supported_code, _sector_group


def test_scale_universe_code_filters_and_board_tags():
    assert _is_supported_code("688981")
    assert _is_supported_code("000001")
    assert not _is_supported_code("900001")
    assert _board("688981") == "star"
    assert _board("300750") == "chinext"
    assert _sector_group("688981") == "star_technology"
