from akkadian_mt.normalization import normalize_akkadian


def test_normalize_markup_and_subscript_digits() -> None:
    assert normalize_akkadian("⸢Šarrum⸣ ma₂ [x]") == "šarrum ma2 <lacuna>"
