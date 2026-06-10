from akkadian_mt.translator import MyTranslatorModel


def test_first_non_empty_selector_skips_blank_candidates() -> None:
    translator = MyTranslatorModel(selector_strategy="first_non_empty")

    assert translator._select_candidate(["", "first translation", "second translation"]) == (
        "first translation"
    )


def test_longest_selector_prefers_more_words() -> None:
    translator = MyTranslatorModel(selector_strategy="longest")

    assert translator._select_candidate(["short", "a longer translation"]) == "a longer translation"

