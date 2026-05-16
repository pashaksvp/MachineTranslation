"""Akkadian to English machine translation package."""

__all__ = ["MyTranslatorModel"]


def __getattr__(name: str):
    if name == "MyTranslatorModel":
        from akkadian_mt.translator import MyTranslatorModel

        return MyTranslatorModel
    raise AttributeError(name)
