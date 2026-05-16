from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_LACUNA_RE = re.compile(r"\[[^\]]*\]")
_ANGLE_RE = re.compile(r"<([^>]*)>")
_BROKEN_RE = re.compile(r"[⸢⸣]")
_BRACES_RE = re.compile(r"\{([^}]*)\}")
_SIGN_INDEX_RE = re.compile(r"([A-Za-zšṣṭŠṢṬāēīūĀĒĪŪḫḪĝĜŋŊ]+)([₀₁₂₃₄₅₆₇₈₉0-9]+)")

_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_akkadian(
    text: str,
    *,
    strip_lacunae: bool = True,
    keep_diacritics: bool = True,
    keep_sign_indices: bool = True,
    lowercase: bool = True,
) -> str:
    """Normalize Akkadian transliteration for reproducible experiments.

    The defaults follow the recommended ByT5 path: keep diacritics and sign indices,
    but normalize Unicode, whitespace, and common markup.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_SUBSCRIPT_DIGITS)
    text = _BROKEN_RE.sub("", text)
    text = _ANGLE_RE.sub(r"\1", text)
    text = _BRACES_RE.sub(r" {\1} ", text)

    if strip_lacunae:
        text = _LACUNA_RE.sub(" <lacuna> ", text)

    if not keep_sign_indices:
        text = _SIGN_INDEX_RE.sub(r"\1", text)

    if not keep_diacritics:
        replacements = {
            "š": "s",
            "ṣ": "s",
            "ṭ": "t",
            "ā": "a",
            "ē": "e",
            "ī": "i",
            "ū": "u",
            "Š": "S",
            "Ṣ": "S",
            "Ṭ": "T",
            "Ā": "A",
            "Ē": "E",
            "Ī": "I",
            "Ū": "U",
        }
        text = "".join(replacements.get(ch, ch) for ch in text)

    if lowercase:
        text = text.lower()

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
