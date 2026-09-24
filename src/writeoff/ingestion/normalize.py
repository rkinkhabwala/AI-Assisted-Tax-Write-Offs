"""Text normalization applied to every block before chunking and hashing.

Deterministic, so that unchanged sources hash identically across runs. NFKC folds
non-breaking spaces and ligatures; typographic quotes are straightened so lexical search
matches "aren't" however the source typed it. Section signs and dashes are preserved.
"""

import re
import unicodedata

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_BLANK_LINES = re.compile(r"\n{3,}")
_SOFT_HYPHEN = chr(0x00AD)
_ZERO_WIDTH = re.compile("[" + "".join(map(chr, (0x200B, 0x200C, 0x200D, 0xFEFF))) + "]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = _ZERO_WIDTH.sub("", text).translate(_QUOTES)
    text = _INLINE_SPACE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text).strip()
