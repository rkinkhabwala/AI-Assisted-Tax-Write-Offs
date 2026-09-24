"""PII redaction for anything that leaves the process or lands in a log (spec section 9).

Taxpayer identifiers are the PII most likely to appear in a tax question: SSNs and ITINs
(123-45-6789) and EINs (12-3456789). The agent's PreToolUse hook (phase 6) reuses this.
"""

import re

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b(?<!\d)\d{9}(?!\d)\b")
_EIN = re.compile(r"\b\d{2}-\d{7}\b")


def redact(text: str) -> str:
    """Replace SSN/ITIN and EIN patterns with placeholders."""
    return _EIN.sub("[EIN]", _SSN.sub("[SSN]", text))
