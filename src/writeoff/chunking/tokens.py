"""Token estimation for chunk sizing.

Neither Voyage nor Claude tokenizers ship offline in our dependency set, and chunk sizing
must be deterministic and free. We therefore estimate conservatively at 3.5 characters per
token. Typical English prose runs about 4 characters per token, so this overestimates prose
by roughly 15% and keeps real sizes under the 1,200-token hard max. Statutory text dense
with "(c)(1)(A)" references tokenizes closer to 3.5, the case the constant is set for.
Calibrate against the provider's token counts once API access is available.
"""

import math

CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
