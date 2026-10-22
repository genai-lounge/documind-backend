"""Vector similarity, as plain math."""

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """How closely two vectors point the same way: 1 same, 0 unrelated, -1 opposite.

    Only the direction matters, not the length, so scaling a vector doesn't
    change the result.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors have different lengths: {len(a)} and {len(b)}.")
    norm_a = math.sqrt(math.sumprod(a, a))
    norm_b = math.sqrt(math.sumprod(b, b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    similarity = math.sumprod(a, b) / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))
