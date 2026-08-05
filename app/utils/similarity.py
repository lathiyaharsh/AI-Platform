from math import sqrt


def cosine_similarity(a, b) -> float:
    """
    Cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sqrt(sum(x * x for x in a))
    mag_b = sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)
