from math import sqrt


def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    mag_a = sqrt(sum(x * x for x in a))

    mag_b = sqrt(sum(x * x for x in b))

    return dot / (mag_a * mag_b)