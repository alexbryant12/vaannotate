"""Agreement statistics used by the Admin adjudication workflows."""
from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple


def percent_agreement(labels: Sequence[Sequence[str]]) -> float:
    """Compute simple percent agreement across raters for categorical labels."""
    if not labels:
        return 0.0
    total = len(labels)
    matches = sum(1 for row in labels if len(set(row)) == 1)
    return matches / total if total else 0.0


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError("Raters must score the same number of units")
    n = len(rater_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    pa = agree / n
    categories = set(rater_a) | set(rater_b)
    pa_exp = sum((rater_a.count(cat) / n) * (rater_b.count(cat) / n) for cat in categories)
    if pa_exp == 1:
        return 1.0
    return (pa - pa_exp) / (1 - pa_exp)


def fleiss_kappa(matrix: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa for more than two raters.

    The ``matrix`` contains counts per category per item.  Each row sums to the
    number of raters who labeled that item.
    """
    if not matrix:
        return 0.0
    n = len(matrix)
    m = sum(matrix[0])
    if n == 0 or m == 0:
        return 0.0
    p_i = []
    category_totals = [0] * len(matrix[0])
    for row in matrix:
        row_sum = sum(row)
        if row_sum == 0:
            continue
        p_i.append((sum(val * val for val in row) - row_sum) / (row_sum * (row_sum - 1)))
        for idx, val in enumerate(row):
            category_totals[idx] += val
    p_bar = sum(p_i) / len(p_i)
    p_e = sum((total / (n * m)) ** 2 for total in category_totals)
    if p_e == 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)
