"""Agreement metrics."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def percent_agreement(unit_values: Iterable[Sequence[str | None]]) -> float:
    """Compute percent agreement across units."""
    total = 0
    agree = 0
    for values in unit_values:
        observed = [value for value in values if value is not None]
        if not observed:
            continue
        total += 1
        if len(set(observed)) == 1:
            agree += 1
    return agree / total if total else 0.0


def cohens_kappa(pairs: Sequence[tuple[str | None, str | None]]) -> float:
    """Compute Cohen's kappa for matched reviewer pairs."""
    filtered = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not filtered:
        return 0.0
    total = len(filtered)
    categories = sorted({value for pair in filtered for value in pair})
    agreement = sum(1 for a, b in filtered if a == b) / total
    counts_a = Counter(a for a, _ in filtered)
    counts_b = Counter(b for _, b in filtered)
    pe = sum((counts_a[cat] / total) * (counts_b[cat] / total) for cat in categories)
    if pe == 1.0:
        return 1.0
    return (agreement - pe) / (1 - pe)


def fleiss_kappa(matrix: list[list[int]]) -> float:
    """Compute Fleiss' kappa given matrix counts per category per subject."""
    if not matrix:
        return 0.0
    n = len(matrix)
    k = len(matrix[0])
    totals = [0] * k
    for row in matrix:
        for j, count in enumerate(row):
            totals[j] += count
    n_raters = sum(matrix[0]) if matrix else 0
    if n_raters == 0:
        return 0.0
    p_j = [total / (n * n_raters) for total in totals]
    p_i = []
    for row in matrix:
        row_total = sum(row)
        if row_total == 0:
            p_i.append(0.0)
            continue
        agreement = sum(count * (count - 1) for count in row)
        p_i.append(agreement / (row_total * (row_total - 1)))
    p_bar = sum(p_i) / n
    p_e = sum(p**2 for p in p_j)
    if p_e == 1:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)
