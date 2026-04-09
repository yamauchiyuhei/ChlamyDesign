"""Codon Pair Score (CPS) — Coleman et al. 2008.

Phase 2 of the ChlamyDesign knowledge-driven framework. The Codon Pair Score
captures bias in adjacent-codon usage that is independent of single-codon
frequency, and reflects the limited tRNA pool of the *Chlamydomonas* chloroplast
(~30 tRNA species). Higher CPS = more "natural" codon-pair usage = expected to
be translated more efficiently.

Definition (Coleman et al., Science 2008):

    CPS(AB) = ln( F(AB) / [ (F(A) F(B)) / (F(X) F(Y)) * F(XY) ] )

where A, B are codons; X, Y are the amino acids encoded by A, B; F(*) denotes
observed frequency in the reference corpus.

A sequence's CPS score is the arithmetic mean of CPS(AB) over all adjacent
codon pairs in the CDS (excluding stop codons).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from Bio.Seq import Seq

# Re-use the standard genetic code via Biopython once at import time.
_CODON_TO_AA: dict[str, str] = {}
for _i in "TCAG":
    for _j in "TCAG":
        for _k in "TCAG":
            _c = _i + _j + _k
            _CODON_TO_AA[_c] = str(Seq(_c).translate())


def _split_codons(dna: str) -> list[tuple[str, str]]:
    """Split DNA into (codon, aa) tuples, dropping stop codons and frame remainder."""
    n = len(dna) - (len(dna) % 3)
    out: list[tuple[str, str]] = []
    for i in range(0, n, 3):
        c = dna[i : i + 3]
        if len(c) != 3 or any(b not in "ACGT" for b in c):
            continue
        aa = _CODON_TO_AA.get(c, "X")
        if aa == "*" or aa == "X":
            continue
        out.append((c, aa))
    return out


def compute_cps_table(dna_list: Iterable[str]) -> dict[str, float]:
    """Compute the CPS lookup table from a reference corpus.

    Args:
        dna_list: iterable of CDS strings (ATG-starting recommended; stop codons
            and partial frames are dropped automatically).

    Returns:
        Dict mapping concatenated codon-pair strings (length 6) to CPS values.
    """
    codon_count: dict[str, int] = defaultdict(int)
    aa_count: dict[str, int] = defaultdict(int)
    pair_count: dict[str, int] = defaultdict(int)
    aa_pair_count: dict[str, int] = defaultdict(int)
    total_codons = 0
    total_pairs = 0

    for dna in dna_list:
        valid = _split_codons(dna)
        for c, aa in valid:
            codon_count[c] += 1
            aa_count[aa] += 1
            total_codons += 1
        for i in range(len(valid) - 1):
            c1, a1 = valid[i]
            c2, a2 = valid[i + 1]
            pair_count[c1 + c2] += 1
            aa_pair_count[a1 + a2] += 1
            total_pairs += 1

    if total_codons == 0 or total_pairs == 0:
        return {}

    cps_table: dict[str, float] = {}
    for pair, count in pair_count.items():
        c1, c2 = pair[:3], pair[3:]
        a1 = _CODON_TO_AA.get(c1)
        a2 = _CODON_TO_AA.get(c2)
        if not a1 or not a2:
            continue
        f_ab = count / total_pairs
        f_a = codon_count[c1] / total_codons
        f_b = codon_count[c2] / total_codons
        f_x = aa_count[a1] / total_codons
        f_y = aa_count[a2] / total_codons
        f_xy = aa_pair_count[a1 + a2] / total_pairs
        if f_a <= 0 or f_b <= 0 or f_x <= 0 or f_y <= 0 or f_xy <= 0:
            continue
        denom = (f_a * f_b / (f_x * f_y)) * f_xy
        if denom <= 0:
            continue
        cps_table[pair] = math.log(f_ab / denom)

    return cps_table


def score_cps(dna: str, cps_table: dict[str, float]) -> float:
    """Mean CPS over adjacent codon pairs in a CDS.

    Returns 0.0 if the sequence is too short or no pairs are found in the table.
    Higher = more natural codon-pair usage.
    """
    if not cps_table:
        return 0.0
    valid = _split_codons(dna)
    if len(valid) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(valid) - 1):
        pair = valid[i][0] + valid[i + 1][0]
        v = cps_table.get(pair)
        if v is not None:
            scores.append(v)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
