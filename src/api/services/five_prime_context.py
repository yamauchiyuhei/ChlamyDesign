"""5' CDS context bias — Phase 3 of the ChlamyDesign framework.

In *Chlamydomonas* chloroplast, nuclear-encoded translational activators
(NAC2, MBB1, RBP40, TBC1/2/3, MCA1, MRL1, …) recognize binding sites that
in many cases extend several codons into the CDS. Naively replacing these
codons with the most-frequent codon disrupts activator binding and abolishes
translation, even when the encoded amino acids are identical.

To preserve this *conserved 5' context* without hard-coding individual
binding motifs, we compute a position-independent log-odds bias from the
first FIVE_PRIME_CONTEXT_CODONS codons of high-expression chloroplast genes
versus the codon usage across the full chloroplast proteome:

    bias(codon | aa) = log( P_5prime(codon | aa) / P_full(codon | aa) )

A candidate sequence's 5' context score is the sum of bias values over its
first FIVE_PRIME_CONTEXT_CODONS codons (skipping the invariant start codon).
Higher = more similar to natural high-expression starts → higher chance of
preserving activator-binding context.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from Bio.Seq import Seq

# Curated list of highly-expressed Chlamydomonas chloroplast genes whose
# transcripts are known to be subject to nuclear-encoded translational
# activator regulation in the literature (PSII core, PSI core, ATP synthase,
# Rubisco LSU, cytochrome b6f).
HIGH_EXPRESSION_GENES: tuple[str, ...] = (
    "rbcL",
    "psbA", "psbB", "psbC", "psbD", "psbE",
    "psaA", "psaB",
    "atpA", "atpB", "atpH",
    "petA",
)

# Number of CDS-internal codons to score (excluding the start codon itself).
FIVE_PRIME_CONTEXT_CODONS = 10

# Pseudocount to avoid log(0) for codons unobserved in the 5' window.
_PSEUDO = 0.5


_CODON_TO_AA: dict[str, str] = {}
for _i in "TCAG":
    for _j in "TCAG":
        for _k in "TCAG":
            _c = _i + _j + _k
            _CODON_TO_AA[_c] = str(Seq(_c).translate())


def _codons(dna: str) -> list[str]:
    n = len(dna) - (len(dna) % 3)
    return [dna[i : i + 3] for i in range(0, n, 3)]


def _aa_codon_counts(
    dna_list: Iterable[str], n_codons: int | None
) -> dict[str, dict[str, float]]:
    """Count codon occurrences per amino acid.

    If `n_codons` is given, only the codons in positions 1..n_codons (after the
    start codon at position 0) are counted; otherwise all codons in the CDS
    body are counted (start and stop codons excluded).
    """
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dna in dna_list:
        cds = _codons(dna)
        if len(cds) < 2:
            continue
        # Skip start codon (position 0). Drop trailing stop if present.
        body = cds[1:]
        if body and _CODON_TO_AA.get(body[-1]) == "*":
            body = body[:-1]
        if n_codons is not None:
            body = body[:n_codons]
        for c in body:
            aa = _CODON_TO_AA.get(c)
            if not aa or aa == "*":
                continue
            counts[aa][c] += 1.0
    return counts


def compute_five_prime_bias(
    high_expr_dnas: Iterable[str],
    all_dnas: Iterable[str],
    n_codons: int = FIVE_PRIME_CONTEXT_CODONS,
) -> dict[str, float]:
    """Build the 5' codon bias table.

    Returns a dict mapping codon → log-odds bias for being used in the 5'
    region of high-expression genes (vs the chloroplast-wide background).
    Codons that never appear in either set are omitted.
    """
    high_expr_dnas = list(high_expr_dnas)
    all_dnas = list(all_dnas)
    if not high_expr_dnas or not all_dnas:
        return {}

    five = _aa_codon_counts(high_expr_dnas, n_codons=n_codons)
    full = _aa_codon_counts(all_dnas, n_codons=None)

    bias: dict[str, float] = {}
    for aa, codon_counts in full.items():
        full_total = sum(codon_counts.values())
        five_codon_counts = five.get(aa, {})
        five_total = sum(five_codon_counts.values())
        if full_total <= 0 or five_total <= 0:
            continue
        # Pseudocount denominator: pseudo per observed codon for this AA.
        n_codons_for_aa = len(codon_counts)
        for codon, full_count in codon_counts.items():
            p_full = (full_count + _PSEUDO) / (full_total + _PSEUDO * n_codons_for_aa)
            p_five = (five_codon_counts.get(codon, 0.0) + _PSEUDO) / (
                five_total + _PSEUDO * n_codons_for_aa
            )
            bias[codon] = math.log(p_five / p_full)

    return bias


def score_five_prime_context(
    dna: str,
    bias_table: dict[str, float],
    n_codons: int = FIVE_PRIME_CONTEXT_CODONS,
) -> float:
    """Sum bias scores over the first n_codons of the CDS (excluding start).

    Returns 0.0 if bias_table is empty or sequence is too short. Higher score
    means the candidate's 5' codons are statistically more similar to natural
    high-expression chloroplast gene starts.
    """
    if not bias_table:
        return 0.0
    cds = _codons(dna)
    if len(cds) < 2:
        return 0.0
    body = cds[1 : 1 + n_codons]
    if not body:
        return 0.0
    total = 0.0
    for c in body:
        v = bias_table.get(c)
        if v is not None:
            total += v
    return total
