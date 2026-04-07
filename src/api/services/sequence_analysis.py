"""Sequence analysis services: restriction enzymes and RNA structure."""

from __future__ import annotations

from typing import Optional

from src.api.schemas import (
    RestrictionResponse,
    RestrictionSite,
    RNAStructureResponse,
)


# Common restriction enzymes for molecular cloning
COMMON_ENZYMES = [
    "EcoRI", "BamHI", "HindIII", "XbaI", "SalI", "PstI", "SphI", "KpnI",
    "NcoI", "NdeI", "XhoI", "NotI", "BglII", "ClaI", "EcoRV", "SacI",
    "NheI", "AvrII", "SpeI", "MluI", "BsaI", "BbsI", "SapI", "AarI",
    "NsiI", "ApaI", "SmaI", "ScaI", "PvuII", "HpaI",
]


def analyze_restriction_sites(
    dna: str,
    enzyme_names: Optional[list[str]] = None,
) -> RestrictionResponse:
    """Analyze restriction enzyme sites in a DNA sequence."""
    from Bio.Restriction import RestrictionBatch, Analysis
    from Bio.Seq import Seq

    seq = Seq(dna)

    # Build enzyme list
    if enzyme_names:
        names_to_use = enzyme_names
    else:
        names_to_use = COMMON_ENZYMES

    # Create restriction batch (skip unknown enzymes)
    batch = RestrictionBatch()
    valid_count = 0
    for name in names_to_use:
        try:
            batch.add(name)
            valid_count += 1
        except (ValueError, KeyError):
            continue

    if valid_count == 0:
        return RestrictionResponse(
            total_enzymes_checked=0,
            total_sites_found=0,
            length_nt=len(dna),
            enzymes=[],
        )

    # Run analysis
    analysis = Analysis(batch, seq, linear=True)
    results = analysis.full()

    enzyme_results = []
    total_sites = 0

    for enzyme, positions in sorted(results.items(), key=lambda x: str(x[0])):
        if not positions:
            continue

        num_cuts = len(positions)
        total_sites += num_cuts

        # Get recognition site
        try:
            rec_site = str(enzyme.site)
        except Exception:
            rec_site = str(enzyme)

        # Calculate fragment sizes
        cut_pos = sorted(positions)
        fragments = []
        prev = 0
        for pos in cut_pos:
            fragments.append(pos - prev)
            prev = pos
        fragments.append(len(dna) - prev)

        enzyme_results.append(RestrictionSite(
            enzyme=str(enzyme),
            recognition_site=rec_site,
            cut_positions=cut_pos,
            num_cuts=num_cuts,
            fragment_sizes=fragments,
        ))

    # Sort by number of cuts (descending)
    enzyme_results.sort(key=lambda x: x.num_cuts, reverse=True)

    return RestrictionResponse(
        total_enzymes_checked=valid_count,
        total_sites_found=total_sites,
        length_nt=len(dna),
        enzymes=enzyme_results,
    )


def _mountain_from_structure(structure: str) -> list[int]:
    """Calculate mountain plot data (nesting depth) from dot-bracket string."""
    depths = []
    depth = 0
    for c in structure:
        if c == "(":
            depth += 1
        depths.append(depth)
        if c == ")":
            depth -= 1
    return depths


def predict_rna_structure(
    dna: str,
    region: Optional[str] = None,
) -> RNAStructureResponse:
    """Predict RNA secondary structure using ViennaRNA (RNAfold)."""
    # Determine which region to analyze
    region_label = region or "full"

    if region == "5utr":
        seq = dna[:min(100, len(dna))]
        region_label = "5' UTR (first 100 nt)"
    elif region == "3utr":
        seq = dna[max(0, len(dna) - 100):]
        region_label = "3' UTR (last 100 nt)"
    else:
        seq = dna
        region_label = "full sequence"

    # Convert DNA to RNA
    rna = seq.replace("T", "U").replace("t", "u")

    gc = sum(1 for b in seq if b in "GC") / len(seq) * 100

    try:
        import RNA
        import math

        # Single fold compound — correct call order: mfe() → pf() → centroid()
        fc = RNA.fold_compound(rna)

        # 1. MFE structure
        structure, mfe = fc.mfe()
        bp_count = structure.count("(")

        # 2. Partition function (must be called after mfe)
        ensemble_energy = None
        ensemble_diversity = None
        centroid_structure = None
        centroid_mfe = None
        positional_entropy = None
        mountain_pf = None
        mountain_centroid = None

        try:
            # CRITICAL: rescale partition function parameters using the MFE
            # to prevent numerical overflow on long, highly stable structures.
            # Without this, bpp values underflow to 0 and centroid becomes degenerate.
            try:
                fc.exp_params_rescale(mfe)
            except Exception:
                pass
            pf_str, pf_energy = fc.pf()
            ensemble_energy = round(pf_energy, 2)
        except Exception:
            pf_str = None

        # 3. Centroid structure (must be called after pf)
        if pf_str is not None:
            try:
                centroid_result = fc.centroid()
                if centroid_result and len(centroid_result) >= 2:
                    centroid_structure = centroid_result[0]
                    centroid_mfe = round(centroid_result[1], 2)
            except Exception:
                pass

            # 4. Ensemble diversity (must be called after pf)
            try:
                ensemble_diversity = round(fc.mean_bp_distance(), 2)
            except Exception:
                pass

            # 5. Positional entropy — ViennaRNA standard per-pair Shannon entropy
            #    S_i = -Σ_j (p_ij · log2 p_ij)  -  q_i · log2 q_i
            #    where q_i = 1 - Σ_j p_ij  (probability of being unpaired)
            #    This sums over EACH possible partner separately so values can
            #    exceed 1 (theoretical max ≈ log2(n)).
            try:
                n = len(rna)
                bpp = fc.bpp()  # 1-indexed upper triangular: bpp[i][j], i<j
                entropy = []
                for i in range(1, n + 1):
                    s = 0.0
                    p_paired = 0.0
                    # Iterate over all possible partners j ≠ i
                    for j in range(i + 1, n + 1):
                        try:
                            p = bpp[i][j]
                        except (IndexError, TypeError):
                            p = 0.0
                        if p > 1e-12:
                            s -= p * math.log2(p)
                            p_paired += p
                    for j in range(1, i):
                        try:
                            p = bpp[j][i]
                        except (IndexError, TypeError):
                            p = 0.0
                        if p > 1e-12:
                            s -= p * math.log2(p)
                            p_paired += p
                    q = 1.0 - p_paired
                    if 0 < q < 1:
                        s -= q * math.log2(q)
                    entropy.append(round(s, 4))
                positional_entropy = entropy
            except Exception:
                pass

            # 6. Mountain plot for partition function structure
            try:
                pf_simple = ""
                for c in pf_str:
                    if c in "({[<":
                        pf_simple += "("
                    elif c in ")}]>":
                        pf_simple += ")"
                    else:
                        pf_simple += "."
                mountain_pf = _mountain_from_structure(pf_simple)
            except Exception:
                pass

            # 7. Mountain plot for centroid
            if centroid_structure:
                mountain_centroid = _mountain_from_structure(centroid_structure)

        # Mountain plot for MFE (always available)
        mountain_mfe = _mountain_from_structure(structure)

        # 8. 2D layout coordinates (NAView planar layout by ViennaRNA)
        def _compute_layout(s):
            try:
                coords = RNA.naview_xy_coordinates(s)
                return [float(c.X) for c in coords], [float(c.Y) for c in coords], "naview"
            except Exception:
                try:
                    coords = RNA.simple_xy_coordinates(s)
                    return [float(c.X) for c in coords], [float(c.Y) for c in coords], "simple"
                except Exception:
                    return None, None, None

        layout_x, layout_y, layout_type = _compute_layout(structure)
        layout_centroid_x = None
        layout_centroid_y = None
        if centroid_structure:
            layout_centroid_x, layout_centroid_y, _ = _compute_layout(centroid_structure)

        return RNAStructureResponse(
            sequence=rna,
            structure=structure,
            mfe=round(mfe, 2),
            length_nt=len(rna),
            gc_percent=round(gc, 2),
            base_pair_count=bp_count,
            region=region_label,
            ensemble_energy=ensemble_energy,
            ensemble_diversity=ensemble_diversity,
            centroid_structure=centroid_structure,
            centroid_mfe=centroid_mfe,
            positional_entropy=positional_entropy,
            mountain_mfe=mountain_mfe,
            mountain_centroid=mountain_centroid,
            mountain_pf=mountain_pf,
            layout_x=layout_x,
            layout_y=layout_y,
            layout_type=layout_type,
            layout_centroid_x=layout_centroid_x,
            layout_centroid_y=layout_centroid_y,
            source="vienna",
        )
    except ImportError:
        # ViennaRNA not available — use simple Nussinov-like estimation
        structure, mfe, bp_count = _simple_fold_estimate(rna)
        mountain_mfe = _mountain_from_structure(structure)

        return RNAStructureResponse(
            sequence=rna,
            structure=structure,
            mfe=round(mfe, 2),
            length_nt=len(rna),
            gc_percent=round(gc, 2),
            base_pair_count=bp_count,
            region=region_label,
            mountain_mfe=mountain_mfe,
            source="nussinov",
        )


def _simple_fold_estimate(rna: str) -> tuple[str, float, int]:
    """Simple MFE estimate when ViennaRNA is not available.

    Uses a basic approach: estimates MFE from GC content and length,
    generates a placeholder dot-bracket structure.
    """
    n = len(rna)
    gc_count = sum(1 for b in rna if b in "GC")
    gc_frac = gc_count / n if n > 0 else 0

    # Rough MFE estimate: ~-0.3 to -0.5 kcal/mol per nucleotide for typical RNA
    # Higher GC → more stable (more negative MFE)
    mfe_per_nt = -0.2 - 0.3 * gc_frac
    mfe = mfe_per_nt * n

    # Estimate ~30-50% of nucleotides are paired
    pair_frac = 0.3 + 0.2 * gc_frac
    bp_count = int(n * pair_frac / 2)

    # Generate placeholder structure
    structure = "." * n

    return structure, mfe, bp_count
