"""Collect chloroplast CDS from green algae species via NCBI.

Downloads chloroplast genome GenBank records for green algae species,
extracts protein-coding CDS, validates sequences, and outputs a training
CSV compatible with CodonTransformer.

Usage:
    python src/data/collect_chloroplast_cds.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

# --- Constants ---

ORGANISM_LABEL = "Chlamydomonas reinhardtii chloroplast"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "green_algae_chloroplast_cds.csv"

REQUEST_DELAY = 0.4  # seconds between NCBI requests
MAX_RETRIES = 3
MIN_DNA_LEN = 30
STOP_CODONS = {"TAA", "TAG", "TGA"}

SPECIES_ACCESSIONS: dict[str, str] = {
    "Chlamydomonas reinhardtii": "BK000554",
    "Volvox carteri": "GU084820",
    "Gonium pectorale": "AP012494",
    "Dunaliella salina": "NC_016732",
    "Chlorella vulgaris": "NC_001865",
    "Scenedesmus obliquus": "NC_008101",
    "Coccomyxa subellipsoidea": "NC_015084",
    "Ostreococcus tauri": "NC_008289",
    "Micromonas pusilla": "NC_012568",
    "Bathycoccus prasinos": "NC_024832",
    "Parachlorella kessleri": "NC_012978",
    "Nephroselmis olivacea": "NC_000927",
}


def fetch_genbank_record(accession: str) -> SeqRecord | None:
    """Download a GenBank record from NCBI with retry logic.

    Args:
        accession: NCBI accession number.

    Returns:
        SeqRecord or None if all retries fail.
    """
    for attempt in range(MAX_RETRIES):
        try:
            handle = Entrez.efetch(
                db="nucleotide", id=accession, rettype="gb", retmode="text"
            )
            record = SeqIO.read(handle, "genbank")
            handle.close()
            time.sleep(REQUEST_DELAY)
            return record
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [警告] {accession} 取得失敗 (試行 {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    print(f"  [エラー] {accession} をスキップします")
    return None


def calc_gc(dna: str) -> float:
    """Calculate GC content percentage.

    Args:
        dna: DNA sequence string.

    Returns:
        GC percentage (0-100).
    """
    if not dna:
        return 0.0
    gc = sum(1 for b in dna if b in "GC")
    return round(gc / len(dna) * 100, 2)


def extract_cds_from_record(
    record: SeqRecord, species_name: str
) -> list[dict]:
    """Extract validated CDS from a GenBank record.

    Args:
        record: BioPython SeqRecord from GenBank.
        species_name: Source species name for metadata.

    Returns:
        List of dicts with CDS data.
    """
    results: list[dict] = []

    for feature in record.features:
        if feature.type != "CDS":
            continue

        # Skip pseudogenes
        if "pseudo" in feature.qualifiers or "pseudogene" in feature.qualifiers:
            continue

        # Require translation qualifier
        translation = feature.qualifiers.get("translation", [None])[0]
        if not translation:
            continue

        # Extract DNA
        try:
            dna = str(feature.location.extract(record.seq)).upper()
        except Exception:
            continue

        # Filter: length divisible by 3 and >= MIN_DNA_LEN
        if len(dna) < MIN_DNA_LEN or len(dna) % 3 != 0:
            continue

        # Append stop codon if missing
        if dna[-3:] not in STOP_CODONS:
            dna = dna + "TAA"

        # Format protein: strip trailing *, append _
        protein = translation.rstrip("*")
        if not protein.endswith("_"):
            protein += "_"

        gene = feature.qualifiers.get(
            "gene", feature.qualifiers.get("locus_tag", ["unknown"])
        )[0]

        results.append({
            "dna": dna,
            "protein": protein,
            "organism": ORGANISM_LABEL,
            "source_species": species_name,
            "gene": gene,
            "gc_percent": calc_gc(dna),
            "length_aa": len(protein) - 1,  # exclude trailing _
        })

    return results


def collect_all_cds() -> pd.DataFrame:
    """Collect CDS from all species.

    Returns:
        DataFrame with all collected CDS.
    """
    Entrez.email = "yuheiyamauchi@example.com"

    all_cds: list[dict] = []

    print(f"対象種: {len(SPECIES_ACCESSIONS)} 種")
    print("=" * 50)

    for species, accession in SPECIES_ACCESSIONS.items():
        print(f"\n[{species}] ({accession}) を処理中...")
        record = fetch_genbank_record(accession)
        if record is None:
            continue

        cds_list = extract_cds_from_record(record, species)
        print(f"  -> {len(cds_list)} CDS 取得")
        all_cds.extend(cds_list)

    df = pd.DataFrame(all_cds)
    if df.empty:
        return df

    # Deduplicate identical DNA within same species
    before = len(df)
    df = df.drop_duplicates(subset=["dna", "source_species"])
    removed = before - len(df)
    if removed:
        print(f"\n重複除去: {removed} 件")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print collection summary statistics.

    Args:
        df: Collected CDS DataFrame.
    """
    print("\n" + "=" * 50)
    print("収集サマリー")
    print("=" * 50)
    print(f"総配列数: {len(df)}")
    print()
    print("種ごとの配列数:")
    for species, count in df["source_species"].value_counts().items():
        print(f"  {species}: {count}")
    print()
    print(f"全体の平均GC%: {df['gc_percent'].mean():.1f}%")
    print(f"平均アミノ酸長: {df['length_aa'].mean():.1f} aa")


def main() -> None:
    """Main entry point."""
    df = collect_all_cds()

    if df.empty:
        print("エラー: CDS が収集できませんでした")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n保存完了: {OUTPUT_PATH}")

    print_summary(df)


if __name__ == "__main__":
    main()
