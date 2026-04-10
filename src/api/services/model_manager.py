"""Singleton for model loading and precomputed reference data."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from CodonTransformer.CodonData import get_codon_frequencies
from CodonTransformer.CodonEvaluation import get_CSI_weights
from transformers import BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import (
    load_model as ct_load_model,
    load_tokenizer,
)
from CodonTransformer.CodonUtils import ORGANISM2ID

from src.api.config import CHLAMYCT_MODEL_PATH, DATA_CSV, DEFAULT_ORGANISM
from src.api.services.codon_pair import compute_cps_table
from src.api.services.five_prime_context import (
    HIGH_EXPRESSION_GENES,
    compute_five_prime_bias,
)


class ModelManager:
    """Manages model loading and precomputed reference data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer = None
        self._base_model = None
        self._chlamyct_model = None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Precomputed at startup (for Chlamydomonas chloroplast)
        self.csi_weights: dict[str, float] = {}
        self.codon_freqs: dict = {}
        self.cps_table: dict[str, float] = {}
        self.five_prime_bias: dict[str, float] = {}
        self.stop_codon_freqs: dict[str, float] = {}
        self.high_expression_count: int = 0
        self.reference_gene_count: int = 0

        # Organism list from CodonTransformer
        self.organism_list: list[dict] = []
        self._init_organism_list()

        # Cache for organism-specific codon frequencies
        self._organism_freq_cache: dict[str, dict] = {}

    def _init_organism_list(self) -> None:
        """Initialize the organism list from CodonTransformer's ORGANISM2ID."""
        self.organism_list = [
            {"id": org_id, "name": org_name}
            for org_name, org_id in sorted(ORGANISM2ID.items(), key=lambda x: x[1])
        ]

    def precompute(self) -> None:
        """Precompute CSI weights and codon frequencies from reference data."""
        if not DATA_CSV.exists():
            print(f"[ModelManager] WARNING: {DATA_CSV} not found, skipping precompute")
            return

        df = pd.read_csv(DATA_CSV)
        cr_df = df[df["source_species"] == "Chlamydomonas reinhardtii"].copy()
        self.reference_gene_count = len(cr_df)

        # Filter to ATG-starting sequences (GTG incompatible with get_codon_frequencies)
        atg_df = cr_df[cr_df["dna"].str.startswith("ATG")]
        if atg_df.empty:
            print("[ModelManager] WARNING: No ATG-starting C. reinhardtii sequences")
            return

        dna_list = atg_df["dna"].tolist()
        protein_list = atg_df["protein"].tolist()

        self.csi_weights = get_CSI_weights(dna_list)
        raw_freqs = get_codon_frequencies(dna_list, protein_list)
        self.cps_table = compute_cps_table(dna_list)

        # 5' context bias from high-expression chloroplast genes only.
        if "gene" in atg_df.columns:
            high_expr_df = atg_df[atg_df["gene"].isin(HIGH_EXPRESSION_GENES)]
            self.high_expression_count = len(high_expr_df)
            if not high_expr_df.empty:
                self.five_prime_bias = compute_five_prime_bias(
                    high_expr_df["dna"].tolist(),
                    dna_list,
                )

        # Compute stop codon frequencies from reference CDS.
        stop_counts: dict[str, int] = {"TAA": 0, "TAG": 0, "TGA": 0}
        for dna in cr_df["dna"].tolist():
            last3 = dna[-3:].upper()
            if last3 in stop_counts:
                stop_counts[last3] += 1
        stop_total = sum(stop_counts.values())
        if stop_total > 0:
            self.stop_codon_freqs = {k: v / stop_total for k, v in stop_counts.items()}
        else:
            self.stop_codon_freqs = {"TAA": 0.94, "TAG": 0.06, "TGA": 0.00}

        # Convert AMINO2CODON_TYPE (tuple format) to dict-of-dict format
        # AMINO2CODON_TYPE: Dict[str, Tuple[List[str], List[float]]]
        # Our format:       Dict[str, Dict[str, float]]
        self.codon_freqs = {}
        for aa, (codon_list, freq_list) in raw_freqs.items():
            self.codon_freqs[aa] = dict(zip(codon_list, freq_list))

        # Cache Chlamydomonas chloroplast frequencies
        self._organism_freq_cache[DEFAULT_ORGANISM] = self.codon_freqs

        print(f"[ModelManager] Precomputed: {len(self.codon_freqs)} amino acids, "
              f"{self.reference_gene_count} reference genes, "
              f"{len(self.cps_table)} codon pairs (CPS table), "
              f"{len(self.five_prime_bias)} codons (5' bias from "
              f"{self.high_expression_count} high-expression genes)")

    def get_organism_frequencies(self, organism: str) -> dict:
        """Get codon frequencies for a given organism.

        For Chlamydomonas reinhardtii chloroplast, returns precomputed frequencies
        from our reference dataset. For other organisms, uses CodonTransformer's
        built-in frequency tables.
        """
        if organism in self._organism_freq_cache:
            return self._organism_freq_cache[organism]

        # Validate organism
        if organism not in ORGANISM2ID:
            raise ValueError(f"Unknown organism: {organism}")

        # Compute frequencies from CodonTransformer's dataset
        try:
            freq_data = _compute_organism_frequencies_from_ct(organism)
            self._organism_freq_cache[organism] = freq_data
            return freq_data
        except Exception as e:
            print(f"[ModelManager] Error computing frequencies for {organism}: {e}")
            raise ValueError(f"Could not compute frequencies for {organism}: {e}")

    @property
    def device(self) -> torch.device:
        return self._device

    def get_tokenizer(self):
        """Get tokenizer (lazy-loaded, thread-safe)."""
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    print("[ModelManager] Loading tokenizer...")
                    self._tokenizer = load_tokenizer()
        return self._tokenizer

    def get_base_model(self):
        """Get base CodonTransformer model (lazy-loaded, thread-safe)."""
        if self._base_model is None:
            with self._lock:
                if self._base_model is None:
                    print(f"[ModelManager] Loading base model from HuggingFace...")
                    self._base_model = ct_load_model(
                        model_path=None, device=self._device
                    )
        return self._base_model

    def get_chlamyct_model(self) -> Optional[Any]:
        """Get ChlamyCT fine-tuned model (lazy-loaded, thread-safe)."""
        if not self.chlamyct_available:
            return None
        if self._chlamyct_model is None:
            with self._lock:
                if self._chlamyct_model is None:
                    print(f"[ModelManager] Loading ChlamyCT: {CHLAMYCT_MODEL_PATH}")
                    p = Path(CHLAMYCT_MODEL_PATH)
                    if p.is_dir():
                        # Local HuggingFace-format directory (from finetune.py).
                        model = BigBirdForMaskedLM.from_pretrained(
                            str(p), local_files_only=True
                        )
                        model.eval()
                        model.to(self._device)
                        self._chlamyct_model = model
                    else:
                        self._chlamyct_model = ct_load_model(
                            model_path=CHLAMYCT_MODEL_PATH, device=self._device
                        )
        return self._chlamyct_model

    @property
    def chlamyct_available(self) -> bool:
        """Check if ChlamyCT model weights exist."""
        if not CHLAMYCT_MODEL_PATH:
            return False
        p = Path(CHLAMYCT_MODEL_PATH)
        if p.is_dir():
            # HuggingFace format: needs config.json at minimum.
            return (p / "config.json").exists()
        if p.suffix in (".pt", ".ckpt"):
            return p.exists()
        # Assume HuggingFace repo ID (e.g. "user/ChlamyCodonTransformer").
        return bool(CHLAMYCT_MODEL_PATH)

    @property
    def base_model_loaded(self) -> bool:
        return self._base_model is not None

    @property
    def chlamyct_model_loaded(self) -> bool:
        return self._chlamyct_model is not None


def _compute_organism_frequencies_from_ct(organism: str) -> dict:
    """Compute codon frequencies for an organism using CT's built-in tables."""
    from Bio.Seq import Seq

    # Build standard genetic code: codon -> amino acid mapping
    codons_for_aa: dict[str, list[str]] = {}
    for i in "TCAG":
        for j in "TCAG":
            for k in "TCAG":
                codon = i + j + k
                aa = str(Seq(codon).translate())
                if aa == "*":
                    continue
                codons_for_aa.setdefault(aa, []).append(codon)

    # Try HuggingFace dataset with codon usage frequencies
    try:
        from datasets import load_dataset
        ds = load_dataset("adibvafa/codon_usage_frequencies", split="train")
        df = ds.to_pandas()
        org_df = df[df["organism"] == organism]
        if not org_df.empty:
            freqs = {}
            for aa, codon_list in codons_for_aa.items():
                aa_freqs = {}
                total = 0
                for codon in codon_list:
                    col = codon.lower()
                    if col in org_df.columns:
                        count = float(org_df[col].iloc[0])
                        aa_freqs[codon] = count
                        total += count
                if total > 0:
                    normalized = {c: v / total for c, v in aa_freqs.items()}
                    # Ensure exact sum to 1.0 by adjusting the last entry
                    keys = list(normalized.keys())
                    s = sum(normalized[k] for k in keys[:-1])
                    normalized[keys[-1]] = 1.0 - s
                    freqs[aa] = normalized
                else:
                    n = len(codon_list)
                    freqs[aa] = {c: (1.0 / n if i < n - 1 else 1.0 - (n - 1) / n) for i, c in enumerate(codon_list)}
            return freqs
    except Exception as e:
        print(f"[ModelManager] HF dataset fallback: {e}")

    # Fallback: uniform distribution
    freqs = {}
    for aa, codon_list in codons_for_aa.items():
        freqs[aa] = {c: round(1.0 / len(codon_list), 6) for c in codon_list}
    return freqs


# Global singleton
manager = ModelManager()
