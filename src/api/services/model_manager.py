"""Singleton for model loading and precomputed reference data."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from CodonTransformer.CodonData import get_codon_frequencies
from CodonTransformer.CodonEvaluation import get_CSI_weights
from CodonTransformer.CodonPrediction import (
    load_model as ct_load_model,
    load_tokenizer,
)

from src.api.config import BASE_MODEL_ID, CHLAMYCT_MODEL_PATH, DATA_CSV


class ModelManager:
    """Manages model loading and precomputed reference data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer = None
        self._base_model = None
        self._chlamyct_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precomputed at startup
        self.csi_weights: dict[str, float] = {}
        self.codon_freqs: dict = {}
        self.reference_gene_count: int = 0

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
        self.codon_freqs = get_codon_frequencies(dna_list, protein_list)
        print(f"[ModelManager] Precomputed: {len(self.codon_freqs)} amino acids, "
              f"{self.reference_gene_count} reference genes")

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
                    self._chlamyct_model = ct_load_model(
                        model_path=CHLAMYCT_MODEL_PATH, device=self._device
                    )
        return self._chlamyct_model

    @property
    def chlamyct_available(self) -> bool:
        """Check if ChlamyCT model path is configured."""
        if not CHLAMYCT_MODEL_PATH:
            return False
        p = Path(CHLAMYCT_MODEL_PATH)
        return p.exists() or not p.suffix  # local path exists or HF repo ID

    @property
    def base_model_loaded(self) -> bool:
        return self._base_model is not None

    @property
    def chlamyct_model_loaded(self) -> bool:
        return self._chlamyct_model is not None


# Global singleton
manager = ModelManager()
