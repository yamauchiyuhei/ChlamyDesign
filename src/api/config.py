"""API configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
DATA_CSV = Path(os.getenv(
    "DATA_CSV",
    str(PROJECT_ROOT / "data" / "processed" / "green_algae_chloroplast_cds.csv"),
))

# Model
BASE_MODEL_ID = "adibvafa/CodonTransformer"
CHLAMYCT_MODEL_PATH = os.getenv("CHLAMYCT_MODEL_PATH", "")
ORGANISM = "Chlamydomonas reinhardtii chloroplast"

# Limits
CT_MAX_AA = 2046
FREQ_MAX_AA = 3000
MFE_MAX_NT = 3000

# Server
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000,https://chlamydesign.web.app").split(",")
PORT = int(os.getenv("PORT", "8080"))
