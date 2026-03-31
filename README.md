# ChlamyDesign

**AI-powered gene design platform for *Chlamydomonas reinhardtii* chloroplast expression**

> Waseda University — Nishimura Laboratory

## Overview

ChlamyDesign is a web-based platform that provides AI-assisted codon optimization specifically tailored for the *Chlamydomonas reinhardtii* chloroplast genome.

## Core Components

| Module | Role | License |
|--------|------|---------|
| **ChlamyCodonTransformer** | DL codon optimization fine-tuned on Chlamy cpDNA | Apache 2.0 |
| **LinearDesign** | mRNA stability + CAI joint optimization | MIT |
| **DNAchisel** | Constraint-based sequence refinement | MIT |
| **ViennaRNA** | RNA secondary structure prediction | Academic free |

## Project Structure
```
ChlamyDesign/
├── src/
│   ├── model/          # ChlamyCodonTransformer training & inference
│   ├── data/           # Data collection & preprocessing
│   ├── evaluation/     # Model evaluation scripts
│   └── web/            # Web frontend (Firebase)
├── notebooks/          # Colab notebooks for training & analysis
├── data/               # Training datasets
├── models/             # Saved model weights
├── results/            # Evaluation outputs
└── docs/               # Documentation
```

## References

- Fallahpour et al. (2025) *Nature Communications* 16, 3205 — CodonTransformer
- Zhang et al. (2023) *Nature* 621, 396–403 — LinearDesign
- Zulkower & Rosser (2020) *Bioinformatics* 36(16), 4508–4509 — DNAchisel
- Lorenz et al. (2011) *Algorithms for Molecular Biology* 6, 26 — ViennaRNA

## License

Apache License 2.0
