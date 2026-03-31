# CLAUDE.md — ChlamyDesign Project Instructions

## Project overview
ChlamyDesign is an AI-powered codon optimization platform for Chlamydomonas reinhardtii chloroplast. Core deliverable: ChlamyCodonTransformer, a fine-tuned CodonTransformer.

## Key facts
- Chloroplast genome: ~34% GC (vs nuclear ~64% GC)
- Base CodonTransformer overshoots chloroplast GC% by +23.8pp
- Chloroplast genome GenBank: BK000554, 69 protein-coding genes
- High-expression genes: rbcL, psbA, psaA, atpA, psbD, psbC

## Repository structure
src/model/ — fine-tuning and inference
src/data/ — data collection and preprocessing
src/evaluation/ — model evaluation
src/web/public/ — Firebase frontend
notebooks/ — Colab notebooks
data/raw/ — downloaded genomes (gitignored)
data/processed/ — preprocessed data
models/ — saved weights (gitignored)
results/ — evaluation outputs

## Conventions
- Python 3.10+, type hints, Google-style docstrings
- Commit prefixes: feat: fix: data: eval: web: docs:
- DNA must: start ATG/GTG, end TAA/TAG/TGA, len % 3 == 0
- Protein sequences end with _ for CodonTransformer format
- Training CSV columns: dna, protein, organism

## Phase 1: Data expansion (NEXT)
Create src/data/collect_chloroplast_cds.py:
- Download chloroplast genomes from NCBI for 15+ green algae species
- Species: Volvox carteri, Gonium pectorale, Dunaliella salina, Chlorella vulgaris, Scenedesmus obliquus, Parachlorella kessleri, Oedogonium cardiacum, Coccomyxa subellipsoidea, Haematococcus lacustris, Monoraphidium neglectum, Micromonas pusilla, Ostreococcus tauri, Botryococcus braunii, Stigeoclonium helveticum, Bathycoccus prasinos
- Extract CDS using BioPython, validate sequences
- Set organism="Chlamydomonas reinhardtii chloroplast" for all
- Combine with existing 69 CDS from BK000554
- Save to data/processed/green_algae_chloroplast_cds.csv
- Target: 1500+ sequences

## Phase 2: Preprocessing
Create src/data/prepare_training_data.py:
- Load combined CDS, validate, calculate CSI
- Add expression weights (Tier1=5x, Tier2=2x, Tier3=1x)
- Run CodonTransformer.CodonData.prepare_training_data()
- Output: data/processed/training_data.json

## Phase 3: Fine-tuning (GPU required)
- Use finetune.py from CodonTransformer repo
- Args: batch_size=6, max_epochs=10, lr=5e-5, warmup=0.1
- Save model to Google Drive + HuggingFace

## Phase 4: Evaluation
Create src/evaluation/compare_models.py:
- Generate DNA for all 69 cp genes using: ChlamyCT, base CT, CAI-max, BFC, natural
- Metrics: CSI, GC%, CFD, %MinMax/DTW, cis-elements, MFE
- Save results/model_comparison.csv and figures

## Phase 6: Web UI
- Full dashboard in src/web/public/index.html
- Deploy: firebase deploy --only hosting

## Git
git remote uses token auth. If 403 error:
git remote set-url origin https://yamauchiyuhei:TOKEN@github.com/yamauchiyuhei/ChlamyDesign.git
