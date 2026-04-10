"""Fine-tune CodonTransformer on Chlamydomonas reinhardtii chloroplast CDS.

Standalone local training script (no Google Colab dependency).
Supports MPS (Apple Silicon), CUDA, and CPU.

Usage:
    python src/model/finetune.py                   # defaults
    python src/model/finetune.py --epochs 5        # fewer epochs
    python src/model/finetune.py --device cpu       # force CPU
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    BigBirdForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from CodonTransformer.CodonPrediction import load_tokenizer, predict_dna_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CodonDataset(Dataset):
    """Map-style dataset that reads CodonTransformer training JSON."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[str] = []
        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                # Format: organism_id + space + codon sequence
                self.samples.append(f"{rec['organism']} {rec['codons']}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

# Defaults matching the original Colab notebook design.
DEFAULTS = {
    "training_data": str(PROJECT_ROOT / "data" / "processed" / "training_data.json"),
    "output_dir": str(PROJECT_ROOT / "models" / "ChlamyCodonTransformer"),
    "base_model": "adibvafa/CodonTransformer",
    "batch_size": 6,
    "epochs": 10,
    "lr": 5e-5,
    "warmup_ratio": 0.1,
    "mlm_probability": 0.15,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "save_total_limit": 3,
}


def _detect_device(preferred: str | None = None) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if preferred:
        return torch.device(preferred)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ChlamyCodonTransformer")
    parser.add_argument("--training-data", default=DEFAULTS["training_data"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--base-model", default=DEFAULTS["base_model"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    args = parser.parse_args()

    device = _detect_device(args.device)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # --- Data ---
    data_path = args.training_data
    if not Path(data_path).exists():
        print(f"ERROR: Training data not found: {data_path}")
        sys.exit(1)

    n_samples = _count_lines(data_path)
    steps_per_epoch = math.ceil(n_samples / args.batch_size)
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = int(max_steps * args.warmup_ratio)

    print(f"Training data: {data_path} ({n_samples} samples)")
    print(f"Steps: {steps_per_epoch}/epoch × {args.epochs} epochs = {max_steps} total")
    print(f"Warmup: {warmup_steps} steps")

    # --- Model & Tokenizer ---
    print(f"Loading base model: {args.base_model}")
    model = BigBirdForMaskedLM.from_pretrained(args.base_model)
    tokenizer = load_tokenizer()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # --- Dataset ---
    train_dataset = CodonDataset(data_path=data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=DEFAULTS["mlm_probability"],
    )

    # --- Training ---
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine fp16/bf16 support.
    use_fp16 = device.type == "cuda"
    # MPS supports float16 but Trainer fp16 flag is CUDA-specific;
    # use bf16 on MPS if supported (PyTorch >= 2.1).
    use_bf16 = device.type == "mps"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=DEFAULTS["weight_decay"],
        logging_steps=DEFAULTS["logging_steps"],
        save_strategy="epoch",
        save_total_limit=DEFAULTS["save_total_limit"],
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,  # MPS-safe
        dataloader_pin_memory=False,  # MPS doesn't support pin_memory
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print(f"\n{'='*60}")
    print("Starting fine-tuning...")
    print(f"{'='*60}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # --- Save ---
    final_dir = str(Path(output_dir) / "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved: {final_dir}")

    # Also save as top-level (for direct from_pretrained loading).
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved: {output_dir}")

    # --- Quick inference test ---
    print(f"\n{'='*60}")
    print("Inference test (rbcL protein)...")
    print(f"{'='*60}\n")

    model.eval()
    model.to(device)

    test_protein = (
        "MSPQTETKASVGFKAGVKEYKLTYYTPEYETKDTDILAAFRVTPQPG"
        "VPPEEAGAAVAAESSTGTWTTVWTDGLTSLDRYKGRCYRIEEGQTINE"
        "DTAYFFDQFKQGCDINIFQYDSAGLLNKFEEMDHYREPEHSPESAGGI"
        "HVWHMPALTEIFGDDSVLQFGGGTLGHPWGNAPGATN_"
    )

    result = predict_dna_sequence(
        protein=test_protein,
        organism="Chlamydomonas reinhardtii chloroplast",
        device=device,
        model=model,
        tokenizer=tokenizer,
        deterministic=True,
    )

    dna = result.predicted_dna
    gc = sum(1 for b in dna if b in "GC") / len(dna) * 100
    print(f"Protein: rbcL ({len(test_protein) - 1} aa)")
    print(f"DNA length: {len(dna)} nt")
    print(f"GC%: {gc:.1f}% (target: ~34%, base CT: ~61%)")
    print(f"Start codon: {dna[:3]}")
    print(f"Stop codon: {dna[-3:]}")

    if gc < 45:
        print("\n✓ GC% significantly improved from base CT (~61%)")
    else:
        print("\n⚠ GC% still elevated — may need more epochs or data")

    print(f"\nDone. Model ready at: {output_dir}")


if __name__ == "__main__":
    main()
