"""ChlamyCodonTransformer 推論スクリプト。

ファインチューニング済みモデルを使ってタンパク質配列からDNAを生成する。

Usage:
    python src/model/predict.py --protein "MSPQTET...K_" --model models/ChlamyCodonTransformer.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from CodonTransformer.CodonPrediction import predict_dna_sequence, load_tokenizer
from transformers import BigBirdForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORGANISM = "Chlamydomonas reinhardtii chloroplast"


def load_model(model_path: str | None, device: torch.device) -> BigBirdForMaskedLM:
    """モデルを読み込む。

    Args:
        model_path: .pt または HuggingFace リポジトリID。Noneでベースモデル。
        device: 使用デバイス。

    Returns:
        読み込み済みモデル。
    """
    if model_path is None:
        print("ベースモデル (adibvafa/CodonTransformer) を使用")
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
    elif Path(model_path).exists():
        print(f"ローカルモデルを読み込み中: {model_path}")
        model = BigBirdForMaskedLM.from_pretrained(model_path)
    else:
        print(f"HuggingFace Hub からモデルを読み込み中: {model_path}")
        model = BigBirdForMaskedLM.from_pretrained(model_path)

    model.to(device)
    model.eval()
    return model


def predict(
    protein: str,
    model: BigBirdForMaskedLM,
    tokenizer,
    device: torch.device,
) -> dict[str, str | float]:
    """タンパク質からDNA配列を予測する。

    Args:
        protein: タンパク質配列（末尾に_）。
        model: 推論用モデル。
        tokenizer: トークナイザー。
        device: 使用デバイス。

    Returns:
        predicted_dna, gc_percent を含む辞書。
    """
    if not protein.endswith("_"):
        protein = protein.rstrip("*") + "_"

    result = predict_dna_sequence(
        protein=protein,
        organism=ORGANISM,
        device=device,
        model=model,
        tokenizer=tokenizer,
    )

    dna = result.predicted_dna
    gc = sum(1 for b in dna if b in "GC") / len(dna) * 100

    return {
        "predicted_dna": dna,
        "gc_percent": round(gc, 2),
        "length_nt": len(dna),
    }


def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="ChlamyCodonTransformer でDNA配列を予測"
    )
    parser.add_argument(
        "--protein", required=True,
        help="タンパク質配列（末尾 _ 付き）"
    )
    parser.add_argument(
        "--model", default=None,
        help="モデルパス (.pt / HuggingFace ID / ディレクトリ)。省略でベースモデル"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    model = load_model(args.model, device)
    tokenizer = load_tokenizer()

    result = predict(args.protein, model, tokenizer, device)

    print(f"\n予測DNA ({result['length_nt']} nt):")
    print(result["predicted_dna"])
    print(f"\nGC%: {result['gc_percent']:.1f}% (目標: ~34%)")


if __name__ == "__main__":
    main()
