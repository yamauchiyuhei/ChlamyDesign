"""ChlamyCodonTransformer 推論スクリプト。

ファインチューニング済みモデルを使ってタンパク質配列からDNAを生成する。

Usage:
    python src/model/predict.py --protein "MSPQTET...K_" --model models/ChlamyCodonTransformer.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from CodonTransformer.CodonPrediction import (
    predict_dna_sequence,
    load_model as ct_load_model,
    load_tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORGANISM = "Chlamydomonas reinhardtii chloroplast"


def load_model(model_path: str | None, device: torch.device):
    """モデルを読み込む（CodonTransformer API使用）。

    Args:
        model_path: .pt/.ckpt ファイルパス、HuggingFace リポジトリID、
                    またはディレクトリ。Noneでベースモデル。
        device: 使用デバイス。

    Returns:
        読み込み済みモデル。
    """
    if model_path:
        print(f"モデル読み込み: {model_path}")
    else:
        print("モデル読み込み: HuggingFace (adibvafa/CodonTransformer)")
    model = ct_load_model(model_path=model_path, device=device)
    return model


def predict(
    protein: str,
    model,
    tokenizer,
    device: torch.device,
    num_sequences: int = 1,
    temperature: float = 0.2,
) -> dict[str, str | float]:
    """タンパク質からDNA配列を予測する。

    Args:
        protein: タンパク質配列（末尾に_）。
        model: 推論用モデル。
        tokenizer: トークナイザー。
        device: 使用デバイス。
        num_sequences: 生成候補数（>1で非決定論的生成、GC%が34%に最も近い配列を選択）。
        temperature: 非決定論的生成時の温度パラメータ (0.2-0.8)。

    Returns:
        predicted_dna, gc_percent を含む辞書。
    """
    if not protein.endswith("_"):
        protein = protein.rstrip("*") + "_"

    deterministic = num_sequences == 1

    result = predict_dna_sequence(
        protein=protein,
        organism=ORGANISM,
        device=device,
        model=model,
        tokenizer=tokenizer,
        deterministic=deterministic,
        temperature=temperature,
        num_sequences=num_sequences,
        match_protein=True,
    )

    if num_sequences > 1:
        # 複数候補からGC%が34%に最も近い配列を選択
        TARGET_GC = 34.0
        best = min(
            result,
            key=lambda r: abs(
                sum(1 for b in r.predicted_dna if b in "GC")
                / len(r.predicted_dna) * 100 - TARGET_GC
            ),
        )
        dna = best.predicted_dna
    else:
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
    parser.add_argument(
        "--num-sequences", type=int, default=1,
        help="生成候補数 (>1で非決定論的生成、GC%%が34%%に最も近い配列を選択)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="非決定論的生成時の温度 (0.2-0.8)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    model = load_model(args.model, device)
    tokenizer = load_tokenizer()

    result = predict(
        args.protein, model, tokenizer, device,
        num_sequences=args.num_sequences,
        temperature=args.temperature,
    )

    print(f"\n予測DNA ({result['length_nt']} nt):")
    print(result["predicted_dna"])
    print(f"\nGC%: {result['gc_percent']:.1f}% (目標: ~34%)")


if __name__ == "__main__":
    main()
