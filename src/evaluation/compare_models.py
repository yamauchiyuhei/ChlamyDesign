"""Phase 4: モデル比較評価スクリプト。

C. reinhardtii葉緑体遺伝子68個について5つのコドン最適化戦略を比較する。

戦略:
  natural   : 実際のC. reinhardtii配列
  base_ct   : ベースCodonTransformer (adibvafa/CodonTransformer)
  chlamyct  : ファインチューニング済みChlamyCodonTransformer
  cai_max   : 最頻コドン (CAI最大化)
  bfc       : コドン頻度比例サンプリング (Balanced Frequency Codon)

メトリクス: CSI, GC%, CFD, %MinMax(mean), cis_elements, MFE

Usage:
    python src/evaluation/compare_models.py
    python src/evaluation/compare_models.py --model models/ChlamyCodonTransformer
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import RNA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from CodonTransformer.CodonData import get_codon_frequencies
from CodonTransformer.CodonEvaluation import (
    get_cfd,
    get_CSI_weights,
    get_CSI_value,
    get_GC_content,
    get_min_max_percentage,
    get_sequence_complexity,
)
from CodonTransformer.CodonPrediction import (
    get_high_frequency_choice_sequence,
    get_background_frequency_choice_sequence,
    load_model as ct_load_model,
    load_tokenizer,
    predict_dna_sequence,
)

# --- Constants ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV      = PROJECT_ROOT / "data" / "processed" / "green_algae_chloroplast_cds.csv"
OUTPUT_CSV    = PROJECT_ROOT / "results" / "model_comparison.csv"
FIGURE_DIR    = PROJECT_ROOT / "results" / "figures"

ORGANISM = "Chlamydomonas reinhardtii chloroplast"

# 葉緑体でよく問題になるcis-element（発現を妨げる配列）
CIS_ELEMENT_PATTERNS = {
    "SD_like":   r"AAGG.{5,10}ATG",        # Shine-Dalgarno様配列 + 内部ATG
    "TATA_box":  r"TATAAA",                 # TATA box様配列
    "polyA":     r"AAAAAA",                 # PolyAシグナル
    "palindrome": None,                     # パリンドロームは別途計算
}

random.seed(42)


# ---------------------------------------------------------------------------
# コドン最適化戦略
# ---------------------------------------------------------------------------

def optimize_cai_max(protein: str, codon_freqs: dict) -> str:
    """CAI-max: CT の High Frequency Choice で最頻コドンを割り当て。

    Args:
        protein: タンパク質配列（末尾 _ 付き）。
        codon_freqs: get_codon_frequencies() の出力。

    Returns:
        最適化DNA配列。
    """
    return get_high_frequency_choice_sequence(protein, codon_freqs)


def optimize_bfc(protein: str, codon_freqs: dict) -> str:
    """BFC: CT の Background Frequency Choice でコドン頻度比例サンプリング。

    Args:
        protein: タンパク質配列（末尾 _ 付き）。
        codon_freqs: get_codon_frequencies() の出力。

    Returns:
        最適化DNA配列。
    """
    return get_background_frequency_choice_sequence(protein, codon_freqs)


def predict_ct(
    protein: str,
    model,
    tokenizer,
    device: torch.device,
) -> str:
    """CodonTransformer（ベースまたはファインチューニング済み）で予測。

    Args:
        protein: タンパク質配列（末尾 _ 付き）。
        model: 推論用モデル。
        tokenizer: トークナイザー。
        device: 使用デバイス。

    Returns:
        予測DNA配列。
    """
    # 評価時は再現性のため deterministic=True（デフォルト）
    result = predict_dna_sequence(
        protein=protein,
        organism=ORGANISM,
        device=device,
        model=model,
        tokenizer=tokenizer,
        match_protein=True,
    )
    return result.predicted_dna


# ---------------------------------------------------------------------------
# メトリクス計算
# ---------------------------------------------------------------------------

def count_cis_elements(dna: str) -> int:
    """cis-element（発現阻害配列）の出現数を数える。

    Args:
        dna: DNA配列。

    Returns:
        検出されたcis-elementの総数。
    """
    count = 0
    for name, pattern in CIS_ELEMENT_PATTERNS.items():
        if pattern is None:
            continue
        count += len(re.findall(pattern, dna, re.IGNORECASE))
    return count


MFE_MAX_NT = 3000  # これ以上の長さはMFE計算をスキップ（計算量が指数的に増加）


def calc_mfe(dna: str) -> float:
    """RNA二次構造の最小自由エネルギー（MFE）を計算。

    Args:
        dna: DNA配列（RNAに変換して計算）。MFE_MAX_NT以上はスキップ。

    Returns:
        MFE値 (kcal/mol)。スキップ/計算失敗時は 0.0。
    """
    if len(dna) > MFE_MAX_NT:
        return 0.0
    rna = dna.replace("T", "U")
    try:
        _, mfe = RNA.fold(rna)
        return round(mfe, 2)
    except Exception:
        return 0.0


def calc_metrics(
    dna: str,
    csi_weights: dict[str, float],
    codon_freqs: dict,
) -> dict[str, float]:
    """1配列の全メトリクスを計算。

    Args:
        dna: DNA配列。
        csi_weights: get_CSI_weightsの出力。
        codon_freqs: get_codon_frequenciesの出力。

    Returns:
        メトリクス辞書。
    """
    gc = get_GC_content(dna)
    csi = get_CSI_value(dna, csi_weights)

    try:
        cfd = get_cfd(dna, codon_freqs)
    except Exception:
        cfd = 0.0

    try:
        minmax_vals = get_min_max_percentage(dna, codon_freqs, window_size=18)
        minmax_mean = sum(minmax_vals) / len(minmax_vals) if minmax_vals else 0.0
    except Exception:
        minmax_mean = 0.0

    cis = count_cis_elements(dna)
    mfe = calc_mfe(dna)

    try:
        complexity = get_sequence_complexity(dna)
    except Exception:
        complexity = 0.0

    return {
        "gc_percent": round(gc, 2),
        "csi": round(csi, 4),
        "cfd": round(cfd, 4),
        "minmax_mean": round(minmax_mean, 4),
        "cis_elements": cis,
        "mfe": mfe,
        "complexity": round(complexity, 4),
    }


# ---------------------------------------------------------------------------
# モデル読み込み
# ---------------------------------------------------------------------------

def load_ct_model(model_path: str | None, device: torch.device):
    """CodonTransformerモデルを読み込む（CT API使用）。

    Args:
        model_path: ローカルパス (.pt/.ckpt)、HuggingFace ID。Noneでベースモデル。
        device: 使用デバイス。

    Returns:
        読み込み済みモデル。
    """
    if model_path:
        print(f"  モデル読み込み: {model_path}")
    else:
        print("  モデル読み込み: HuggingFace (adibvafa/CodonTransformer)")
    return ct_load_model(model_path=model_path, device=device)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(description="モデル比較評価")
    parser.add_argument(
        "--model", default=None,
        help="ChlamyCodonTransformerのパス (省略時はbase CTのみ評価)"
    )
    parser.add_argument(
        "--genes", nargs="*",
        help="評価対象の遺伝子名リスト (省略時は全遺伝子)"
    )
    args = parser.parse_args()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"デバイス: {device}")

    # --- データ読み込み ---
    print("\n[1] C. reinhardtii 配列を読み込み中...")
    df = pd.read_csv(DATA_CSV)
    cr_df = df[df["source_species"] == "Chlamydomonas reinhardtii"].copy()
    print(f"  C. reinhardtii遺伝子数: {len(cr_df)}")

    if args.genes:
        cr_df = cr_df[cr_df["gene"].isin(args.genes)]
        print(f"  フィルタ後: {len(cr_df)} 遺伝子")

    # --- コドン頻度表構築 (CodonTransformer API) ---
    print("\n[2] コドン頻度表を構築中...")
    # GTG開始コドンはget_codon_frequenciesが非対応のためATG開始配列のみ使用
    atg_mask = cr_df["dna"].str.startswith("ATG")
    atg_df = cr_df[atg_mask]
    csi_weights = get_CSI_weights(atg_df["dna"].tolist())
    codon_freqs = get_codon_frequencies(
        atg_df["dna"].tolist(),
        atg_df["protein"].tolist(),
    )
    print(f"  コドン頻度表完成: {len(codon_freqs)} アミノ酸")

    # --- モデル読み込み ---
    print("\n[3] モデルを読み込み中...")
    tokenizer = load_tokenizer()

    base_model = load_ct_model(None, device)

    chlamyct_model = None
    if args.model:
        print("  ChlamyCodonTransformerを読み込み中...")
        model_path = Path(args.model)
        if model_path.is_dir() and (model_path / "config.json").exists():
            # HuggingFace format directory
            from transformers import BigBirdForMaskedLM as _BigBird
            chlamyct_model = _BigBird.from_pretrained(str(model_path), local_files_only=True)
            chlamyct_model.eval()
            chlamyct_model.to(device)
            print(f"  ✓ ChlamyCT (HF dir): {sum(p.numel() for p in chlamyct_model.parameters()):,} params")
        else:
            chlamyct_model = load_ct_model(args.model, device)
    else:
        # Auto-detect default path
        default_path = PROJECT_ROOT / "models" / "ChlamyCodonTransformer"
        if (default_path / "config.json").exists():
            from transformers import BigBirdForMaskedLM as _BigBird
            print(f"  ChlamyCT自動検出: {default_path}")
            chlamyct_model = _BigBird.from_pretrained(str(default_path), local_files_only=True)
            chlamyct_model.eval()
            chlamyct_model.to(device)
            print(f"  ✓ ChlamyCT: {sum(p.numel() for p in chlamyct_model.parameters()):,} params")
        else:
            print("  --model 未指定 & デフォルトパスなし: ChlamyCTはスキップ")

    # --- 評価ループ ---
    print(f"\n[4] {len(cr_df)} 遺伝子を評価中...")
    records: list[dict] = []

    strategies = ["natural", "base_ct", "cai_max", "bfc"]
    if chlamyct_model is not None:
        strategies.insert(2, "chlamyct")

    # CodonTransformerの最大入力長 (2046 aa + stop)
    CT_MAX_AA = 2046

    for _, row in cr_df.iterrows():
        gene      = row["gene"]
        natural   = row["dna"]
        protein   = row["protein"]
        length_aa = len(protein.rstrip("_"))

        print(f"  {gene} ({len(natural)} nt)...", end=" ", flush=True)

        # CT予測: 最大長を超える場合はスキップ
        ct_skip = length_aa > CT_MAX_AA
        if ct_skip:
            print(f"[CT skip: {length_aa}aa>{CT_MAX_AA}aa] ", end="", flush=True)

        dna_seqs: dict[str, str] = {
            "natural": natural,
            "base_ct": predict_ct(protein, base_model, tokenizer, device) if not ct_skip else natural,
            "cai_max": optimize_cai_max(protein, codon_freqs),
            "bfc":     optimize_bfc(protein, codon_freqs),
        }
        if chlamyct_model is not None:
            dna_seqs["chlamyct"] = (
                predict_ct(protein, chlamyct_model, tokenizer, device) if not ct_skip else natural
            )

        for strategy in strategies:
            dna = dna_seqs[strategy]
            metrics = calc_metrics(dna, csi_weights, codon_freqs)
            records.append({
                "gene":     gene,
                "strategy": strategy,
                "dna":      dna,
                "ct_skipped": ct_skip and strategy in ("base_ct", "chlamyct"),
                **metrics,
            })

        gcs = {s: calc_metrics(dna_seqs[s], csi_weights, codon_freqs)["gc_percent"]
               for s in strategies}
        gc_str = " | ".join(f"{s}={v:.1f}%" for s, v in gcs.items())
        print(f"GC: {gc_str}")

    # --- 結果保存 ---
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n結果保存: {OUTPUT_CSV} ({len(results_df)} 行)")

    # --- 集計サマリー ---
    print("\n" + "=" * 60)
    print("戦略別メトリクス (平均値)")
    print("=" * 60)
    summary = results_df.groupby("strategy")[
        ["gc_percent", "csi", "cfd", "minmax_mean", "cis_elements", "mfe", "complexity"]
    ].mean().round(3)
    print(summary.to_string())

    # --- 可視化 ---
    print("\n[5] 図を生成中...")
    _plot_results(results_df)
    print(f"図を保存: {FIGURE_DIR}/")


def _plot_results(df: pd.DataFrame) -> None:
    """メトリクス比較図を生成・保存。

    Args:
        df: compare_models の出力DataFrame。
    """
    metrics = [
        ("gc_percent",   "GC含量 (%)",   [34],      "赤点線=目標34%"),
        ("csi",          "CSI",           None,      None),
        ("cfd",          "CFD",           None,      None),
        ("minmax_mean",  "%MinMax (平均)", None,     None),
        ("cis_elements", "Cis-elements数", [0],      None),
        ("mfe",          "MFE (kcal/mol)", None,     None),
        ("complexity",   "Sequence Complexity", None, None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("コドン最適化戦略比較: C. reinhardtii葉緑体遺伝子", fontsize=14)

    palette = {
        "natural":  "#2ecc71",
        "base_ct":  "#3498db",
        "chlamyct": "#e74c3c",
        "cai_max":  "#f39c12",
        "bfc":      "#9b59b6",
    }
    # 存在する戦略のみ
    present = df["strategy"].unique().tolist()
    pal = {k: v for k, v in palette.items() if k in present}

    # 余ったサブプロットを非表示にする
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    for ax, (col, ylabel, hlines, note) in zip(axes.flat, metrics):
        sns.boxplot(
            data=df, x="strategy", y=col,
            palette=pal, order=[s for s in palette if s in present],
            ax=ax, width=0.6,
        )
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        if hlines:
            for h in hlines:
                ax.axhline(h, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        if note:
            ax.text(0.02, 0.97, note, transform=ax.transAxes,
                    fontsize=8, va="top", color="red")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # GC% ヒストグラム（戦略ごと）
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in [s for s in palette if s in present]:
        subset = df[df["strategy"] == strategy]["gc_percent"]
        ax.hist(subset, bins=20, alpha=0.6, label=strategy, color=palette[strategy])
    ax.axvline(34, color="black", linestyle="--", linewidth=2, label="目標GC% (34%)")
    ax.set_xlabel("GC含量 (%)")
    ax.set_ylabel("遺伝子数")
    ax.set_title("GC含量分布 (戦略別)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gc_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  model_comparison_boxplot.png")
    print(f"  gc_distribution.png")


if __name__ == "__main__":
    main()
