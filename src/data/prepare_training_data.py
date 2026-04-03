"""Preprocess chloroplast CDS for CodonTransformer fine-tuning.

Loads green_algae_chloroplast_cds.csv, calculates CSI (Codon Sequence Index)
based on C. reinhardtii chloroplast codon usage, applies expression weights
via row duplication, then outputs CodonTransformer-compatible training_data.json.

Usage:
    python src/data/prepare_training_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from CodonTransformer.CodonData import prepare_training_data
from CodonTransformer.CodonEvaluation import get_CSI_weights, get_CSI_value

# --- Constants ---

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "green_algae_chloroplast_cds.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "training_data.json"

MIN_DNA_LEN = 30
CSI_THRESHOLD = 0.3  # 緩めの閾値（多様性確保）

TIER1_GENES = {"rbcL", "psbA", "psaA", "atpA", "psbD", "psbC"}
TIER2_GENES = {
    "psbB", "psbE", "psbF", "psbH", "psbI", "psbJ", "psbK", "psbL",
    "psbM", "psbN", "psbT", "psaB", "psaC", "petA", "petB", "petD",
    "atpB", "atpE", "atpH", "atpI",
}
TIER_WEIGHTS = {1: 5, 2: 2, 3: 1}


def validate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """バリデーション: 長さ・形式チェック。

    Args:
        df: 入力DataFrame。

    Returns:
        バリデーション通過した行のみのDataFrame。
    """
    n_before = len(df)

    mask = (
        (df["dna"].str.len() >= MIN_DNA_LEN) &
        (df["dna"].str.len() % 3 == 0) &
        (df["protein"].str.endswith("_"))
    )
    df = df[mask].copy()

    removed = n_before - len(df)
    if removed:
        print(f"  バリデーション除外: {removed} 件")
    return df


def compute_csi_weights(df: pd.DataFrame) -> dict[str, float]:
    """C. reinhardtiiのATG開始CDSからCSI重みを計算（CodonTransformer API使用）。

    Args:
        df: 全CDS DataFrame（source_species, dna カラム必須）。

    Returns:
        CSI重み辞書（get_CSI_value() に渡す用）。
    """
    cr_df = df[df["source_species"] == "Chlamydomonas reinhardtii"]
    if cr_df.empty:
        print("  [警告] C. reinhardtii CDSが見つかりません。全種で重みを計算します。")
        cr_df = df

    # GTG開始コドンはget_CSI_weightsが非対応のためATG開始配列のみ使用
    atg_seqs = cr_df[cr_df["dna"].str.startswith("ATG")]["dna"].tolist()
    if not atg_seqs:
        print("  [警告] ATG開始配列がありません。全配列を使用します。")
        atg_seqs = cr_df["dna"].tolist()

    return get_CSI_weights(atg_seqs)


def assign_expression_tier(gene: str) -> int:
    """遺伝子名から発現Tierを返す。

    Args:
        gene: 遺伝子名。

    Returns:
        1 (高発現), 2 (中発現), 3 (その他)。
    """
    gene_lower = gene.lower()
    if gene_lower in {g.lower() for g in TIER1_GENES}:
        return 1
    if gene_lower in {g.lower() for g in TIER2_GENES}:
        return 2
    return 3


def apply_expression_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Tier重みに応じて行を複製し過剰サンプリング。

    Args:
        df: gene カラムを含むDataFrame。

    Returns:
        重み付き（複製済み）DataFrame。
    """
    rows: list[pd.DataFrame] = []
    tier_counts = {1: 0, 2: 0, 3: 0}

    for tier, weight in TIER_WEIGHTS.items():
        mask = df["gene"].apply(assign_expression_tier) == tier
        subset = df[mask]
        tier_counts[tier] = len(subset)
        if not subset.empty:
            rows.append(pd.concat([subset] * weight, ignore_index=True))

    result = pd.concat(rows, ignore_index=True)

    print(f"  発現Tier分布 (重み適用前):")
    print(f"    Tier1 (×5): {tier_counts[1]} 件 → {tier_counts[1] * 5} 件")
    print(f"    Tier2 (×2): {tier_counts[2]} 件 → {tier_counts[2] * 2} 件")
    print(f"    Tier3 (×1): {tier_counts[3]} 件 → {tier_counts[3]} 件")

    return result


def main() -> None:
    """メインエントリーポイント。"""
    print("=" * 50)
    print("Phase 2: 訓練データ前処理")
    print("=" * 50)

    # Step 1: 読み込み
    print(f"\n[1] CSVを読み込み中: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  読み込み: {len(df)} 行")

    # Step 2: バリデーション
    print("\n[2] バリデーション中...")
    df = validate_sequences(df)
    print(f"  バリデーション通過: {len(df)} 行")

    # Step 3: CSI計算 (CodonTransformer API使用)
    print("\n[3] CSI（コドン適応指数）を計算中...")
    csi_weights = compute_csi_weights(df)
    print(f"  CSI重み計算完了 (CodonTransformer API)")

    df["csi"] = df["dna"].apply(lambda d: get_CSI_value(d, csi_weights))

    csi_stats = df["csi"].describe()
    print(f"  CSI統計: mean={csi_stats['mean']:.3f}, "
          f"min={csi_stats['min']:.3f}, max={csi_stats['max']:.3f}")

    # CSIフィルタ
    n_before = len(df)
    df = df[df["csi"] >= CSI_THRESHOLD].copy()
    print(f"  CSI≥{CSI_THRESHOLD} フィルタ後: {len(df)} 行 (除外: {n_before - len(df)} 件)")

    # Step 4: 発現重み付け
    print("\n[4] 発現重み付け（行複製）中...")
    n_before_weight = len(df)
    df_weighted = apply_expression_weights(df)
    print(f"  重み適用後: {len(df_weighted)} 行 ({n_before_weight} → {len(df_weighted)})")

    # Step 5: prepare_training_data 実行
    print(f"\n[5] CodonTransformer形式に変換中...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    training_df = df_weighted[["dna", "protein", "organism"]].copy()
    prepare_training_data(training_df, str(OUTPUT_JSON), shuffle=True)

    # 出力確認
    with open(OUTPUT_JSON) as f:
        lines = f.readlines()
    n_output = len(lines)

    print(f"\n{'=' * 50}")
    print(f"完了: {OUTPUT_JSON}")
    print(f"{'=' * 50}")
    print(f"入力配列数:     {len(pd.read_csv(INPUT_CSV))}")
    print(f"バリデーション後: {n_before_weight}")
    print(f"重み適用後:      {len(df_weighted)}")
    print(f"出力行数:        {n_output}")
    print(f"\nサンプル出力:")
    sample = json.loads(lines[0])
    print(f"  organism: {sample['organism']}")
    print(f"  codons (先頭60文字): {sample['codons'][:60]}...")


if __name__ == "__main__":
    main()
