"""ChlamyDesign environment check"""
import sys

def check_imports():
    checks = []
    for name, imp in [
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("CodonTransformer", "CodonTransformer"),
        ("Biopython", "Bio"),
        ("DNAchisel", "dnachisel"),
        ("Pandas", "pandas"),
        ("NumPy", "numpy"),
    ]:
        try:
            __import__(imp)
            checks.append(f"  [OK] {name}")
        except ImportError:
            checks.append(f"  [NG] {name}")
    return checks

if __name__ == "__main__":
    print("=" * 40)
    print("  ChlamyDesign Environment Check")
    print("=" * 40)
    for r in check_imports():
        print(r)
    ng = sum(1 for r in check_imports() if "[NG]" in r)
    if ng:
        print(f"\n  {ng} package(s) missing.")
    else:
        print("\n  All packages OK!")
