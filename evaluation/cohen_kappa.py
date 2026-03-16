"""
Inter-annotator Agreement: Human vs LLM Judge
Computes Cohen's Kappa, percentage agreement, and confusion matrix
for summary-based and theme-based annotation columns.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import numpy as np
import sys

# ── Load data ──────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/judged-results.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows from '{csv_path}'\n")

# ── Normalisation helpers ──────────────────────────────────────────────────────
def norm_bool(series):
    """Convert TRUE/FALSE (string or bool) → 1/0."""
    return series.astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0})

def norm_yesno(series):
    """Convert Yes/No (string) → 1/0."""
    return series.astype(str).str.strip().str.capitalize().map({"Yes": 1, "No": 0})

# ── Define column pairs ────────────────────────────────────────────────────────
pairs = {
    "Summary-based (llm_summ)": {
        "human_col": "llm_summ_human",   # TRUE / FALSE
        "llm_col":   "llm_summ_llm",     # Yes  / No
        "human_norm": norm_bool,
        "llm_norm":   norm_bool,
    },
    "Theme-based (llm_theme)": {
        "human_col": "llm_theme_human",  # TRUE / FALSE
        "llm_col":   "llm_theme_llm",    # Yes  / No
        "human_norm": norm_bool,
        "llm_norm":   norm_bool,
    },
}

# ── Compute agreement ──────────────────────────────────────────────────────────
for label, cfg in pairs.items():
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    # Check columns exist
    missing = [c for c in [cfg["human_col"], cfg["llm_col"]] if c not in df.columns]
    if missing:
        print(f"  [WARNING] Missing columns: {missing}. Skipping.\n")
        continue

    human = cfg["human_norm"](df[cfg["human_col"]])
    llm   = cfg["llm_norm"](df[cfg["llm_col"]])

    # Drop rows where either annotation is NaN after mapping
    valid = pd.DataFrame({"human": human, "llm": llm}).dropna()
    n_dropped = len(df) - len(valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with unmappable values.")

    h = valid["human"].astype(int)
    l = valid["llm"].astype(int)

    # ── Metrics ──
    pct_agree = (h == l).mean() * 100
    kappa     = cohen_kappa_score(h, l)

    print(f"  N (valid pairs)        : {len(valid)}")
    print(f"  % Agreement            : {pct_agree:.2f}%")
    print(f"  Cohen's Kappa          : {kappa:.4f}")

    # Kappa interpretation
    if kappa < 0:
        interp = "Poor (less than chance)"
    elif kappa < 0.20:
        interp = "Slight"
    elif kappa < 0.40:
        interp = "Fair"
    elif kappa < 0.60:
        interp = "Moderate"
    elif kappa < 0.80:
        interp = "Substantial"
    else:
        interp = "Almost perfect"
    print(f"  Kappa interpretation   : {interp}")

    # ── Confusion matrix ──
    cm = confusion_matrix(h, l, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Human=FALSE", "Human=TRUE"],
        columns=["LLM=No", "LLM=Yes"]
    )
    print("\n  Confusion Matrix (rows=Human, cols=LLM):")
    print(cm_df.to_string(index=True))

    # ── Per-class report ──
    print("\n  Classification Report (LLM vs Human as reference):")
    report = classification_report(h, l, target_names=["FALSE/No", "TRUE/Yes"],
                                   digits=4, zero_division=0)
    # indent
    for line in report.splitlines():
        print("  " + line)

    print()

print("Done.")
