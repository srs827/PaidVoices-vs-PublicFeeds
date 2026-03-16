"""
Temporal Distribution Analysis & Chi-Square Test
Compares the monthly distribution of a 20K sample vs. full 1.3M Bluesky corpus.
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
FULL_PATH   = "data/bluesky_pipeline/All_Data_Jul_2024-Sept_2025.csv"    # path to full 1.3M dataset
SAMPLE_PATH = "data/bluesky_pipeline/combined_deduped_sbert080.csv"  # path to 20K sample

df_full   = pd.read_csv(FULL_PATH,   usecols=["indexed_at"])
df_sample = pd.read_csv(SAMPLE_PATH, usecols=["indexed_at"])

# ── 2. PARSE TIMESTAMPS & EXTRACT YEAR-MONTH ─────────────────────────────────
for df in [df_full, df_sample]:
    df["indexed_at"] = pd.to_datetime(df["indexed_at"], utc=True, errors="coerce")
    df["year_month"] = df["indexed_at"].dt.to_period("M")

# ── 3. MONTHLY COUNTS & PROPORTIONS ──────────────────────────────────────────
full_counts   = df_full["year_month"].value_counts().sort_index()
sample_counts = df_sample["year_month"].value_counts().sort_index()

# Align on the same months (fill missing months with 0)
all_months = full_counts.index.union(sample_counts.index)
full_counts   = full_counts.reindex(all_months, fill_value=0)
sample_counts = sample_counts.reindex(all_months, fill_value=0)

full_props   = full_counts   / full_counts.sum()
sample_props = sample_counts / sample_counts.sum()

# ── 4. SUMMARY STATISTICS ─────────────────────────────────────────────────────
print("=" * 60)
print("TEMPORAL DISTRIBUTION SUMMARY")
print("=" * 60)
print(f"Full corpus  : {len(df_full):,} posts  |  {df_full['indexed_at'].min().date()} → {df_full['indexed_at'].max().date()}")
print(f"Sample       : {len(df_sample):,} posts  |  {df_sample['indexed_at'].min().date()} → {df_sample['indexed_at'].max().date()}")
print(f"Months covered (full)  : {full_counts.index.min()} → {full_counts.index.max()}")
print()

# Monthly breakdown table
table = pd.DataFrame({
    "full_count":   full_counts,
    "sample_count": sample_counts,
    "full_%":       (full_props   * 100).round(2),
    "sample_%":     (sample_props * 100).round(2),
    "abs_diff_%":   ((sample_props - full_props).abs() * 100).round(2),
})
print(table.to_string())
print()

# ── 5. CHI-SQUARE GOODNESS-OF-FIT TEST ───────────────────────────────────────
# Expected counts in sample = full proportions × sample size
# Only include months where expected count ≥ 5 (chi-square assumption)
expected = full_props * sample_counts.sum()
mask = expected >= 5

obs_filtered = sample_counts[mask].values
exp_filtered = expected[mask].values

chi2, p_value = chisquare(f_obs=obs_filtered, f_exp=exp_filtered)
n_bins = mask.sum()
dof    = n_bins - 1

print("=" * 60)
print("CHI-SQUARE GOODNESS-OF-FIT TEST")
print("=" * 60)
print(f"Months included (expected ≥ 5) : {n_bins}  (excluded: {(~mask).sum()})")
print(f"Chi-square statistic           : {chi2:.4f}")
print(f"Degrees of freedom             : {dof}")
print(f"p-value                        : {p_value:.4f}")
if p_value > 0.05:
    print("✓ Result: Cannot reject H₀ — sample is temporally representative (p > 0.05)")
else:
    print("✗ Result: Sample deviates significantly from full corpus (p ≤ 0.05)")
print()

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
months_str = [str(m) for m in all_months]
x = np.arange(len(months_str))
width = 0.4

# Top panel: raw counts
axes[0].bar(x - width/2, full_counts.values,   width, label="Full corpus",  color="#4C72B0", alpha=0.85)
axes[0].bar(x + width/2, sample_counts.values, width, label="Sample (20K)", color="#DD8452", alpha=0.85)
axes[0].set_ylabel("Post count")
axes[0].set_title("Monthly Post Counts: Full Corpus vs. Sample")
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

# Bottom panel: proportions
axes[1].plot(x, full_props.values   * 100, "o-", label="Full corpus",  color="#4C72B0", linewidth=2)
axes[1].plot(x, sample_props.values * 100, "s--", label="Sample (20K)", color="#DD8452", linewidth=2)
axes[1].set_ylabel("Proportion (%)")
axes[1].set_title("Monthly Proportions: Full Corpus vs. Sample")
axes[1].legend()
axes[1].set_xticks(x)
axes[1].set_xticklabels(months_str, rotation=45, ha="right", fontsize=8)

# Annotate chi-square result
annot = f"χ²({dof}) = {chi2:.2f},  p = {p_value:.3f}"
fig.text(0.99, 0.01, annot, ha="right", va="bottom", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig("temporal_distribution.png", dpi=150, bbox_inches="tight")
print("Plot saved → temporal_distribution.png")
