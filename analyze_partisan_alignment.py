"""
analyze_partisan_alignment.py

Correlates Pro-Climate ad saturation (from region_stance.csv) with 2024
presidential vote margins by state.

Outputs
-------
  partisan_alignment_scatter.pdf   — scatter plot with trend line + annotations
  partisan_alignment_stats.csv     — per-state merged table with alignment flag

Usage
-----
  python analyze_partisan_alignment.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
STANCE_FILE   = "region_stance.csv"
OUT_SCATTER   = "partisan_alignment_scatter.pdf"
OUT_CSV       = "partisan_alignment_stats.csv"
DPI           = 300
# ─────────────────────────────────────────────────────────────────────────────

# ── 2024 Presidential margins (Trump % - Harris %) ────────────────────────────
# Positive = Trump won, Negative = Harris won.
# Source: AP/NYT certified results (Dec 2024).
MARGINS_2024 = {
    "Alabama":              +27.9,
    "Alaska":               +13.2,
    "Arizona":               +5.5,
    "Arkansas":             +28.2,
    "California":           -20.3,
    "Colorado":             -11.1,
    "Connecticut":          -12.8,
    "Delaware":             -12.7,
    "District of Columbia": -76.2,
    "Florida":              +13.4,
    "Georgia":              +11.9,
    "Hawaii":               -29.8,
    "Idaho":                +38.9,
    "Illinois":             -14.1,
    "Indiana":              +18.4,
    "Iowa":                 +13.4,
    "Kansas":               +20.2,
    "Kentucky":             +30.4,
    "Louisiana":            +20.0,
    "Maine":                 -7.0,
    "Maryland":             -32.8,
    "Massachusetts":        -28.2,
    "Michigan":              +1.4,
    "Minnesota":             -5.2,
    "Mississippi":          +16.3,
    "Missouri":             +18.3,
    "Montana":              +20.2,
    "Nebraska":             +20.2,
    "Nevada":                -3.1,
    "New Hampshire":         -2.2,
    "New Jersey":            -6.1,
    "New Mexico":           -10.1,
    "New York":             -12.4,
    "North Carolina":        +3.2,
    "North Dakota":         +33.2,
    "Ohio":                 +11.0,
    "Oklahoma":             +32.7,
    "Oregon":               -16.0,
    "Pennsylvania":          +2.1,
    "Rhode Island":         -19.4,
    "South Carolina":       +13.2,
    "South Dakota":         +26.3,
    "Tennessee":            +29.0,
    "Texas":                +14.2,
    "Utah":                 +13.4,
    "Vermont":              -35.3,
    "Virginia":              -6.4,
    "Washington":           -17.8,
    "Washington, District of Columbia": -76.2,
    "West Virginia":        +38.7,
    "Wisconsin":             +0.9,
    "Wyoming":              +43.3,
}

# ── Load stance data ──────────────────────────────────────────────────────────
stance = pd.read_csv(STANCE_FILE)
stance = stance.rename(columns={"region": "state"})
stance["state"] = stance["state"].replace(
    "Washington, District of Columbia", "District of Columbia"
)

# ── Merge ─────────────────────────────────────────────────────────────────────
margins = pd.DataFrame(
    MARGINS_2024.items(), columns=["state", "trump_margin_2024"]
)
df = stance.merge(margins, on="state", how="inner")
df = df.dropna(subset=["pct_pro_climate", "trump_margin_2024"])

# ── Derived columns ───────────────────────────────────────────────────────────
# alignment: does the dominant ad lean match the partisan lean?
#   D state (margin < 0) + more Pro-Climate (sat > 50) = aligned
#   R state (margin > 0) + more Pro-Energy  (sat < 50) = aligned
df["partisan_lean"] = df["trump_margin_2024"].apply(
    lambda m: "D" if m < -5 else ("R" if m > 5 else "Swing")
)
df["ad_lean"] = df["pct_pro_climate"].apply(
    lambda p: "Pro-Climate" if p > 50 else ("Pro-Energy" if p < 50 else "Balanced")
)
df["aligned"] = (
    ((df["partisan_lean"] == "D") & (df["ad_lean"] == "Pro-Climate")) |
    ((df["partisan_lean"] == "R") & (df["ad_lean"] == "Pro-Energy"))
)

# ── Correlation ───────────────────────────────────────────────────────────────
# Negate margin so x-axis reads left=Dem, right=Rep intuitively but we also
# report the raw Pearson r (negative r means more D → more Pro-Climate).
x = df["trump_margin_2024"].values
y = df["pct_pro_climate"].values

r, p_val = stats.pearsonr(x, y)
slope, intercept, _, _, se = stats.linregress(x, y)
r_sq = r ** 2

print(f"\nCorrelation results (n={len(df)})")
print(f"  Pearson r  : {r:.3f}")
print(f"  R²         : {r_sq:.3f}")
print(f"  p-value    : {p_val:.4f}")
print(f"  Slope      : {slope:.3f}  (% Pro-Climate per % Trump margin)")
print(f"  Intercept  : {intercept:.2f}")
print(f"\nAlignment summary")
print(df.groupby(["partisan_lean", "aligned"]).size().unstack(fill_value=0).to_string())

# ── Scatter plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

COLOR_MAP = {
    ("D", True):     ("#27ae60", "Dem-leaning, Pro-Climate ads (aligned)"),
    ("R", True):     ("#c0392b", "Rep-leaning, Pro-Energy ads (aligned)"),
    ("D", False):    ("#f39c12", "Dem-leaning, Pro-Energy ads (mismatch)"),
    ("R", False):    ("#8e44ad", "Rep-leaning, Pro-Climate ads (mismatch)"),
    ("Swing", True): ("#7f8c8d", "Swing state (aligned)"),
    ("Swing", False):("#7f8c8d", "Swing state (mismatch)"),
}

plotted_labels = {}
for _, row in df.iterrows():
    key = (row["partisan_lean"], row["aligned"])
    color, label = COLOR_MAP[key]
    marker = "o" if row["aligned"] else "X"
    if label not in plotted_labels:
        ax.scatter(
            row["trump_margin_2024"], row["pct_pro_climate"],
            c=color, s=70, marker=marker, alpha=0.85,
            edgecolors="white", linewidths=0.5, label=label, zorder=3
        )
        plotted_labels[label] = True
    else:
        ax.scatter(
            row["trump_margin_2024"], row["pct_pro_climate"],
            c=color, s=70, marker=marker, alpha=0.85,
            edgecolors="white", linewidths=0.5, zorder=3
        )

# State abbreviation labels
ABBREV = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
    "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
    "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
    "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
    "Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
    "District of Columbia":"DC","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}
for _, row in df.iterrows():
    abbr = ABBREV.get(row["state"], "")
    if not abbr:
        continue
    ax.annotate(
        abbr,
        (row["trump_margin_2024"], row["pct_pro_climate"]),
        fontsize=5.5, ha="center", va="bottom",
        xytext=(0, 5), textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        zorder=4,
    )

# Trend line
x_line = np.linspace(x.min() - 3, x.max() + 3, 200)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color="#555555", lw=1.2, ls="--", alpha=0.6, zorder=2)

# Reference lines
ax.axvline(0,  color="#aaaaaa", lw=0.8, ls=":", zorder=1)
ax.axhline(50, color="#aaaaaa", lw=0.8, ls=":", zorder=1)

# Quadrant labels
ax.text(-42, 96, "Dem state\nPro-Climate ads", fontsize=7.5, color="#27ae60",
        ha="left", va="top", alpha=0.7)
ax.text(38, 96,  "Rep state\nPro-Climate ads\n(mismatch)", fontsize=7.5, color="#8e44ad",
        ha="right", va="top", alpha=0.7)
ax.text(-42,  4, "Dem state\nPro-Energy ads\n(mismatch)", fontsize=7.5, color="#f39c12",
        ha="left", va="bottom", alpha=0.7)
ax.text(38,   4, "Rep state\nPro-Energy ads", fontsize=7.5, color="#c0392b",
        ha="right", va="bottom", alpha=0.7)

# Stats annotation
ax.text(
    0.02, 0.97,
    f"Pearson r = {r:.2f}   R² = {r_sq:.2f}   p = {p_val:.4f}   n = {len(df)}",
    transform=ax.transAxes, fontsize=8.5, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8)
)

ax.set_xlabel("← More Democratic       2024 Trump Margin (%)       More Republican →",
              fontsize=9.5)
ax.set_ylabel("Pro-Climate Ad Saturation (%)", fontsize=9.5)
ax.set_title("Pro-Climate Ad Saturation vs. 2024 Presidential Lean by State",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(-50, 55)
ax.set_ylim(-5, 105)
ax.tick_params(labelsize=8)

ax.legend(loc="center right", fontsize=7.5, framealpha=0.9, title="State / Ad alignment")

ax.text(
    0.5, -0.06,
    "Ad saturation from weighted regional delivery (700-ad sample). "
    "Election margins from AP/NYT 2024 certified results.",
    transform=ax.transAxes, ha="center", fontsize=7, color="#555555"
)

plt.tight_layout()
fig.savefig(OUT_SCATTER, dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved → {OUT_SCATTER}")

# ── Export CSV ────────────────────────────────────────────────────────────────
out_cols = ["state", "pct_pro_climate", "trump_margin_2024",
            "partisan_lean", "ad_lean", "aligned"]
df[out_cols].sort_values("trump_margin_2024").to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}")