"""
Joint Theme Analysis & Visualization
=====================================
Comprehensive analysis of joint clustering results

Inputs:
    - joint_clusters.csv   : full clustering output
    - joint_themes.csv     : theme metadata with framing dimensions
    - the above files are obtained by running joint_theme_analysis.py

Outputs (written to ANALYSIS_DIR):
    fig1_top15_meta.png                  — top Meta-dominant themes
    fig2_top15_bluesky.png               — top Bluesky-dominant themes
    fig3_diverging_shared.png            — shared themes: platform composition
    fig4_framing_heatmap.png             — framing dimensions by category
    fig5_framing_radar.png               — radar chart: framing fingerprints
    fig6_urgency_hope_scatter.png        — emotional register per theme
    fig7_economic_political_scatter.png  — problem framing space
    fig8_solution_rate.png               — % solution-present by category
    fig9_skew_vs_size_bubble.png         — platform skew vs cluster size
    fig10_framing_pca.png                — themes in 2D framing space
    fig11_text_length_by_category.png    — linguistic surface features
    fig12_platform_coverage.png          — how corpus is divided across categories

    (if stance data available)
    fig13_stance_by_category.png
    fig14_stance_per_theme_bluesky.png
    fig15_stance_per_theme_meta.png
    fig16_extreme_stance_themes.png

Dependencies:
    pip install pandas numpy matplotlib seaborn scikit-learn squarify
"""

import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

try:
    from sklearn.decomposition import PCA as skPCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False

# ===========================================================================
# CONFIG
# ===========================================================================

OUT_DIR      = Path("joint_analysis_outputs")
CLUSTERS_CSV = OUT_DIR / "joint_clusters.csv"
THEMES_CSV   = OUT_DIR / "joint_themes.csv"
META_STANCE_CSV    = Path("meta_stance.csv")
BLUESKY_STANCE_CSV = Path("bluesky_stance.csv")
# Optional: stance sample files (set to None to skip stance figures)

META_ID_COL    = "cid"
BLUESKY_ID_COL = "cid"
STANCE_COL     = "stance"
STANCE_ORDER   = ["pro-climate", "neutral", "pro-energy"]
STANCE_COLORS  = {
    "pro-climate": "#2ca02c",
    "neutral":     "#aec7e8",
    "pro-energy":  "#d62728",
}
 
DOMINANT_THRESHOLD  = 0.70
TOP_N               = 10
MIN_TEXTS_FRAMING   = 10   # min cluster size to include in framing figures
MIN_TEXTS_STANCE    = 5    # min stance-labelled texts per theme for stance figs
 
ANALYSIS_DIR = OUT_DIR / "analysis_figures"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
 
# Palette 
C_META    = "#E8534A"
C_BLUESKY = "#4C72B0"
C_SHARED  = "#2ca02c"
PALETTE   = {
    "meta_dominant":    C_META,
    "bluesky_dominant": C_BLUESKY,
    "shared":           C_SHARED,
    "noise":            "#cccccc",
}
 
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})
 
FRAMING_COLS = [
    "actor_institution", "problem_economic", "problem_moral",
    "problem_scientific", "problem_political",
    "temporal_future", "temporal_crisis",
    "emotional_urgency", "emotional_hope",
]
FRAMING_LABELS = {
    "actor_institution":  "Institutional\nactors",
    "problem_economic":   "Economic\nframing",
    "problem_moral":      "Moral\nframing",
    "problem_scientific": "Scientific\nframing",
    "problem_political":  "Political\nframing",
    "temporal_future":    "Future\noriented",
    "temporal_crisis":    "Crisis\nframing",
    "emotional_urgency":  "Urgency /\nalarm",
    "emotional_hope":     "Hope /\nsolution",
}
 
# ===========================================================================
# Helpers
# ===========================================================================
 
def _clean(s, maxlen=36):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r'^\*{1,2}["\']?|["\']?\*{1,2}$', '', s)
    s = re.sub(r'[*"\']+', '', s)
    s = s.strip()
    return s[:maxlen] + "…" if len(s) > maxlen else s
 
 
def _savefig(name):
    path = ANALYSIS_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
def _stacked_hbar(tbl, title, fname, subtitle=""):
    """100% horizontal stacked bar from a pct table with 'total' column."""
    cols = [s for s in STANCE_ORDER if s in tbl.columns]
    fig, ax = plt.subplots(figsize=(11, max(5, len(tbl) * 0.48)))
    left = np.zeros(len(tbl))
    y    = np.arange(len(tbl))
    for s in cols:
        vals = tbl[s].values
        ax.barh(y, vals, left=left, color=STANCE_COLORS[s],
                alpha=0.88, label=s, height=0.72)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(tbl.index.tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("% of stance-labelled texts within theme", fontsize=10)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    for i, (_, r) in enumerate(tbl.iterrows()):
        ax.text(101, i, f"n={int(r['total'])}", va="center",
                fontsize=7, color="#555")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    _savefig(fname)
 
 
# ===========================================================================
# Load data
# ===========================================================================
 
print("Loading data …")
df        = pd.read_csv(CLUSTERS_CSV, low_memory=False)
themes_df = pd.read_csv(THEMES_CSV)
 
def _clean_raw(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r'^\*{1,2}["\']?|["\']?\*{1,2}$', '', s)
    s = re.sub(r'[*"\']+', '', s)
    return s.strip()
 
themes_df["theme_label"] = themes_df["theme_label"].apply(_clean_raw)
df["theme_label"]        = df["theme_label"].apply(_clean_raw)
 
clustered   = df[df["cluster_id"] >= 0].copy()
coherent_t  = themes_df[themes_df.get("coherent", pd.Series([True]*len(themes_df))) == True].copy()
framing_ok  = [c for c in FRAMING_COLS if c in themes_df.columns]
 
print(f"  {len(df):,} texts · {len(themes_df)} themes · "
      f"{len(coherent_t)} coherent · {len(framing_ok)} framing dims available")
 
# ===========================================================================
# Load stance sample 
# ===========================================================================
 
stance_available = False
stance_df = pd.DataFrame()
 
def _load_stance(path, id_col, platform):
    if path is None or not path.exists():
        return pd.DataFrame()
    src = pd.read_csv(path, low_memory=False)
    src.columns = src.columns.str.lower().str.strip()
    id_col = id_col.lower()
    if id_col not in src.columns or STANCE_COL not in src.columns:
        return pd.DataFrame()
    src = src.rename(columns={id_col: "doc_id"})
    src["doc_id"]   = src["doc_id"].astype(str)
    src["platform"] = platform
    src[STANCE_COL] = src[STANCE_COL].str.strip().str.lower()
    src = src[src[STANCE_COL].isin(STANCE_ORDER)]
    return src[["doc_id", "platform", STANCE_COL]].drop_duplicates("doc_id")
 
meta_s    = _load_stance(META_STANCE_CSV,    META_ID_COL,    "meta")
bluesky_s = _load_stance(BLUESKY_STANCE_CSV, BLUESKY_ID_COL, "bluesky")
parts     = [p for p in [meta_s, bluesky_s] if not p.empty]
 
if parts:
    stance_raw = pd.concat(parts, ignore_index=True)
    df["doc_id"] = df["doc_id"].astype(str)
    join_cols    = ["doc_id", "cluster_id", "theme_label", "theme_category"]
    stance_df    = stance_raw.merge(
        df[[c for c in join_cols if c in df.columns]].drop_duplicates("doc_id"),
        on="doc_id", how="left"
    )
    stance_clustered = stance_df[
        stance_df["cluster_id"].notna() &
        (stance_df["cluster_id"] != -1) &
        stance_df["theme_label"].notna() &
        (stance_df["theme_label"].astype(str).str.strip() != "") &
        (stance_df["theme_label"].astype(str).str.lower() != "nan") &
        (stance_df["theme_label"].astype(str).str.lower() != "unassigned")
    ]
    stance_available = len(stance_clustered) > 0
    print(f"  Stance sample: {len(stance_df):,} texts "
          f"({len(stance_clustered):,} in real clusters)")
 
# ===========================================================================
# Fig 1 & 2: Top-15 per platform
# ===========================================================================
 
print("\nFig 1 & 2 — top-15 per platform …")
 
for cat, color, fname, title, count_col in [
    ("meta_dominant",    C_META,    "fig1_top10_meta.png",
     "Top 10 Meta-dominant themes", "n_meta"),
    ("bluesky_dominant", C_BLUESKY, "fig2_top10_bluesky.png",
     "Top 10 Bluesky-dominant themes", "n_bluesky"),
]:
    top = (
        themes_df[(themes_df["category"] == cat) &
                  themes_df["theme_label"].notna() &
                  (themes_df["theme_label"] != "")]
        .nlargest(TOP_N, "n_total").copy()
    )
    top["theme_label"] = top["theme_label"].apply(_clean)
 
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.48)))
    y = np.arange(len(top))
    ax.barh(y, top[count_col].values, color=color, alpha=0.85, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(top["theme_label"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of texts", fontsize=10)
    ax.set_title(f"{title}  (by total cluster size)", fontsize=11)
    for i, v in enumerate(top[count_col].values):
        ax.text(v + 1, i, f"{int(v):,}", va="center", fontsize=8, color="#333")
    plt.tight_layout()
    _savefig(fname)
 
# ===========================================================================
# Fig 3: Diverging bar — shared themes (% Meta vs % Bluesky)
# ===========================================================================
 
print("Fig 3 — diverging bar (shared themes) …")
 
shared = (
    themes_df[(themes_df["category"] == "shared") &
              themes_df["theme_label"].notna() &
              (themes_df["theme_label"] != "")]
    .sort_values("n_total", ascending=False).head(20).copy()
)
shared["theme_label"] = shared["theme_label"].apply(_clean)
shared["pct_meta"]    = shared["n_meta"]    / shared["n_total"] * 100
shared["pct_bluesky"] = shared["n_bluesky"] / shared["n_total"] * 100
 
fig, ax = plt.subplots(figsize=(11, max(6, len(shared) * 0.48)))
y = np.arange(len(shared))
ax.barh(y,  shared["pct_meta"].values,    color=C_META,    alpha=0.85,
        label="Meta ads",     height=0.65)
ax.barh(y, -shared["pct_bluesky"].values, color=C_BLUESKY, alpha=0.85,
        label="Bluesky posts", height=0.65)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(shared["theme_label"].values, fontsize=9)
ax.invert_yaxis()
xlim = max(shared["pct_meta"].max(), shared["pct_bluesky"].max()) + 5
ax.set_xlim(-xlim, xlim)
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{abs(x):.0f}%"))
ax.set_xlabel("← Bluesky %          Meta % →", fontsize=10)
ax.set_title("Shared themes — Relative platform composition\n"
             "(Top 20 by total size; Bars show % of texts per theme)",
             fontsize=11)
for i, (_, r) in enumerate(shared.iterrows()):
    ax.text(xlim - 0.3, i, f"n={int(r['n_total']):,}",
            va="center", ha="right", fontsize=7, color="#555")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
_savefig("fig3_diverging_shared.png")
 
# ===========================================================================
# Fig 4: Framing heatmap by category
# ===========================================================================
 
if framing_ok:
    print("Fig 4 — framing heatmap …")
 
    hm_data = (
        coherent_t.groupby("category")[framing_ok].mean()
        .rename(columns=FRAMING_LABELS)
    )
    row_order = [r for r in ["bluesky_dominant","shared","meta_dominant"]
                 if r in hm_data.index]
    hm_data = hm_data.reindex(row_order)
    hm_data.index = [i.replace("_"," ").title() for i in hm_data.index]
 
    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Mean framing dimension scores by cluster category\n"
                 "(coherent clusters only — higher = stronger presence of dimension)",
                 fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster category", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    _savefig("fig4_framing_heatmap.png")
 
# ===========================================================================
# Fig 5: Framing radar chart — fingerprint per category
# Reveals the distinct "framing signature" of paid ads vs organic posts
# ===========================================================================
 
if framing_ok and len(framing_ok) >= 4:
    print("Fig 5 — framing radar chart …")
 
    radar_data = coherent_t.groupby("category")[framing_ok].mean()
    cats_radar = [c for c in ["meta_dominant","shared","bluesky_dominant"]
                  if c in radar_data.index]
    labels_r   = [FRAMING_LABELS[c].replace("\n", " ") for c in framing_ok]
    N          = len(framing_ok)
    angles     = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles    += angles[:1]   # close the polygon
 
    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
 
    radar_colors = {
        "meta_dominant":    C_META,
        "shared":           C_SHARED,
        "bluesky_dominant": C_BLUESKY,
    }
    radar_labels = {
        "meta_dominant":    "Meta-dominant (paid)",
        "shared":           "Shared",
        "bluesky_dominant": "Bluesky-dominant (organic)",
    }
    for cat in cats_radar:
        vals  = radar_data.loc[cat, framing_ok].tolist()
        vals += vals[:1]
        color = radar_colors[cat]
        ax.plot(angles, vals, color=color, linewidth=2, label=radar_labels[cat])
        ax.fill(angles, vals, color=color, alpha=0.08)
 
    ax.set_title("Framing by cluster category\n"
                 "(paid ads vs organic posts — average across coherent clusters)",
                 fontsize=11, pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=9)
    plt.tight_layout()
    _savefig("fig5_framing_radar.png")
 
# ===========================================================================
# Fig 6: Urgency vs Hope scatter — emotional register of climate discourse
# ===========================================================================
 
if "emotional_urgency" in themes_df.columns and "emotional_hope" in themes_df.columns:
    print("Fig 6 — urgency vs hope scatter …")
 
    emo = coherent_t[
        coherent_t["n_total"] >= MIN_TEXTS_FRAMING
    ].copy()
    emo["theme_label"] = emo["theme_label"].apply(_clean)
 
    fig, ax = plt.subplots(figsize=(9, 7))
 
    for cat in ["bluesky_dominant", "shared", "meta_dominant"]:
        sub = emo[emo["category"] == cat]
        ax.scatter(
            sub["emotional_urgency"], sub["emotional_hope"],
            s=sub["n_total"] / emo["n_total"].max() * 400 + 20,
            color=PALETTE[cat], alpha=0.7, zorder=3,
            label=cat.replace("_"," ").title(),
        )
 
    # Quadrant lines
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
 
    # Quadrant labels
    for (x, y, txt) in [
        (0.02, 0.97, "Low urgency\nHigh hope"),
        (0.75, 0.97, "High urgency\nHigh hope\n(constructive alarm)"),
        (0.02, 0.03, "Low urgency\nLow hope"),
        (0.75, 0.03, "High urgency\nLow hope\n(doom framing)"),
    ]:
        ax.text(x, y, txt, transform=ax.transAxes,
                fontsize=7, color="#888", va="top" if y > 0.5 else "bottom")
 
    # Label largest clusters per category
    for cat in ["bluesky_dominant","meta_dominant","shared"]:
        sub = emo[emo["category"] == cat].nlargest(4, "n_total")
        for _, r in sub.iterrows():
            ax.annotate(r["theme_label"],
                        (r["emotional_urgency"], r["emotional_hope"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=PALETTE[cat])
 
    ax.set_xlabel("Emotional urgency / alarm  (0 = calm, 1 = high alarm)", fontsize=10)
    ax.set_ylabel("Hope / solution orientation  (0 = none, 1 = strong)", fontsize=10)
    ax.set_title("Emotional register of climate themes\n"
                 "(bubble size ∝ cluster size; top-4 per category labelled)",
                 fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    _savefig("fig6_urgency_hope_scatter.png")
 
# ===========================================================================
# Fig 7: Economic vs Political framing scatter
# ===========================================================================
 
if "problem_economic" in themes_df.columns and "problem_political" in themes_df.columns:
    print("Fig 7 — economic vs political framing scatter …")
 
    ep = coherent_t[coherent_t["n_total"] >= MIN_TEXTS_FRAMING].copy()
    ep["theme_label"] = ep["theme_label"].apply(_clean)
 
    fig, ax = plt.subplots(figsize=(9, 7))
    for cat in ["bluesky_dominant","shared","meta_dominant"]:
        sub = ep[ep["category"] == cat]
        ax.scatter(
            sub["problem_economic"], sub["problem_political"],
            s=sub["n_total"] / ep["n_total"].max() * 400 + 20,
            color=PALETTE[cat], alpha=0.7, zorder=3,
            label=cat.replace("_"," ").title(),
        )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
 
    for (x, y, txt) in [
        (0.02, 0.97, "Low economic\nHigh political"),
        (0.62, 0.97, "High economic\nHigh political"),
        (0.02, 0.03, "Low economic\nLow political"),
        (0.62, 0.03, "High economic\nLow political"),
    ]:
        ax.text(x, y, txt, transform=ax.transAxes,
                fontsize=7, color="#888", va="top" if y > 0.5 else "bottom")
 
    for cat in ["bluesky_dominant","meta_dominant","shared"]:
        sub = ep[ep["category"] == cat].nlargest(4, "n_total")
        for _, r in sub.iterrows():
            ax.annotate(r["theme_label"],
                        (r["problem_economic"], r["problem_political"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=PALETTE[cat])
 
    ax.set_xlabel("Economic problem framing  (0 = absent, 1 = dominant)", fontsize=10)
    ax.set_ylabel("Political problem framing  (0 = absent, 1 = dominant)", fontsize=10)
    ax.set_title("Problem framing space: economic vs political\n"
                 "(tests whether paid ads prefer economic framing over political)",
                 fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    _savefig("fig7_economic_political_scatter.png")
 
# ===========================================================================
# Fig 8: Solution-present rate by category
# ===========================================================================
 
if "solution_present" in themes_df.columns:
    print("Fig 8 — solution-present rate …")
 
    sol = coherent_t.copy()
    sol["solution_present"] = sol["solution_present"].astype(str).str.lower() \
                                .map({"true": True, "false": False, "1": True, "0": False}) \
                                .fillna(False)
 
    sol_rate = (
        sol.groupby("category")["solution_present"]
        .agg(["sum","count"])
        .rename(columns={"sum":"n_solution","count":"n_total"})
    )
    sol_rate["rate"] = sol_rate["n_solution"] / sol_rate["n_total"] * 100
 
    row_order = [r for r in ["meta_dominant","shared","bluesky_dominant"]
                 if r in sol_rate.index]
    sol_rate  = sol_rate.reindex(row_order)
    sol_rate.index = [i.replace("_"," ").title() for i in sol_rate.index]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = [PALETTE.get(r.lower().replace(" ","_"), "#aaa") for r in sol_rate.index]
    bars    = ax.bar(sol_rate.index, sol_rate["rate"].values,
                     color=colors, alpha=0.85, width=0.5)
    for bar, (_, r) in zip(bars, sol_rate.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{r['rate']:.1f}%\n(n={int(r['n_total'])})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of coherent themes proposing a solution", fontsize=10)
    ax.set_ylim(0, min(100, sol_rate["rate"].max() + 15))
    ax.set_title("Solution-oriented framing by cluster category\n"
                 "(tests whether paid ads are more constructive than organic critique)",
                 fontsize=11)
    plt.tight_layout()
    _savefig("fig8_solution_rate.png")
 
# ===========================================================================
# Fig 9: Platform skew vs cluster size bubble chart
# ===========================================================================
 
print("Fig 9 — skew vs size bubble …")
 
labeled = themes_df[
    themes_df["theme_label"].notna() & (themes_df["theme_label"] != "")
].copy()
labeled["theme_label"] = labeled["theme_label"].apply(_clean)
 
fig, ax = plt.subplots(figsize=(11, 7))
for cat in ["bluesky_dominant","shared","meta_dominant"]:
    sub = labeled[labeled["category"] == cat]
    ax.scatter(
        sub["platform_skew"], sub["n_total"],
        s=sub["n_total"] / labeled["n_total"].max() * 500 + 15,
        color=PALETTE[cat], alpha=0.65, zorder=3,
        label=cat.replace("_"," ").title(),
    )
for cat in ["bluesky_dominant","shared","meta_dominant"]:
    for _, r in labeled[labeled["category"]==cat].nlargest(4,"n_total").iterrows():
        ax.annotate(r["theme_label"],
                    (r["platform_skew"], r["n_total"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=PALETTE[cat])
 
ax.axvline(1 - DOMINANT_THRESHOLD, color=C_BLUESKY, linestyle="--",
           linewidth=1, alpha=0.5)
ax.axvline(DOMINANT_THRESHOLD,     color=C_META,    linestyle="--",
           linewidth=1, alpha=0.5)
ax.set_xlabel("Platform skew  (0 = all Bluesky, 1 = all Meta)", fontsize=10)
ax.set_ylabel("Cluster size (number of texts)", fontsize=10)
ax.set_title("Platform skew vs cluster size\n"
             "(bubble area ∝ cluster size; top-4 per category labelled)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
_savefig("fig9_skew_vs_size_bubble.png")
 
# ===========================================================================
# Fig 10: Themes in 2D framing space (PCA on framing dimensions)
# ===========================================================================
 
if HAS_SKLEARN and len(framing_ok) >= 4:
    print("Fig 10 — framing PCA …")
 
    pca_data = coherent_t[framing_ok + ["category","theme_label","n_total"]].dropna()
    if len(pca_data) >= 10:
        pca_data["theme_label"] = pca_data["theme_label"].apply(_clean)
        X      = pca_data[framing_ok].values
        pca    = skPCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        pca_data = pca_data.copy()
        pca_data["pc1"] = coords[:, 0]
        pca_data["pc2"] = coords[:, 1]
 
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
 
        fig, ax = plt.subplots(figsize=(10, 7))
        for cat in ["bluesky_dominant","shared","meta_dominant"]:
            sub = pca_data[pca_data["category"] == cat]
            ax.scatter(sub["pc1"], sub["pc2"],
                       s=sub["n_total"] / pca_data["n_total"].max() * 300 + 15,
                       color=PALETTE[cat], alpha=0.7,
                       label=cat.replace("_"," ").title(), zorder=3)
        for cat in ["bluesky_dominant","shared","meta_dominant"]:
            sub = pca_data[pca_data["category"]==cat].nlargest(3,"n_total")
            for _, r in sub.iterrows():
                ax.annotate(r["theme_label"], (r["pc1"], r["pc2"]),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=6.5, color=PALETTE[cat])
 
        # Loading arrows
        loadings = pca.components_.T
        scale    = max(abs(coords).max(axis=0)) * 0.4
        for i, feat in enumerate(framing_ok):
            ax.annotate("", xy=(loadings[i,0]*scale, loadings[i,1]*scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
            ax.text(loadings[i,0]*scale*1.1, loadings[i,1]*scale*1.1,
                    FRAMING_LABELS[feat].replace("\n"," "),
                    fontsize=6.5, color="#555", ha="center")
 
        ax.set_xlabel(f"PC1  ({var1:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"PC2  ({var2:.1f}% variance)", fontsize=10)
        ax.set_title("Themes in framing space (PCA on 9 framing dimensions)\n"
                     "(arrows show framing dimension loadings; "
                     "separation = structural framing difference between categories)",
                     fontsize=11)
        ax.legend(fontsize=9, markerscale=1.5)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        _savefig("fig10_framing_pca.png")
 
# ===========================================================================
# Fig 11: Text length distribution by category
# ===========================================================================
 
print("Fig 11 — text length distribution …")
 
df_len = clustered.copy()
df_len["text_len"] = df_len["text"].fillna("").str.split().str.len()
df_len["category"] = df_len["theme_category"].fillna("noise")
 
cat_order  = ["meta_dominant","shared","bluesky_dominant"]
cat_labels = ["Meta-dominant\n(paid)", "Shared", "Bluesky-dominant\n(organic)"]
present    = [c for c in cat_order if c in df_len["category"].values]
 
fig, ax = plt.subplots(figsize=(9, 5))
data_box = [df_len[df_len["category"]==c]["text_len"].clip(upper=500).values
            for c in present]
bp = ax.boxplot(data_box, patch_artist=True, notch=True,
                widths=0.5, showfliers=False)
for patch, cat in zip(bp["boxes"], present):
    patch.set_facecolor(PALETTE[cat])
    patch.set_alpha(0.75)
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(1.5)
 
ax.set_xticks(range(1, len(present)+1))
ax.set_xticklabels([cat_labels[cat_order.index(c)] for c in present], fontsize=10)
ax.set_ylabel("Text length (words, clipped at 500)", fontsize=10)
ax.set_title("Text length distribution by cluster category\n"
             "(tests whether paid ads are more formulaic/shorter than organic posts)",
             fontsize=11)
 
# Annotate medians
for i, vals in enumerate(data_box):
    med = np.median(vals)
    ax.text(i+1, med + 3, f"med={int(med)}", ha="center",
            va="bottom", fontsize=8, color="black")
plt.tight_layout()
_savefig("fig11_text_length_by_category.png")
 
# ===========================================================================
# Fig 12: Platform coverage — how corpus divides across categories
# ===========================================================================
 
print("Fig 12 — platform coverage …")
 
coverage = (
    df[df["cluster_id"] >= 0]
    .groupby(["platform","theme_category"])
    .size()
    .unstack(fill_value=0)
)
# Add noise row
noise_counts = df[df["cluster_id"] == -1].groupby("platform").size()
for p in ["meta","bluesky"]:
    if "noise" not in coverage.columns:
        coverage["noise"] = 0
    coverage.loc[p, "noise"] = noise_counts.get(p, 0)
 
col_order = [c for c in ["meta_dominant","shared","bluesky_dominant","noise"]
             if c in coverage.columns]
coverage  = coverage[col_order]
coverage_pct = coverage.div(coverage.sum(axis=1), axis=0) * 100
 
fig, ax = plt.subplots(figsize=(9, 4))
x     = np.arange(len(coverage_pct))
left  = np.zeros(len(coverage_pct))
for cat in col_order:
    vals = coverage_pct[cat].values
    ax.bar(x, vals, bottom=left, color=PALETTE.get(cat,"#aaa"),
           alpha=0.85, label=cat.replace("_"," ").title(), width=0.5)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v > 5:
            ax.text(i, l + v/2, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    left += vals
 
ax.set_xticks(x)
ax.set_xticklabels(["Meta ads\n(paid)", "Bluesky posts\n(organic)"], fontsize=11)
ax.set_ylabel("% of texts", fontsize=10)
ax.set_ylim(0, 100)
ax.set_title("How each platform's corpus distributes across theme categories\n"
             "(reveals structural differences in thematic coverage)",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.22, 1.02))
plt.tight_layout()
_savefig("fig12_platform_coverage.png")
 
# ===========================================================================
# Stance figures (figs 13–16) — only if stance data loaded
# ===========================================================================
 
if not stance_available:
    print("\nNo stance data loaded — skipping figs 13–16.")
else:
    print("\nGenerating stance figures …")
 
    def _is_valid_label(s):
        if pd.isna(s):
            return False
        s = str(s).strip().lower()
        return s not in ("", "nan", "unassigned")
 
    def _stance_pct_table(df_sub, min_texts=MIN_TEXTS_STANCE, top_n=20):
        # Exclude incoherent/unlabeled cluster texts
        df_sub = df_sub[df_sub["theme_label"].apply(_is_valid_label)]
        if df_sub.empty:
            return pd.DataFrame()
        tbl = (
            df_sub.groupby(["theme_label", STANCE_COL])
            .size().unstack(fill_value=0)
        )
        for s in STANCE_ORDER:
            if s not in tbl.columns:
                tbl[s] = 0
        tbl = tbl[[s for s in STANCE_ORDER if s in tbl.columns]]
        tbl["total"] = tbl.sum(axis=1)
        tbl = tbl[tbl["total"] >= min_texts].sort_values("total", ascending=False).head(top_n)
        if tbl.empty:
            return pd.DataFrame()
        pct = tbl[STANCE_ORDER].div(tbl["total"], axis=0) * 100
        pct["total"] = tbl["total"]
        pct.index = pct.index.map(_clean)
        return pct
 
    # Diagnostic: show theme_label value distribution in stance sample
    label_counts = stance_clustered["theme_label"].astype(str).value_counts()
    n_valid   = stance_clustered["theme_label"].apply(_is_valid_label).sum()
    n_invalid = len(stance_clustered) - n_valid
    print(f"  stance_clustered: {len(stance_clustered):,} texts "
          f"({n_valid:,} labeled, {n_invalid:,} unlabeled/incoherent excluded)")
 
    # Fig 13: Stance by category
    print("  Fig 13 — stance by category …")
    cat_stance = (
        stance_clustered.groupby(["theme_category", STANCE_COL])
        .size().unstack(fill_value=0)
    )
    for s in STANCE_ORDER:
        if s not in cat_stance.columns:
            cat_stance[s] = 0
    cat_stance = cat_stance[[s for s in STANCE_ORDER if s in cat_stance.columns]]
    cat_stance_pct = cat_stance.div(cat_stance.sum(axis=1), axis=0) * 100
    cat_stance_pct["total"] = cat_stance.sum(axis=1)
    row_order = [r for r in ["bluesky_dominant","shared","meta_dominant"]
                 if r in cat_stance_pct.index]
    cat_stance_pct = cat_stance_pct.reindex(row_order)
    cat_stance_pct.index = [i.replace("_"," ").title() for i in cat_stance_pct.index]
    _stacked_hbar(cat_stance_pct,
                  "Stance composition by cluster category",
                  "fig13_stance_by_category.png",
                  "(stance-labelled sample only)")
 
    # Figs 14–15: Stance per theme, per platform category
    for cat, fname, title in [
        ("bluesky_dominant", "fig14_stance_per_theme_bluesky.png",
         "Stance per theme — Bluesky-dominant clusters"),
        ("meta_dominant",    "fig15_stance_per_theme_meta.png",
         "Stance per theme — Meta-dominant clusters"),
    ]:
        print(f"  {fname} …")
        sub = stance_clustered[stance_clustered["theme_category"] == cat]
        tbl = _stance_pct_table(sub)
        if not tbl.empty:
            _stacked_hbar(tbl, title, fname,
                          f"(top 20 themes; min {MIN_TEXTS_STANCE} stance texts)")
 
    # Fig 16: Most extreme-stance themes (3-panel)
    print("  Fig 16 — extreme stance themes …")
    full_tbl = _stance_pct_table(stance_clustered, min_texts=MIN_TEXTS_STANCE, top_n=200)
    if not full_tbl.empty:
        n_panels = len([s for s in STANCE_ORDER if s in full_tbl.columns])
        fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 6))
        if n_panels == 1:
            axes = [axes]
        for ax, s in zip(axes, [s for s in STANCE_ORDER if s in full_tbl.columns]):
            top = full_tbl.sort_values(s, ascending=False).head(12)
            color = STANCE_COLORS[s]
            y = np.arange(len(top))
            ax.barh(y, top[s].values, color=color, alpha=0.85, height=0.65)
            ax.set_yticks(y)
            ax.set_yticklabels(top.index.tolist(), fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter())
            ax.set_title(f"Most {s}\nthemes", fontsize=10)
            ax.set_xlabel(f"% {s}", fontsize=9)
            for i, (v, n) in enumerate(zip(top[s].values, top["total"].values)):
                ax.text(v+0.5, i, f"{v:.0f}% (n={int(n)})",
                        va="center", fontsize=7, color="#333")
        fig.suptitle("Themes most dominated by each stance\n"
                     "(all categories; stance-labelled sample)",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        _savefig("fig16_extreme_stance_themes.png")
 

print(f"\n{'='*60}")
print(f"All figures saved to: {ANALYSIS_DIR}/")
print("\nFigures and their analytical purpose:")
print("  fig1  — Meta-dominant themes (what paid ads uniquely cover)")
print("  fig2  — Bluesky-dominant themes (what organic posts uniquely cover)")
print("  fig3  — Shared theme balance (paid vs organic ratio per theme)")
print("  fig4  — Framing heatmap (structural framing differences)")
print("  fig5  — Framing radar (fingerprint of paid vs organic discourse)")
print("  fig6  — Urgency/hope scatter (emotional register comparison)")
print("  fig7  — Economic/political framing scatter (problem frame types)")
print("  fig8  — Solution-present rate (constructiveness by category)")
print("  fig9  — Skew vs size bubble (platform concentration patterns)")
print("  fig10 — Framing PCA (structural separation in framing space)")
print("  fig11 — Text length distributions (surface linguistic features)")
print("  fig12 — Platform coverage (how each corpus divides thematically)")
if stance_available:
    print("  fig13 — Stance by category")
    print("  fig14 — Stance per Bluesky theme")
    print("  fig15 — Stance per Meta theme")
    print("  fig16 — Most extreme-stance themes")
 