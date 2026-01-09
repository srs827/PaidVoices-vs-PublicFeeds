"""
Analysis of common themes between Bluesky posts and Meta ads.

Functions:
1) UMAP representation of all texts with theme labels and platform markers.
2) Table with one Meta ad and one Bluesky post per theme (representative examples).
3) Extra figures: bar chart of counts per theme per platform.
"""

import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
import umap

import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import numpy as np

# joint themes from map_common_themes.py
JOINT_THEMES = {
    "Sustainable Fashion",
    "Anti-Deforestation",
    "Water pollution",
    "Ocean conservation",
    "Anti-Big Oil",
    "Clean energy",
    "Anti-Drilling",
    "Climate Wildfires",
}
PLATFORM_COLORS = {
    "Bluesky": "#1f77b4",  # blue
    "Meta":    "#d62728",  # red
}

# Input CSVs (already filtered to joint themes)
# replace with proper paths for meta/bluesky containining only the common themes
# these csvs are produced by map_common_themes.py
POSTS_CSV = "bluesky_combined_themes_common.csv"
ADS_CSV   = "meta_combined_themes_common.csv"

# Column names in those CSVs
# replace with proper values if needed
POSTS_TEXT_COL  = "text"                  # Bluesky post text
POSTS_THEME_COL = "llm_theme_theme"   # Combined theme label for posts

ADS_TEXT_COL    = "ad_text"               # Meta ad text
ADS_THEME_COL   = "final_theme_llm_theme" # Combined theme label for ads

# Output directory for figures & tables
OUTPUT_DIR = "theme_comparison_outputs"

# Sentence embedding model 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RANDOM_SEED = 42

def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_data():
    """Load posts and ads CSVs, enforce required columns, and align into common schema."""
    posts_df = pd.read_csv(POSTS_CSV, encoding="latin1")
    ads_df   = pd.read_csv(ADS_CSV,   encoding="latin1")

    for col in [POSTS_TEXT_COL, POSTS_THEME_COL]:
        if col not in posts_df.columns:
            raise ValueError(f"Column '{col}' not found in posts dataframe.")

    for col in [ADS_TEXT_COL, ADS_THEME_COL]:
        if col not in ads_df.columns:
            raise ValueError(f"Column '{col}' not found in ads dataframe.")

    # Standardized columns: text, theme, platform
    posts = posts_df[[POSTS_TEXT_COL, POSTS_THEME_COL]].copy()
    posts.columns = ["text", "theme"]
    posts["platform"] = "Bluesky"

    ads = ads_df[[ADS_TEXT_COL, ADS_THEME_COL]].copy()
    ads.columns = ["text", "theme"]
    ads["platform"] = "Meta"

    # Drop rows with missing text or theme
    posts = posts.dropna(subset=["text", "theme"])
    ads   = ads.dropna(subset=["text", "theme"])

    combined = pd.concat([posts, ads], ignore_index=True)

    return posts, ads, combined

def compute_embeddings(df, model_name=EMBEDDING_MODEL_NAME):
    """Compute sentence embeddings for df['text']."""
    model = SentenceTransformer(model_name)
    texts = df["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings

def print_theme_counts_and_rank_by_difference(
    combined_df,
    output_dir=OUTPUT_DIR,
    save_csv=True,
):
    """
    Prints:
      1) Counts per combined theme per platform (Bluesky/Meta)
      2) Themes ranked by highest platform difference

    Restricts to JOINT_THEMES.
    """

    df = combined_df[combined_df["theme"].isin(JOINT_THEMES)].copy()

    counts = (
        df.groupby(["theme", "platform"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to wide: one row per theme, columns = platforms
    pivot = (
        counts.pivot(index="theme", columns="platform", values="count")
        .fillna(0)
        .astype(int)
    )

    # Ensure both columns exist (in case one platform is missing for a theme)
    if "Bluesky" not in pivot.columns:
        pivot["Bluesky"] = 0
    if "Meta" not in pivot.columns:
        pivot["Meta"] = 0

    # Add signed and absolute differences
    pivot["diff_signed"] = pivot["Bluesky"] - pivot["Meta"]
    pivot["diff_abs"] = (pivot["Bluesky"] - pivot["Meta"]).abs()

    # Sort by biggest gap
    ranked = pivot.sort_values(["diff_abs", "diff_signed"], ascending=[False, False]).copy()

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "theme_counts_ranked_by_platform_difference.csv")
        ranked.reset_index().to_csv(out_path, index=False)
        print(f"[Counts+Diff CSV] Saved to {out_path}")

    # Print counts table
    print("\nCounts per combined theme (Bluesky vs Meta):")
    print(ranked[["Bluesky", "Meta", "diff_signed", "diff_abs"]].to_string())

    print("\nThemes ranked by highest platform difference (abs gap):")
    for theme, row in ranked.iterrows():
        b = int(row["Bluesky"])
        m = int(row["Meta"])
        ds = int(row["diff_signed"])
        da = int(row["diff_abs"])
        direction = "more Bluesky" if ds > 0 else ("more Meta" if ds < 0 else "equal")
        print(f"  {theme}: Bluesky={b}, Meta={m}, diff={ds} ({direction}), abs_diff={da}")

    return ranked

def run_umap(embeddings, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42):
    """Apply UMAP to embeddings and return 2D coordinates."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    return coords


def _adjust_label_positions(label_entries, min_delta=0.25):
    """
    Simple greedy vertical de-overlap: sort by y, and if two labels are
    closer than min_delta along y, push the lower one down.
    """
    # indices sorted by y-coordinate
    sorted_idx = sorted(range(len(label_entries)), key=lambda i: label_entries[i]["y"])
    last_y = None
    for idx in sorted_idx:
        entry = label_entries[idx]
        y = entry["y"]
        if last_y is None:
            last_y = y
            continue
        if y - last_y < min_delta:
            y = last_y + min_delta
            entry["y"] = y
        last_y = entry["y"]
    return label_entries


def plot_umap(combined_df, coords, output_dir="theme_comparison_outputs"):
    """
    UMAP representation:
    - Points: colored by platform (blue = Bluesky, red = Meta).
    """
    df = combined_df.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    df["is_joint_theme"] = df["theme"].isin(JOINT_THEMES)

    sns.set(style="white", context="talk")

    plt.figure(figsize=(8, 6))

    # scatter points: colored by platform
    ax = sns.scatterplot(
        data=df,
        x="umap_x",
        y="umap_y",
        hue="platform",
        palette=PLATFORM_COLORS,
        s=20,
        alpha=0.18,      
        edgecolor=None,
        linewidth=0,
    )

    theme_colors = sns.color_palette("tab10", n_colors=len(JOINT_THEMES))
    theme_color_map = {theme: theme_colors[i] for i, theme in enumerate(JOINT_THEMES)}

    # compute a global median radius for normalization of circle sizes
    base_radii = []
    for theme in JOINT_THEMES:
        group = df[df["theme"] == theme]
        if group.empty:
            continue
        x = group["umap_x"].values
        y = group["umap_y"].values
        cx, cy = x.mean(), y.mean()
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if len(dists):
            base_radii.append(np.percentile(dists, 50))
    global_med = np.median(base_radii) if base_radii else 0.5

    # collect label entries so we can de-overlap after all circles are known
    label_entries = []

    # one circle per theme
    for theme in JOINT_THEMES:
        group = df[df["theme"] == theme]
        if group.empty:
            continue

        x = group["umap_x"].values
        y = group["umap_y"].values
        cx, cy = x.mean(), y.mean()

        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if len(dists) == 0:
            continue

        # base radius from 50th percentile (median)
        base_r = np.percentile(dists, 50)

        # shrink relative to global median to keep circles modest
        radius = base_r
        if global_med > 0:
            radius = base_r * 0.7 * (base_r / global_med)

        # final clamp to keep circles small and consistent
        radius = float(np.clip(radius, 0.15, 0.6))

        circle_color = theme_color_map[theme]

        circle = patches.Circle(
            (cx, cy),
            radius,
            fill=False,
            linestyle="solid",
            linewidth=2.0,
            edgecolor=circle_color,
            alpha=1.0,  # less transparent
        )
        ax.add_patch(circle)
        # prepare label entry (to be adjusted later)
        lx = cx
        ly = cy + radius + 0.05

        label_entries.append(
            dict(
                x=lx,
                y=ly,
                theme=theme,
                color=circle_color,
            )
        )

    # adjust label positions to avoid overlaps 
    label_entries = _adjust_label_positions(label_entries, min_delta=0.25)

    # draw labels with adjusted positions 
    for entry in label_entries:
        ax.text(
            entry["x"],
            entry["y"],
            entry["theme"],
            fontsize=9,
            weight="bold",
            ha="center",
            va="bottom",
            color=entry["color"],
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec=entry["color"],
                lw=0.6,
                alpha=0.9,
            ),
        )

    plt.title(
        "UMAP Representation of Bluesky Posts and Meta Ads, Labeled with Themes",
        fontsize=14,
    )
    plt.xlabel("")
    plt.ylabel("")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([], [], marker='o', color='white',
               markerfacecolor=PLATFORM_COLORS["Bluesky"],
               markersize=14,
               markeredgewidth=1.5,
               label="Bluesky"),
        Line2D([], [], marker='o', color='white',
               markerfacecolor=PLATFORM_COLORS["Meta"],
               markersize=14,
               markeredgewidth=1.5,
               label="Meta"),
    ]

    ax.legend(
        handles=legend_handles,
        title="Platform",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.88)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "umap_platform_combined_theme_circles.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[UMAP] Saved to {out_path}")

    return df


# 2) REPRESENTATIVE TABLE

def print_top_theme_counts_per_platform(
    combined_df,
    top_k=10,
    output_dir=OUTPUT_DIR,
    save_csv=True,
):
    """
    Prints the number of items per theme for each platform (Bluesky/Meta),
    restricted to JOINT_THEMES. Also prints top-K themes for each platform
    and optionally saves the counts to CSV.
    """
    df = combined_df[combined_df["theme"].isin(JOINT_THEMES)].copy()

    counts = (
        df.groupby(["platform", "theme"])
        .size()
        .reset_index(name="count")
    )

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "theme_counts_per_platform.csv")
        counts.to_csv(out_path, index=False)
        print(f"[Counts CSV] Saved to {out_path}")

    # Print top-K per platform
    for platform in ["Bluesky", "Meta"]:
        sub = counts[counts["platform"] == platform].sort_values("count", ascending=False)
        print(f"\nTop {top_k} themes for {platform} (within JOINT_THEMES):")
        if sub.empty:
            print("  (no rows)")
            continue
        for _, row in sub.head(top_k).iterrows():
            print(f"  {row['theme']}: {int(row['count'])}")

    pivot = (
        counts.pivot(index="theme", columns="platform", values="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    print("\nCounts table (themes x platform):")
    print(pivot.to_string())

def build_representative_table(posts, ads, output_dir=OUTPUT_DIR):
    """
    For each theme, pick one Meta ad and one Bluesky post as representatives.
    - Currently random sampling 
    - Not used
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    themes = sorted(set(posts["theme"]).intersection(set(ads["theme"])))
    rows = []

    for theme in themes:
        subset_posts = posts[posts["theme"] == theme]
        subset_ads   = ads[ads["theme"] == theme]

        if subset_posts.empty or subset_ads.empty:
            continue 

        post_example = subset_posts.sample(1, random_state=RANDOM_SEED)["text"].iloc[0]
        ad_example   = subset_ads.sample(1, random_state=RANDOM_SEED)["text"].iloc[0]

        rows.append({
            "theme": theme,
            "bluesky_example": post_example,
            "meta_example": ad_example,
        })

    rep_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "representative_examples_per_theme.csv")
    rep_df.to_csv(out_path, index=False)
    print(f"[Table] Saved representative examples table to {out_path}")

    return rep_df

# 3) EXTRA FIGURES
def plot_theme_counts_per_platform(combined_df, output_dir=OUTPUT_DIR):
    """
    Bar chart: number of posts/ads per theme per platform.
    Only shows the combined JOINT_THEMES.
    """
    df = combined_df[combined_df["theme"].isin(JOINT_THEMES)].copy()

    counts = (
        df.groupby(["theme", "platform"])
        .size()
        .reset_index(name="count")
    )

    theme_order = sorted(JOINT_THEMES)

    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=counts,
        x="theme",
        y="count",
        hue="platform",
        order=theme_order,
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Theme")
    plt.ylabel("Number of items")
    plt.title("Number of Bluesky Posts and Meta Ads per Combined Theme")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "theme_counts_per_platform_combined_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Barplot] Saved theme counts per platform (combined themes only) to {out_path}")

def main():
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    posts, ads, combined = load_data()
    print(f"Loaded {len(posts)} Bluesky posts and {len(ads)} Meta ads.")

    print("Computing sentence embeddings...")
    embeddings = compute_embeddings(combined)

    print("Running UMAP...")
    coords = run_umap(embeddings)

    print("Plotting UMAP...")
    combined_with_coords = plot_umap(combined, coords)

    print("Building representative examples table...")
    rep_df = build_representative_table(posts, ads)
    print("Printing counts per theme + ranking by platform difference...")
    print_theme_counts_and_rank_by_difference(combined_with_coords)
    print("Plotting extra figures (theme counts per platform)...")
    plot_theme_counts_per_platform(combined_with_coords)

    print("Done.")


if __name__ == "__main__":
    main()
