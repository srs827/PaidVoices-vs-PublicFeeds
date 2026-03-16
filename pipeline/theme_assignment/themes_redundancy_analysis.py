"""
Theme Set Redundancy Analysis
Computes pairwise cosine similarities between theme labels using SentenceBERT,
reports mean/distribution, and flags potentially redundant theme pairs.

Usage:
    python theme_similarity.py --meta meta_themes.csv --bluesky bluesky_themes.csv
    python theme_similarity.py --meta meta_themes.csv  # single platform
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_theme(raw: str) -> str:
    """Strip markdown bold/italic markers and surrounding quotes from a theme label."""
    s = str(raw).strip()
    s = re.sub(r'\*+', '', s)   # remove ** or *
    s = s.strip('"\'')
    return s.strip()


def load_themes(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if 'theme' not in df.columns:
        raise ValueError(f"'theme' column not found in {csv_path}")
    themes = [clean_theme(t) for t in df['theme'].dropna().tolist()]
    return themes


def compute_similarity_matrix(themes: list[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(themes, convert_to_numpy=True, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix


def pairwise_upper(sim_matrix: np.ndarray) -> np.ndarray:
    """Return a flat array of upper-triangle (off-diagonal) similarities."""
    n = sim_matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return sim_matrix[idx]


def summarise(name: str, themes: list[str], sim_matrix: np.ndarray, threshold: float = 0.80):
    pw = pairwise_upper(sim_matrix)
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(themes)} themes)")
    print(f"{'='*60}")
    print(f"  Mean cosine similarity : {pw.mean():.4f}")
    print(f"  Median                 : {np.median(pw):.4f}")
    print(f"  Std dev                : {pw.std():.4f}")
    print(f"  Min                    : {pw.min():.4f}")
    print(f"  Max                    : {pw.max():.4f}")
    print(f"  Pairs > {threshold:.2f}           : {(pw > threshold).sum()} / {len(pw)}")

    # flag high-similarity pairs
    n = len(themes)
    flagged = []
    for i, j in combinations(range(n), 2):
        if sim_matrix[i, j] > threshold:
            flagged.append((sim_matrix[i, j], themes[i], themes[j]))
    flagged.sort(reverse=True)

    if flagged:
        print(f"\n  High-similarity pairs (cosine > {threshold}):")
        for score, t1, t2 in flagged:
            print(f"    {score:.4f}  |  '{t1}'  ↔  '{t2}'")
    else:
        print(f"\n  No pairs exceed the {threshold} threshold.")

    return pw, flagged


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, threshold: float = 0.80, out_path: str = "theme_similarity.png"):
    """
    results: { platform_name: (themes, sim_matrix, pw_array) }
    """
    n_platforms = len(results)
    fig = plt.figure(figsize=(7 * n_platforms, 10))
    gs = gridspec.GridSpec(2, n_platforms, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['steelblue', 'darkorange']

    for col, (name, (themes, sim_matrix, pw)) in enumerate(results.items()):
        color = colors[col % len(colors)]
        n = len(themes)

        # ── heatmap ──────────────────────────────────────────────────────────
        ax_heat = fig.add_subplot(gs[0, col])
        im = ax_heat.imshow(sim_matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax_heat.set_title(f"{name}\nPairwise Cosine Similarity\n({n} themes)", fontsize=11)
        ax_heat.set_xlabel("Theme index")
        ax_heat.set_ylabel("Theme index")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # ── histogram ────────────────────────────────────────────────────────
        ax_hist = fig.add_subplot(gs[1, col])
        ax_hist.hist(pw, bins=30, color=color, edgecolor='white', linewidth=0.5)
        ax_hist.axvline(pw.mean(), color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean = {pw.mean():.3f}')
        ax_hist.axvline(threshold, color='black', linestyle=':', linewidth=1.5,
                        label=f'Threshold = {threshold}')
        ax_hist.set_title(f"{name}\nDistribution of Pairwise Similarities", fontsize=11)
        ax_hist.set_xlabel("Cosine similarity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend(fontsize=8)

    fig.suptitle("Theme Set Redundancy Analysis", fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {out_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Theme redundancy analysis via SentenceBERT.")
    parser.add_argument('--meta',     type=str, default=None, help="Path to Meta themes CSV")
    parser.add_argument('--bluesky',  type=str, default=None, help="Path to Bluesky themes CSV")
    parser.add_argument('--threshold', type=float, default=0.80,
                        help="Cosine similarity threshold for flagging redundant pairs (default: 0.80)")
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help="SentenceBERT model to use")
    parser.add_argument('--out_plot', type=str, default='theme_similarity.png',
                        help="Output path for the similarity plot")
    parser.add_argument('--out_csv', type=str, default='flagged_pairs.csv',
                        help="Output path for flagged-pairs CSV")
    args = parser.parse_args()

    if not args.meta and not args.bluesky:
        parser.error("Provide at least one of --meta or --bluesky.")

    print(f"Loading model: {args.model} ...")
    model = SentenceTransformer(args.model)

    platform_files = {}
    if args.meta:
        platform_files['Meta'] = args.meta
    if args.bluesky:
        platform_files['Bluesky'] = args.bluesky

    results = {}
    all_flagged = []

    for name, path in platform_files.items():
        print(f"\nLoading {name} themes from: {path}")
        themes = load_themes(path)
        print(f"  Found {len(themes)} themes.")
        sim_matrix = compute_similarity_matrix(themes, model)
        pw, flagged = summarise(name, themes, sim_matrix, threshold=args.threshold)
        results[name] = (themes, sim_matrix, pw)
        for score, t1, t2 in flagged:
            all_flagged.append({'platform': name, 'cosine_similarity': round(score, 4),
                                 'theme_1': t1, 'theme_2': t2})

    # save flagged pairs
    if all_flagged:
        df_flagged = pd.DataFrame(all_flagged).sort_values('cosine_similarity', ascending=False)
        df_flagged.to_csv(args.out_csv, index=False)
        print(f"\n  Flagged pairs saved → {args.out_csv}")
    else:
        print(f"\n  No pairs exceeded threshold {args.threshold}; no CSV written.")

    plot_results(results, threshold=args.threshold, out_path=args.out_plot)
    print("\nDone.")


if __name__ == '__main__':
    main()
