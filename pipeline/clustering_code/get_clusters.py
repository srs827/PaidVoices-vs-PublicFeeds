import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap
import hdbscan

# replace with proper paths for meta/bluesky from preprocessing subfolders
DEDUP_CSV = "preprocessing/combined_deduped_sbert080.csv"
EMB_NPY   = "preprocessing/combined_deduped_sbert080.embeddings.npy"

USE_PCA = True

# number of top keywords to save per cluster
TOP_K = 5

df = pd.read_csv(DEDUP_CSV)
X = np.load(EMB_NPY)
if len(df) != X.shape[0]:
    raise RuntimeError("Row mismatch between CSV and embeddings array.")
# L2 normalization
X = normalize(X, norm="l2", axis=1, copy=False)

# PCA to reduce to ~100 dims
if USE_PCA:
    pca = PCA(n_components=100, random_state=42)
    X_reduced = pca.fit_transform(X)
else:
    X_reduced = X

# transforms high dim vectors to lower dim w/ UMAP
u = umap.UMAP(
    n_neighbors=50, 
    min_dist=0.05, 
    n_components=20,
    metric="cosine",
    random_state=42,
    verbose=True
)
X_umap = u.fit_transform(X_reduced)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
    prediction_data=True
)
labels = clusterer.fit_predict(X_umap)

# add cluster labels + probabilities to df
df["cluster"] = labels
df["probability"] = clusterer.probabilities_
df.to_csv("ads_with_clusters.csv", index=False) # or posts_with_clusters for bluesky

counts = df["cluster"].value_counts(dropna=False).sort_index()
n_outliers = int(counts.get(-1, 0))
n_clusters = (counts.index != -1).sum()
print("\n=== Clustering summary ===")
print(f"Total rows: {len(df)}")
print(f"Outliers (-1): {n_outliers}")
print(f"Num clusters (excl -1): {n_clusters}")
print("Sizes by label:")
print(counts.to_string())

# top-k ranking key
def rank_key(subdf: pd.DataFrame):
    return subdf["probability"]

mask = df["cluster"].ne(-1)
work = df.loc[mask].copy()

# compute ranking score
work["_rank"] = rank_key(work)


# take top-k per cluster into a single DataFrame
topk_all = (
    work.sort_values(["cluster", "_rank"], ascending=[True, False])
        .groupby("cluster", group_keys=False)
        .head(TOP_K)
        .assign(topk_rank=lambda x: x.groupby("cluster")["_rank"].rank(method="first", ascending=False).astype(int))
        .drop(columns=["_rank"])
        .sort_values(["cluster", "topk_rank"])
        .reset_index(drop=True)
)

# replace cid with ad_archive_id for meta if needed
cols_pref = ["cluster", "topk_rank", "cid", "probability", "text"]
cols_exist = [c for c in cols_pref if c in topk_all.columns]
topk_out = topk_all[cols_exist] if cols_exist else topk_all

topk_out.to_csv("topk_posts_by_cluster.csv", index=False)
print("Wrote topk_posts_by_cluster.csv (one row per post/ad; includes topk_rank within each cluster).")

summary = (
    df.loc[mask].groupby("cluster").size()
      .reset_index(name="size")
      .sort_values("size", ascending=False)
)
summary.to_csv("cluster_summary.csv", index=False)
print("Wrote cluster_summary.csv.")
