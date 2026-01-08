import os, json, math
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

# replace with proper paths for meta/bluesky
SUMMARIES_JSONL = "cluster_summarization/cluster_summaries.jsonl"
OUT_PREFIX      = "merged_"

EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"

GRID_SPEC          = "0.6:0.9:.02"  # similarity threshold sweep for summary merge
DEVICE             = None
LIMIT_CLUSTERS     = None  


def read_summaries(path):
    """
    Read summaries
    Returns a DataFrame with columns: cluster, summary 
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("summary") and "error" not in obj:
                rows.append({
                    "cluster": int(obj["cluster"]),
                    "summary": str(obj["summary"]).strip()
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["cluster"])
    if df.empty:
        raise ValueError("No valid summaries found.")

    df = df.sort_values("cluster").reset_index(drop=True)
    if LIMIT_CLUSTERS is not None:
        df = df.head(LIMIT_CLUSTERS).reset_index(drop=True)

    return df


def build_grid(spec: str):
    """
    Format: "start:stop:step" or list like "0.7,0.75,0.8".
    Returns list[float] of thresholds.
    """
    if "," in spec:
        vals = sorted(set(float(x) for x in spec.split(",")))
        return [v for v in vals if 0.0 <= v <= 1.0]
    a, b, c = [float(x) for x in spec.split(":")]
    n = max(1, int(round((b - a) / c)) + 1)
    arr = [a + i*c for i in range(n)]
    if arr[-1] < b - 1e-9:
        arr.append(b)
    return [min(1.0, max(0.0, v)) for v in arr]

def write_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(
                {"cluster": int(r["cluster"]), "summary": str(r["summary"])},
                ensure_ascii=False
            ) + "\n")
    print(f"Wrote: {path}")

# merge similar summaries based on the given threshold
def threshold_to_labels(sim, thr):
    """
    - Start a new group at i
    - Add any j with sim[i,j] >= thr
    - Assign them all the same merged label
    """
    n = sim.shape[0]
    labels = -np.ones(n, dtype=int)
    gid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        members = [i]
        for j in range(i + 1, n):
            if labels[j] == -1 and sim[i, j] >= thr:
                members.append(j)
        for idx in members:
            labels[idx] = gid
        gid += 1
    return labels, gid


def eval_summary_sil_dbi(E_summ_unit: np.ndarray, labels_summ: np.ndarray):
    """
    Compute silhouette and DBI on summary embeddings, given merged labels
    at a particular threshold.

    - silhouette (cosine) on non-singleton clusters (need >= 2 clusters with size>1)
    - DBI (euclidean) on all summaries if we have >=2 clusters total
    """
    u, count = np.unique(labels_summ, return_counts=True)
    n_clusters = len(u)

    sil = np.nan
    try:
        non_single_mask = np.array([
            count[np.where(u == lb)[0][0]] > 1
            for lb in labels_summ
        ])
        if (
            non_single_mask.sum() >= 2 and
            len(np.unique(labels_summ[non_single_mask])) >= 2
        ):
            # can only calcuate for non-single clusters
            sil = silhouette_score(
                E_summ_unit[non_single_mask],
                labels_summ[non_single_mask],
                metric="cosine",
            )
    except Exception:
        pass

    dbi = np.nan
    try:
        if n_clusters >= 2:
            dbi = davies_bouldin_score(E_summ_unit, labels_summ)
    except Exception:
        pass

    return sil, dbi


def grid_search(df_summ,
                              E_summ_unit,
                              grid,
                              threshold_to_labels_fn,
                              eval_fn):
    """
    For each threshold:
        Merge summaries based on threshold
        Score the merge with silhouette and DBI.
        Keep per-threshold stats.
    Then, we pick the best threshold based on a combined score.
    """

    # cosine sim between summaries
    sim_summ = E_summ_unit @ E_summ_unit.T

    rows = []
    label_cache = {}

    for thr in grid:
        # merge
        labels_summ, _ = threshold_to_labels_fn(sim_summ, float(thr))

        # score 
        sil, dbi = eval_fn(E_summ_unit, labels_summ)

        uniq, counts = np.unique(labels_summ, return_counts=True)
        n_clusters = len(uniq)
        n_singletons = int(np.sum(counts == 1))

        rows.append({
            "threshold": float(thr),
            "silhouette": sil,
            "dbi": dbi,
            "n_clusters_summ": int(n_clusters),
            "n_singletons_summ": int(n_singletons),
        })
        label_cache[float(thr)] = labels_summ

    ranked = pd.DataFrame(rows)

    # helper functions for ranking/normalization with dbi and silhouette [0, 1]
    def minmax_up(s):
        v = s.dropna()
        if v.empty:
            return s*0 + np.nan
        lo, hi = v.min(), v.max()
        mm = (v - lo)/(hi - lo) if hi > lo else v/v
        out = s*0 + np.nan
        out.loc[mm.index] = mm
        return out

    def minmax_down(s):
        v = s.dropna()
        if v.empty:
            return s*0 + np.nan
        lo, hi = v.min(), v.max()
        mm = (v - lo)/(hi - lo) if hi > lo else v/v
        out = s*0 + np.nan
        out.loc[mm.index] = 1 - mm
        return out

    ranked["sil_norm"] = minmax_up(ranked["silhouette"])
    ranked["dbi_norm"] = minmax_down(ranked["dbi"])

    ranked["score"] = (
        ranked["sil_norm"].fillna(0.0)
        + ranked["dbi_norm"].fillna(0.0)
    )

    ranked = ranked.sort_values(
        ["score", "silhouette"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best = ranked.iloc[0].to_dict()
    return best, ranked, label_cache

def main():
    # 1. Read coherent cluster summaries
    df = read_summaries(SUMMARIES_JSONL)  # columns: cluster, summary

    # 2. Load embedding model
    print(f"Loading embedder: {EMBED_MODEL} on device={DEVICE}")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    # 3. Embed summaries, L2-normalize
    summ_texts = df["summary"].astype(str).tolist()
    E_summ = model.encode(
        summ_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    E_summ = normalize(E_summ)  # cosine sim 

    # 4. Build threshold grid
    grid = build_grid(GRID_SPEC)

    # 5. Grid search 
    best, ranked, cache = grid_search(
        df_summ=df[["cluster", "summary"]],
        E_summ_unit=E_summ,
        grid=grid,
        threshold_to_labels_fn=threshold_to_labels,
        eval_fn=eval_summary_sil_dbi,
    )

    # 6. Write sweep of summary metrics 
    sweep_path = f"{OUT_PREFIX}_summary_eval_threshold_sweep.csv"
    ranked.to_csv(sweep_path, index=False, float_format="%.6f")
    print(f"Wrote: {sweep_path}")
    print(
        f"Best threshold: {best['threshold']} | "
        f"sil={best['silhouette']} | dbi={best['dbi']} | "
        f"n_clusters={best['n_clusters_summ']}"
    )

    # 7. Dump mapping original cluster -> merged_id under best threshold
    best_thr = float(best["threshold"])
    labels_summ = cache[best_thr]

    map_df = pd.DataFrame({
        "cluster": df["cluster"].astype(int).to_numpy(),
        "merged_id": labels_summ
    })
    map_path = f"{OUT_PREFIX}_best_cluster_to_merged.csv"
    map_df.to_csv(map_path, index=False)
    print(f"Wrote: {map_path}")

    # Choose a representative original cluster per merged group.
    # Policy: keep the smallest original cluster id within each merged_id.
    rep_df = (
        map_df.loc[map_df.groupby("merged_id")["cluster"].idxmin()]
        .sort_values(["merged_id", "cluster"])
        .reset_index(drop=True)
    )
    keep_clusters = set(rep_df["cluster"].tolist())

    # Build pruned summaries JSONL: keep only representatives, using their original ids and summaries.
    kept_df = (
        df[df["cluster"].isin(keep_clusters)]
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    out_jsonl = f"{OUT_PREFIX}_merged_summaries.jsonl"
    write_jsonl(kept_df[["cluster", "summary"]], out_jsonl)

    # also log what we removed for traceability.
    removed_df = df[~df["cluster"].isin(keep_clusters)].sort_values("cluster")
    removed_csv = f"{OUT_PREFIX}_removed_clusters.csv"
    removed_df[["cluster"]].to_csv(removed_csv, index=False)
    print(
        f"Kept {len(kept_df)} clusters, removed {len(removed_df)}. "
        f"Details: {removed_csv}"
    )

if __name__ == "__main__":
    main()
