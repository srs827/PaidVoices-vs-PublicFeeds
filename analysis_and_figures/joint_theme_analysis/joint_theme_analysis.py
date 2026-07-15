"""
Joint Embedding Space Analysis for Cross-Platform Theme Discovery.

Replaces the separate per-platform clustering + manual cross-platform
theme consolidation with a single joint clustering step that naturally
produces shared, Meta-dominant, and Bluesky-dominant themes.

Inputs expected:
    - meta_texts.csv   : columns [ad_id, text, stance]
    - bluesky_texts.csv: columns [post_id, text, stance]

Outputs:
    - joint_clusters.csv        : every text with its cluster assignment,
                                  platform composition stats, and theme category
    - joint_themes.csv          : one row per theme with framing vectors,
                                  platform skew, and summary
    - platform_skew_dist.png    : histogram of platform skew per cluster
    - joint_umap.png            : UMAP coloured by platform + theme category

Pipeline stages:
    1.  Load and embed texts from both platforms jointly
    2.  Dimensionality reduction (PCA -> UMAP) in joint space
    3.  HDBSCAN clustering
    4.  Compute platform composition ratio per cluster
    5.  Classify clusters: Shared / Meta-dominant / Bluesky-dominant
    6.  LLM coherence filtering
    7.  LLM summarization + theme label generation for coherent clusters
    8.  LLM framing dimension extraction
    9.  Recursive reclustering of incoherent clusters
        (local re-UMAP + HDBSCAN on each subset, up to MAX_RECLUSTER_DEPTH)
    10. LLM coherence + labeling for sub-clusters
    11. Outlier assignment: nearest-centroid LLM check for noise posts
    ─────────
    12. Export

Dependencies:
    pip install sentence-transformers hdbscan umap-learn scikit-learn
                pandas numpy matplotlib seaborn httpx tqdm
"""

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import hdbscan
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

META_CSV       = Path("ccombined_deduped_sbert080.csv")
BLUESKY_CSV    = Path("combined_deduped_sbert080.csv")
TEXT_COL       = "text"
ID_COL_META    = "ad_archive_id"
ID_COL_BLUESKY = "cid"

# SBERT model
SBERT_MODEL = "all-MiniLM-L6-v2"

# Dimensionality reduction (global)
PCA_COMPONENTS   = 100
UMAP_COMPONENTS  = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1

# HDBSCAN (global)
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES      = 1

# Platform composition thresholds
DOMINANT_THRESHOLD = 0.70

# Top-k representative texts for LLM steps (global clusters)
TOP_K = 5

# ---------------------------------------------------------------------------
# Reclustering configuration  
# ---------------------------------------------------------------------------

# Maximum recursion depth for incoherent cluster splitting.
# Depth 0 = initial global clustering; depth 1 = first recluster pass, etc.
MAX_RECLUSTER_DEPTH = 2

# Local UMAP parameters used when reclustering a subset.
# Fewer neighbors captures tighter local structure within a subset.
RECLUSTER_UMAP_COMPONENTS = 5
RECLUSTER_UMAP_NEIGHBORS  = 10
RECLUSTER_UMAP_MIN_DIST   = 0.05

# HDBSCAN min_cluster_size for sub-clustering is computed dynamically:
#   max(RECLUSTER_MIN_CLUSTER_FLOOR, subset_size // RECLUSTER_SIZE_DIVISOR)
# Lower divisor → smaller min_cluster_size → fewer posts ejected as sub-noise.
# 200 is the safe default: 16800 posts → min_cluster_size=84 (was 168 at 100).
RECLUSTER_MIN_CLUSTER_FLOOR   = 10
RECLUSTER_SIZE_DIVISOR        = 200   # e.g. 16800 posts → min_cluster_size=84

# After local HDBSCAN, border points (low membership probability) that land
# in noise are soft-assigned to their nearest sub-cluster centroid if their
# cosine similarity exceeds this threshold. This reclaims most border posts
# without inflating the outlier pool.
SOFT_ASSIGN_SIMILARITY_THRESHOLD = 0.50

# Outlier assignment uses ONLY cosine similarity — no LLM calls.
# Posts whose nearest-centroid similarity meets this threshold are assigned
# to that cluster; everything below is marked "outlier".
# Tune this after inspecting the similarity histogram logged during the run.
OUTLIER_AUTO_ASSIGN_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Mistral API
# ---------------------------------------------------------------------------

MISTRAL_MODEL = "mistral-large-2407"
API_KEY       = os.getenv("MISTRAL_API_KEY")
BASE_URL      = "https://api.mistral.ai/v1"
API_URL       = f"{BASE_URL}/chat/completions"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_DIR        = Path("joint_analysis_outputs")
OUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClusterTheme:
    cluster_id:         int
    parent_cluster_id:  int           # -1 if top-level cluster
    recluster_depth:    int           # 0 = top-level
    theme_label:        str
    summary:            str
    platform_skew:      float
    category:           str
    n_meta:             int
    n_bluesky:          int
    n_total:            int
    coherent:           bool
    actor_institution:  float = 0.5
    problem_economic:   float = 0.5
    problem_moral:      float = 0.5
    problem_scientific: float = 0.5
    problem_political:  float = 0.5
    solution_present:   bool  = False
    temporal_future:    float = 0.5
    temporal_crisis:    float = 0.5
    emotional_urgency:  float = 0.5
    emotional_hope:     float = 0.5


# ---------------------------------------------------------------------------
# Stage 1: Load and embed
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    meta    = pd.read_csv(META_CSV)
    bluesky = pd.read_csv(BLUESKY_CSV)

    meta["platform"]    = "meta"
    bluesky["platform"] = "bluesky"

    meta    = meta.rename(columns={ID_COL_META:    "doc_id"})
    bluesky = bluesky.rename(columns={ID_COL_BLUESKY: "doc_id"})

    combined = pd.concat([meta, bluesky], ignore_index=True)
    combined = combined.dropna(subset=[TEXT_COL])
    combined = combined[combined[TEXT_COL].str.strip() != ""]
    combined = combined.reset_index(drop=True)

    log.info(f"Loaded {len(meta)} Meta + {len(bluesky)} Bluesky = {len(combined)} total texts.")
    return combined


def embed_texts(df: pd.DataFrame) -> np.ndarray:
    log.info(f"Embedding {len(df)} texts with {SBERT_MODEL}...")
    model      = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(
        df[TEXT_COL].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = normalize(embeddings, norm="l2")
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_texts_cached(df: pd.DataFrame) -> np.ndarray:
    path = CHECKPOINT_DIR / "embeddings.npy"
    if path.exists():
        log.info(f"Loading cached embeddings from {path}")
        return np.load(path)
    embeddings = embed_texts(df)
    np.save(path, embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# Stage 2: Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (pca_reduced, umap_reduced).
    pca_reduced is kept for reclustering (local re-UMAP on subsets).
    umap_reduced is used for global HDBSCAN clustering.
    """
    log.info(f"PCA: {embeddings.shape[1]}d → {PCA_COMPONENTS}d...")
    pca         = PCA(n_components=PCA_COMPONENTS, random_state=42)
    pca_reduced = pca.fit_transform(embeddings)
    log.info(f"Explained variance (PCA): {pca.explained_variance_ratio_.sum():.3f}")

    log.info(f"UMAP: {PCA_COMPONENTS}d → {UMAP_COMPONENTS}d...")
    reducer      = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    umap_reduced = reducer.fit_transform(pca_reduced)
    return pca_reduced, umap_reduced


def reduce_dimensions_cached(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca_path  = CHECKPOINT_DIR / "pca_reduced.npy"
    umap_path = CHECKPOINT_DIR / "umap_reduced.npy"

    if pca_path.exists() and umap_path.exists():
        log.info("Loading cached PCA and UMAP representations.")
        return np.load(pca_path), np.load(umap_path)

    pca_reduced, umap_reduced = reduce_dimensions(embeddings)
    np.save(pca_path,  pca_reduced)
    np.save(umap_path, umap_reduced)
    log.info(f"Saved PCA → {pca_path},  UMAP → {umap_path}")
    return pca_reduced, umap_reduced


# ---------------------------------------------------------------------------
# Stage 3: HDBSCAN clustering
# ---------------------------------------------------------------------------

def cluster(
    reduced: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int      = HDBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    log.info(
        f"HDBSCAN (min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples})..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.2,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    probs  = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(f"  → {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels):.1%})")
    return labels, probs


# ---------------------------------------------------------------------------
# Stage 4 & 5: Platform composition
# ---------------------------------------------------------------------------

def compute_platform_composition(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()
    df["cluster_id"] = labels

    clustered   = df[df["cluster_id"] != -1]
    composition = (
        clustered.groupby("cluster_id")["platform"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={"meta": "n_meta", "bluesky": "n_bluesky"})
    )
    for col in ("n_meta", "n_bluesky"):
        if col not in composition.columns:
            composition[col] = 0

    composition["n_total"]       = composition["n_meta"] + composition["n_bluesky"]
    composition["platform_skew"] = composition["n_meta"] / composition["n_total"]

    def categorise(skew: float) -> str:
        if skew >= DOMINANT_THRESHOLD:
            return "meta_dominant"
        elif skew <= (1 - DOMINANT_THRESHOLD):
            return "bluesky_dominant"
        return "shared"

    composition["category"] = composition["platform_skew"].apply(categorise)

    counts = composition["category"].value_counts()
    log.info(
        f"  Cluster categories — shared: {counts.get('shared', 0)}, "
        f"meta_dominant: {counts.get('meta_dominant', 0)}, "
        f"bluesky_dominant: {counts.get('bluesky_dominant', 0)}"
    )
    return composition


def get_representative_texts(
    df: pd.DataFrame,
    global_indices: np.ndarray,   # row indices into df for this cluster
    probs: np.ndarray,             # membership probs aligned with global_indices
    k: int = TOP_K,
) -> list[str]:
    """Returns top-k texts by HDBSCAN membership probability."""
    order   = np.argsort(probs)[::-1][:k]
    indices = global_indices[order]
    return df.iloc[indices][TEXT_COL].tolist()


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_mistral(messages: list[dict], max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      MISTRAL_MODEL,
        "messages":   messages,
        "max_tokens": max_tokens,
    }
    for attempt in range(2):
        try:
            resp = httpx.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"Mistral call failed (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(3)
    return ""


# ---------------------------------------------------------------------------
# Stage 6: LLM coherence filtering
# ---------------------------------------------------------------------------

COHERENCE_PROMPT = """\
You are evaluating whether a set of texts share a consistent underlying theme.

Given the following {k} texts, determine whether they are COHERENT \
(all addressing the same core theme) or INCOHERENT (addressing different \
themes or too mixed to summarise meaningfully).

Texts:
{texts}

Reply with exactly one word: COHERENT or INCOHERENT. \
Then on a new line, give a one-sentence reason."""


def check_coherence(texts: list[str]) -> bool:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt   = COHERENCE_PROMPT.format(k=len(texts), texts=numbered)
    answer   = _call_mistral([{"role": "user", "content": prompt}], max_tokens=60)
    return answer.upper().startswith("COHERENT")


# ---------------------------------------------------------------------------
# Stage 7: LLM summarisation + theme label
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
Summarise the following {k} texts in 1–3 sentences (≤100 words), \
capturing their shared theme, framing, and communicative intent.

Texts:
{texts}

Summary:"""

THEME_PROMPT = """\
Produce a short theme label (1–3 words) for the following cluster summary.
The label should capture the central idea, stance, and/or topic.

Examples: 'Clean energy advocacy', 'Fossil fuel corruption', 'Climate denial backlash'

Summary: {summary}

Theme label:"""


def generate_summary(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt   = SUMMARY_PROMPT.format(k=len(texts), texts=numbered)
    return _call_mistral([{"role": "user", "content": prompt}], max_tokens=150)


def generate_theme_label(summary: str) -> str:
    prompt = THEME_PROMPT.format(summary=summary)
    result = _call_mistral([{"role": "user", "content": prompt}], max_tokens=20)
    return result.strip('"').strip("'")


# ---------------------------------------------------------------------------
# Stage 8: LLM framing dimension extraction
# ---------------------------------------------------------------------------

FRAMING_PROMPT = """\
You are a computational social science researcher analysing climate discourse framing.

Given the following cluster summary, score each framing dimension on a scale of 0.0 to 1.0.

Summary: {summary}

Return ONLY a JSON object with these exact keys:
{{
  "actor_institution":  <0.0-1.0>,
  "problem_economic":   <0.0-1.0>,
  "problem_moral":      <0.0-1.0>,
  "problem_scientific": <0.0-1.0>,
  "problem_political":  <0.0-1.0>,
  "solution_present":   <true|false>,
  "temporal_future":    <0.0-1.0>,
  "temporal_crisis":    <0.0-1.0>,
  "emotional_urgency":  <0.0-1.0>,
  "emotional_hope":     <0.0-1.0>
}}"""


def extract_framing(summary: str) -> dict:
    prompt = FRAMING_PROMPT.format(summary=summary)
    raw    = _call_mistral([{"role": "user", "content": prompt}], max_tokens=200)
    raw    = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Could not parse framing JSON: {raw[:100]}")
        return {
            "actor_institution": 0.5, "problem_economic": 0.5,
            "problem_moral": 0.5, "problem_scientific": 0.5,
            "problem_political": 0.5, "solution_present": False,
            "temporal_future": 0.5, "temporal_crisis": 0.5,
            "emotional_urgency": 0.5, "emotional_hope": 0.5,
        }


# ---------------------------------------------------------------------------
# Shared: LLM label generation for a single cluster
# ---------------------------------------------------------------------------

def label_cluster(
    cluster_id:        int,
    parent_cluster_id: int,
    recluster_depth:   int,
    rep_texts:         list[str],
    composition_row:   pd.Series,
) -> dict:
    """
    Runs coherence → summary → theme_label → framing for one cluster.
    Returns a record dict ready to append to theme_records.
    """
    time.sleep(0.5)
    coherent = check_coherence(rep_texts)

    summary     = ""
    theme_label = ""
    framing     = {}

    if coherent:
        time.sleep(0.5)
        summary = generate_summary(rep_texts)
        time.sleep(0.5)
        theme_label = generate_theme_label(summary)
        time.sleep(0.5)
        framing = extract_framing(summary)

    return {
        "cluster_id":         cluster_id,
        "parent_cluster_id":  parent_cluster_id,
        "recluster_depth":    recluster_depth,
        "theme_label":        theme_label,
        "summary":            summary,
        "platform_skew":      composition_row["platform_skew"],
        "category":           composition_row["category"],
        "n_meta":             composition_row["n_meta"],
        "n_bluesky":          composition_row["n_bluesky"],
        "n_total":            composition_row["n_total"],
        "coherent":           coherent,
        **framing,
    }


# ---------------------------------------------------------------------------
# Stage 9: Recursive reclustering of incoherent clusters  (NEW)
# ---------------------------------------------------------------------------

def recluster_subset(
    pca_subset: np.ndarray,
    subset_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a local UMAP → HDBSCAN on a subset of PCA-reduced embeddings.

    After HDBSCAN, any posts labelled noise (-1) are soft-assigned to their
    nearest sub-cluster centroid in PCA space if cosine similarity exceeds
    SOFT_ASSIGN_SIMILARITY_THRESHOLD.  This reclaims border posts that
    HDBSCAN's density model rejects but that are semantically close to a
    real sub-cluster, preventing them from inflating the global outlier pool.
    """
    n_neighbors = min(RECLUSTER_UMAP_NEIGHBORS, subset_size - 1)
    local_umap  = umap.UMAP(
        n_components=RECLUSTER_UMAP_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=RECLUSTER_UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    local_reduced = local_umap.fit_transform(pca_subset)

    min_cluster_size = max(
        RECLUSTER_MIN_CLUSTER_FLOOR,
        subset_size // RECLUSTER_SIZE_DIVISOR,
    )
    log.info(
        f"  Local HDBSCAN on {subset_size} posts "
        f"(min_cluster_size={min_cluster_size})..."
    )
    clusterer  = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,   # tighter than global (0.2)
        prediction_data=True,
    )
    sub_labels = clusterer.fit_predict(local_reduced)
    sub_probs  = clusterer.probabilities_

    # ── Soft assignment of border/noise points ────────────────────────────
    noise_local = sub_labels == -1
    n_noise_before = noise_local.sum()

    if noise_local.any():
        unique_sub = sorted(set(sub_labels) - {-1})
        if unique_sub:
            # Centroids in PCA space (not UMAP — more stable for similarity)
            centroids = np.vstack([
                pca_subset[sub_labels == cid].mean(axis=0)
                for cid in unique_sub
            ])
            noise_pca   = pca_subset[noise_local]
            sims        = cosine_similarity(noise_pca, centroids)  # (n_noise, n_clusters)
            best_sim    = sims.max(axis=1)
            best_cid    = np.array(unique_sub)[sims.argmax(axis=1)]

            assign_mask = best_sim >= SOFT_ASSIGN_SIMILARITY_THRESHOLD
            noise_idx   = np.where(noise_local)[0]

            sub_labels[noise_idx[assign_mask]]  = best_cid[assign_mask]
            sub_probs[noise_idx[assign_mask]]   = best_sim[assign_mask]

            n_recovered = assign_mask.sum()
            log.info(
                f"  Soft assignment: recovered {n_recovered}/{n_noise_before} "
                f"border posts (threshold={SOFT_ASSIGN_SIMILARITY_THRESHOLD}). "
                f"{n_noise_before - n_recovered} remain as sub-noise."
            )

    n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    n_noise    = (sub_labels == -1).sum()
    log.info(f"  → {n_clusters} sub-clusters, {n_noise} noise ({n_noise/subset_size:.1%})")
    return sub_labels, sub_probs


def process_incoherent_clusters(
    df:            pd.DataFrame,
    pca_reduced:   np.ndarray,
    theme_records: list[dict],
    depth:         int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Iterates over all incoherent clusters (coherent=False in theme_records),
    reclusters their posts, and recursively labels sub-clusters.

    Sub-cluster IDs are allocated starting from max(existing cluster_id) + 1
    to avoid collisions with existing IDs.

    Returns an updated df (with new cluster_id / theme assignments for posts
    that moved into coherent sub-clusters) and an extended theme_records list.
    """
    if depth >= MAX_RECLUSTER_DEPTH:
        log.info(f"Max reclustering depth ({MAX_RECLUSTER_DEPTH}) reached — stopping.")
        return df, theme_records

    incoherent = [r for r in theme_records if not r["coherent"]]
    if not incoherent:
        log.info("No incoherent clusters to recluster.")
        return df, theme_records

    log.info(
        f"[Depth {depth}] Reclustering {len(incoherent)} incoherent cluster(s) "
        f"covering {sum(r['n_total'] for r in incoherent):,} posts..."
    )

    # Track highest cluster ID so we can allocate new ones without collision
    next_cluster_id = int(df["cluster_id"].max()) + 1

    new_theme_records: list[dict] = []

    for record in tqdm(incoherent, desc=f"Reclustering (depth {depth})"):
        parent_cid = record["cluster_id"]
        mask       = (df["cluster_id"] == parent_cid).values
        subset_idx = np.where(mask)[0]           # row indices into df
        subset_pca = pca_reduced[subset_idx]     # PCA embeddings for subset

        if len(subset_idx) < RECLUSTER_MIN_CLUSTER_FLOOR * 2:
            # Too small to split further — stays incoherent / noise
            log.info(f"  Cluster {parent_cid}: too small to recluster ({len(subset_idx)} posts), keeping as outlier.")
            df.loc[mask, "cluster_id"] = -1   # demote to noise
            continue

        sub_labels, sub_probs = recluster_subset(subset_pca, len(subset_idx))

        # Map sub-cluster labels (0,1,2,...) to globally unique IDs
        unique_sub = sorted(set(sub_labels) - {-1})
        sub_id_map: dict[int, int] = {}
        for local_id in unique_sub:
            sub_id_map[local_id] = next_cluster_id
            next_cluster_id += 1

        # Write new cluster IDs back into df
        for local_id, global_id in sub_id_map.items():
            local_mask = sub_labels == local_id
            df_rows    = subset_idx[local_mask]
            df.iloc[df_rows, df.columns.get_loc("cluster_id")] = global_id
            df.iloc[df_rows, df.columns.get_loc("prob")]        = sub_probs[local_mask]

        # Posts sub-HDBSCAN couldn't assign → keep as noise
        noise_mask = sub_labels == -1
        if noise_mask.any():
            df_rows = subset_idx[noise_mask]
            df.iloc[df_rows, df.columns.get_loc("cluster_id")] = -1

        # Compute composition for each sub-cluster and label
        for local_id, global_id in sub_id_map.items():
            local_mask   = sub_labels == local_id
            df_rows      = subset_idx[local_mask]
            sub_df       = df.iloc[df_rows]
            sub_probs_cl = sub_probs[local_mask]

            n_meta    = int((sub_df["platform"] == "meta").sum())
            n_bluesky = int((sub_df["platform"] == "bluesky").sum())
            n_total   = n_meta + n_bluesky
            skew      = n_meta / n_total if n_total > 0 else 0.5
            category  = (
                "meta_dominant"    if skew >= DOMINANT_THRESHOLD else
                "bluesky_dominant" if skew <= (1 - DOMINANT_THRESHOLD) else
                "shared"
            )

            comp_row = pd.Series({
                "platform_skew": skew,
                "category":      category,
                "n_meta":        n_meta,
                "n_bluesky":     n_bluesky,
                "n_total":       n_total,
            })

            # Top-k texts by membership probability
            order     = np.argsort(sub_probs_cl)[::-1][:TOP_K]
            rep_texts = sub_df.iloc[order][TEXT_COL].tolist()

            log.info(
                f"  Sub-cluster {global_id} (parent={parent_cid}, "
                f"n={n_total}, depth={depth+1}): labeling..."
            )
            rec = label_cluster(
                cluster_id=global_id,
                parent_cluster_id=parent_cid,
                recluster_depth=depth + 1,
                rep_texts=rep_texts,
                composition_row=comp_row,
            )
            new_theme_records.append(rec)

        n_sub_noise = noise_mask.sum()
        log.info(
            f"  Cluster {parent_cid}: split into {len(unique_sub)} sub-clusters, "
            f"{n_sub_noise} posts demoted to noise."
        )

    # Merge new records into theme_records
    theme_records.extend(new_theme_records)

    # Recurse on any still-incoherent sub-clusters
    if depth + 1 < MAX_RECLUSTER_DEPTH:
        df, theme_records = process_incoherent_clusters(
            df, pca_reduced, theme_records, depth=depth + 1
        )

    return df, theme_records


# ---------------------------------------------------------------------------
# Stage 11: Outlier assignment  
# ---------------------------------------------------------------------------


def assign_outliers(
    df:            pd.DataFrame,
    pca_reduced:   np.ndarray,
    theme_records: list[dict],
) -> pd.DataFrame:
    """
    Assigns noise posts (cluster_id == -1) using cosine similarity only.
    Zero LLM calls — this runs in seconds regardless of outlier count.

    Posts whose similarity to the nearest coherent cluster centroid meets
    OUTLIER_AUTO_ASSIGN_THRESHOLD are absorbed into that cluster.
    Everything below is marked theme_label='outlier'.

    After running, check the logged similarity histogram to decide whether
    to raise or lower OUTLIER_AUTO_ASSIGN_THRESHOLD.
    """
    coherent_records = [r for r in theme_records if r.get("coherent") and r.get("theme_label")]
    if not coherent_records:
        log.warning("No coherent clusters available for outlier assignment.")
        return df

    noise_mask = df["cluster_id"] == -1
    n_noise    = noise_mask.sum()
    log.info(f"Outlier assignment (similarity-only) for {n_noise} posts...")

    # Build centroids in PCA space
    centroids:     list[np.ndarray] = []
    centroid_recs: list[dict]       = []
    for rec in coherent_records:
        cid  = rec["cluster_id"]
        rows = (df["cluster_id"] == cid).values
        if rows.any():
            centroids.append(pca_reduced[rows].mean(axis=0))
            centroid_recs.append(rec)

    if not centroids:
        return df

    centroid_matrix = np.vstack(centroids)
    noise_idx       = np.where(noise_mask.values)[0]
    noise_pca       = pca_reduced[noise_idx]
    similarities    = cosine_similarity(noise_pca, centroid_matrix)
    best_sim        = similarities.max(axis=1)
    best_cid_local  = similarities.argmax(axis=1)

    # Log similarity distribution so the threshold can be tuned
    for pct in [25, 50, 75, 90, 95]:
        log.info(f"  Similarity p{pct}: {np.percentile(best_sim, pct):.3f}")

    assign_mask = best_sim >= OUTLIER_AUTO_ASSIGN_THRESHOLD
    reject_mask = ~assign_mask

    for i in np.where(assign_mask)[0]:
        df_row_idx = noise_idx[i]
        rec = centroid_recs[best_cid_local[i]]
        df.at[df_row_idx, "cluster_id"]     = rec["cluster_id"]
        df.at[df_row_idx, "theme_label"]    = rec["theme_label"]
        df.at[df_row_idx, "theme_category"] = rec["category"]

    for i in np.where(reject_mask)[0]:
        df_row_idx = noise_idx[i]
        df.at[df_row_idx, "theme_label"]    = "outlier"
        df.at[df_row_idx, "theme_category"] = "outlier"

    n_assigned = assign_mask.sum()
    n_outlier  = reject_mask.sum()
    total      = len(df)
    log.info(
        f"Outlier assignment complete — "
        f"assigned: {n_assigned}, kept as outlier: {n_outlier} "
        f"({n_outlier/total:.1%} of all posts)"
    )
    return df


# ---------------------------------------------------------------------------
# Visualisations 
# ---------------------------------------------------------------------------

def plot_platform_skew(composition: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    composition["platform_skew"].hist(bins=20, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(1 - DOMINANT_THRESHOLD, color="red",    linestyle="--",
               label=f"Bluesky-dominant threshold ({1-DOMINANT_THRESHOLD})")
    ax.axvline(DOMINANT_THRESHOLD,     color="orange", linestyle="--",
               label=f"Meta-dominant threshold ({DOMINANT_THRESHOLD})")
    ax.set_xlabel("Platform skew (0 = all Bluesky, 1 = all Meta)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Distribution of platform composition across joint clusters")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUT_DIR / "platform_skew_dist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved: {path}")


def plot_joint_umap(
    df:          pd.DataFrame,
    reduced:     np.ndarray,
    composition: pd.DataFrame,
):
    log.info("Fitting 2D UMAP for visualisation...")
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1, metric="cosine", random_state=42,
    )
    n   = min(len(df), 10_000)
    idx = np.random.choice(len(df), n, replace=False)
    coords = reducer_2d.fit_transform(reduced[idx])

    plot_df = df.iloc[idx].copy().reset_index(drop=True)
    plot_df["umap_x"]     = coords[:, 0]
    plot_df["umap_y"]     = coords[:, 1]
    plot_df["cluster_id"] = df.iloc[idx]["cluster_id"].values

    def get_category(cid):
        if cid == -1:
            return "noise"
        try:
            return composition.loc[cid, "category"]
        except KeyError:
            return "sub-cluster"

    plot_df["category"] = plot_df["cluster_id"].apply(get_category)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    palette_platform = {"meta": "#1877F2", "bluesky": "#0085FF"}
    for platform, grp in plot_df.groupby("platform"):
        axes[0].scatter(
            grp["umap_x"], grp["umap_y"], s=2, alpha=0.4,
            color=palette_platform.get(platform, "grey"),
            label=platform.capitalize(),
        )
    axes[0].set_title("Joint UMAP — by platform")
    axes[0].legend(markerscale=4, fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    palette_cat = {
        "shared":           "#2ca02c",
        "meta_dominant":    "#1877F2",
        "bluesky_dominant": "#ff7f0e",
        "noise":            "#cccccc",
        "outlier":          "#888888",
        "sub-cluster":      "#9467bd",
    }
    for cat, grp in plot_df.groupby("category"):
        axes[1].scatter(
            grp["umap_x"], grp["umap_y"], s=2, alpha=0.5,
            color=palette_cat.get(cat, "grey"),
            label=cat.replace("_", " ").capitalize(),
        )
    axes[1].set_title("Joint UMAP — by theme category")
    axes[1].legend(markerscale=4, fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    plt.suptitle("Joint embedding space: Meta ads + Bluesky posts", fontsize=12)
    plt.tight_layout()
    path = OUT_DIR / "joint_umap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved: {path}")


def plot_framing_heatmap(themes_df: pd.DataFrame):
    framing_cols = [
        "actor_institution", "problem_economic", "problem_moral",
        "problem_scientific", "problem_political",
        "temporal_future", "temporal_crisis",
        "emotional_urgency", "emotional_hope",
    ]
    existing = [c for c in framing_cols if c in themes_df.columns]
    if not existing:
        return

    summary = (
        themes_df[themes_df["coherent"]]
        .groupby("category")[existing]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Mean framing dimension scores by cluster category")
    ax.set_xlabel("Framing dimension")
    ax.set_ylabel("Cluster category")
    plt.tight_layout()
    path = OUT_DIR / "framing_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    # ── Stages 1-2 ───────────────────────────────────────────────────────
    df         = load_data()
    embeddings = embed_texts_cached(df)
    pca_reduced, umap_reduced = reduce_dimensions_cached(embeddings)

    # ── Stage 3: Global HDBSCAN ──────────────────────────────────────────
    labels, probs   = cluster(umap_reduced)
    df["cluster_id"] = labels
    df["prob"]       = probs

    # ── Stages 4-5: Platform composition ─────────────────────────────────
    composition = compute_platform_composition(df, labels)
    df = df.merge(
        composition[["category", "platform_skew", "n_meta", "n_bluesky", "n_total"]],
        left_on="cluster_id", right_index=True, how="left",
    )

    plot_platform_skew(composition)
    plot_joint_umap(df, umap_reduced, composition)

    if not API_KEY:
        log.warning("MISTRAL_API_KEY not set — skipping all LLM stages.")
        df.to_csv(OUT_DIR / "joint_clusters.csv", index=False)
        return

    # ── Stages 6-8: LLM label generation for top-level clusters ─────────
    cluster_ids = sorted(
        [c for c in composition.index if c != -1],
        key=lambda c: composition.loc[c, "n_total"],
        reverse=True,
    )

    theme_records: list[dict] = []

    for cid in tqdm(cluster_ids, desc="LLM labeling (top-level)"):
        mask      = labels == cid
        idx       = np.where(mask)[0]
        probs_cid = probs[mask]
        order     = np.argsort(probs_cid)[::-1][:TOP_K]
        rep_texts = df.iloc[idx[order]][TEXT_COL].tolist()

        if not rep_texts:
            continue

        rec = label_cluster(
            cluster_id=cid,
            parent_cluster_id=-1,
            recluster_depth=0,
            rep_texts=rep_texts,
            composition_row=composition.loc[cid],
        )
        theme_records.append(rec)

    n_coherent   = sum(1 for r in theme_records if r["coherent"])
    n_incoherent = sum(1 for r in theme_records if not r["coherent"])
    log.info(f"Top-level: {n_coherent} coherent, {n_incoherent} incoherent clusters.")

    # ── Stage 9-10: Recursive reclustering of incoherent clusters ────────
    df, theme_records = process_incoherent_clusters(
        df=df,
        pca_reduced=pca_reduced,
        theme_records=theme_records,
        depth=0,
    )

    # ── Stage 11: Outlier assignment ──────────────────────────────────────
    df = assign_outliers(df, pca_reduced, theme_records)

    # ── Framing heatmap ───────────────────────────────────────────────────
    themes_df = pd.DataFrame(theme_records)
    plot_framing_heatmap(themes_df)

    # ── Assign theme labels back into df ─────────────────────────────────
    theme_map    = themes_df.set_index("cluster_id")["theme_label"].to_dict()
    category_map = themes_df.set_index("cluster_id")["category"].to_dict()

    # Only overwrite for posts still in a labeled cluster
    # (outlier assignment already wrote theme_label for reassigned posts)
    assigned_mask = df["cluster_id"] != -1
    df.loc[assigned_mask, "theme_label"]   = df.loc[assigned_mask, "cluster_id"].map(theme_map)
    df.loc[assigned_mask, "theme_category"] = df.loc[assigned_mask, "cluster_id"].map(category_map)

    # Posts still at cluster_id=-1 are true outliers
    df["theme_label"]   = df["theme_label"].fillna("outlier")
    df["theme_category"] = df["theme_category"].fillna("outlier")

    # Add lineage columns
    depth_map  = themes_df.set_index("cluster_id")["recluster_depth"].to_dict()
    parent_map = themes_df.set_index("cluster_id")["parent_cluster_id"].to_dict()
    df["recluster_depth"]    = df["cluster_id"].map(depth_map).fillna(-1).astype(int)
    df["parent_cluster_id"]  = df["cluster_id"].map(parent_map).fillna(-1).astype(int)

    # ── Save outputs ──────────────────────────────────────────────────────
    df.to_csv(OUT_DIR / "joint_clusters.csv", index=False)
    themes_df.to_csv(OUT_DIR / "joint_themes.csv", index=False)
    log.info(f"All outputs saved to {OUT_DIR}/")

    # ── Summary statistics ─────────────────────────────────────────────────
    n_total     = len(df)
    n_outlier   = (df["theme_label"] == "outlier").sum()
    n_labeled   = n_total - n_outlier
    n_subclusters = (themes_df["recluster_depth"] > 0).sum()

    print("\n── Joint clustering summary ──────────────────────────────")
    print(f"  Total posts:              {n_total:,}")
    print(f"  Labeled (theme assigned): {n_labeled:,}  ({n_labeled/n_total:.1%})")
    print(f"  Outliers:                 {n_outlier:,}  ({n_outlier/n_total:.1%})")
    print(f"  Top-level clusters:       {len(cluster_ids)}")
    print(f"  Sub-clusters (depth≥1):   {n_subclusters}")
    print(f"  Total coherent themes:    {themes_df['coherent'].sum()}")

    cat_counts = themes_df[themes_df["coherent"]]["category"].value_counts()
    print(f"  Shared:                   {cat_counts.get('shared', 0)}")
    print(f"  Meta-dominant:            {cat_counts.get('meta_dominant', 0)}")
    print(f"  Bluesky-dominant:         {cat_counts.get('bluesky_dominant', 0)}")

    top_shared = themes_df[
        (themes_df["category"] == "shared") & themes_df["coherent"]
    ].sort_values("n_total", ascending=False)
    print("\n  Top shared themes:")
    for _, r in top_shared.head(10).iterrows():
        depth_str = f" [depth={int(r['recluster_depth'])}]" if r["recluster_depth"] > 0 else ""
        print(
            f"    [{r['theme_label']}]{depth_str}  "
            f"Meta={r['n_meta']} Bsky={r['n_bluesky']}  skew={r['platform_skew']:.2f}"
        )


if __name__ == "__main__":
    run_pipeline()