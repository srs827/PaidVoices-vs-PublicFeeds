import re, regex as re2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_CSV  = "bluesky_pipeline/All_Data_Jul_2024-Sept_2025.csv" # or meta input file (all data)
OUTPUT_CSV = "combined_deduped_sbert080.csv"
OUTPUT_NPY = "combined_deduped_sbert080.embeddings.npy"

ID_COL   = "cid" # ad_archive_id for meta
TEXT_COL = "text" # ad_creative_bodies for meta

SIM_THRESH = 0.80
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
BATCH_SIZE = 256

# regexes 
URL_RE    = re.compile(r"http\\S+|www\\.\\S+")
SPACE_RE  = re.compile(r"\\s+")
EMOJI_RE  = re2.compile(r"\\p{Extended_Pictographic}")

# normalize text by lowercasing, stripping URLs/emojis/punct, cleaning spaces
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


df = pd.read_csv(INPUT_CSV)
df = df.sample(n=20000, random_state=42)
if ID_COL not in df.columns or TEXT_COL not in df.columns:
    raise ValueError(f"Expected columns '{ID_COL}' and '{TEXT_COL}' in the CSV.")

# clean up text, drop url/emoji-only rows
df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)
df["_clean_text"] = df[TEXT_COL].astype(str).map(clean_text)
df = df[df["_clean_text"].str.len() > 0].reset_index(drop=True)

print(f"Embedding {len(df)} rows with {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
texts = df["_clean_text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
embeddings = embeddings.astype(np.float32, copy=False)
n = embeddings.shape[0]
to_drop = np.zeros(n, dtype=bool)

print(f"Greedy dedup: drop any later ad with cosine sim > {SIM_THRESH}")
for i in tqdm(range(n), desc="Dedup (cosine)"):
    if to_drop[i]:
        continue  # already dropped due to a previous match

    # Compute cosine sims of ad i with all later ads (vectorized)
    tail = embeddings[i+1:]                      # shape (n-i-1, d)
    if tail.size == 0:
        break
    sims = tail @ embeddings[i]                  # shape (n-i-1,), dot products

    # Find duplicates above threshold and drop them
    dup_rel_idx = np.where(sims > SIM_THRESH)[0]     # relative to i+1
    if dup_rel_idx.size:
        to_drop[(i + 1) + dup_rel_idx] = True

kept_mask = ~to_drop
kept_df = df.loc[kept_mask].reset_index(drop=True)
kept_embeddings = embeddings[kept_mask]

print(f"Kept {kept_mask.sum()} / {n} ads "
      f"({(kept_mask.sum()/n)*100:.2f}%), removed {to_drop.sum()} duplicates.")

# Save results
kept_df.drop(columns=["_clean_text"]).to_csv(OUTPUT_CSV, index=False)
np.save(OUTPUT_NPY, kept_embeddings)
kept_df[[ID_COL]].to_csv("kept_ids80.csv", index=False)
print(f"Wrote: {OUTPUT_CSV}, {OUTPUT_NPY}, kept_ids80.csv")
