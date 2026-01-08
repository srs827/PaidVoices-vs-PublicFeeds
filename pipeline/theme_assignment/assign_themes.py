# save as: assign_with_llm_theme_batches.py

import os
import json
import time
import random
import pandas as pd
from typing import Dict, Any, List, Tuple
import httpx
import re
from textwrap import shorten

# replace with proper paths for meta/bluesky
ADS_WITH_CLUSTERS_PATH = "clustering_code/ads_with_clusters.csv"
CLUSTER_THEMES_CSV     = "coherency_check/cluster_themes.csv"

OUT_JSONL = "assigned_themes.jsonl"
OUT_CSV   = "assigned_themes.csv"

API_KEY      = os.getenv("MISTRAL_API_KEY")
BASE_URL = "https://api.mistral.ai/v1"
API_URL  = f"{BASE_URL}/chat/completions"
MODEL_ID = "mistral-large-2407"

# llm settings
TEMPERATURE = 0.2
TOP_P       = 0.9
MAX_TOKENS  = 512

# batching
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 0.5
MAX_ADS_TO_CLASSIFY = None  # e.g., 500 for testing

# debug
DEBUG_VERBOSE      = True
DEBUG_DUMP_TO_DISK = True
DEBUG_DIR          = "llm_batch_debug"

def _ensure_debug_dir():
    if DEBUG_DUMP_TO_DISK:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def _dump_text(tag: str, batch_idx: int, retry_num: int, content: str):
    if not DEBUG_DUMP_TO_DISK:
        return
    _ensure_debug_dir()
    suffix = f"batch{batch_idx:05d}_retry{retry_num}"
    path = os.path.join(DEBUG_DIR, f"{tag}_{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")

def normalize_theme(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def normalize_theme_loose(s: str) -> str:
    s = normalize_theme(s)
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def strip_md_quotes(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'^\*{1,3}\s*', '', s)     # leading *, **, ***
    s = re.sub(r'\s*\*{1,3}$', '', s)     # trailing *, **, ***
    s = s.strip("`'\"“”‘’")
    return s.strip()


def load_cluster_themes(csv_path: str) -> Tuple[List[Dict[str, Any]], Dict[int, str], set]:
    """
    CSV must have columns: cluster (int), theme (str)
    Returns:
      themes_list: [{"cluster": int, "theme": str}, ...] sorted by cluster
      theme_lookup: {cluster_id: theme_str}
      kept_cluster_ids: set(int)
    """
    df = pd.read_csv(csv_path)
    for col in ["cluster", "theme"]:
        if col not in df.columns:
            raise RuntimeError(f"{csv_path} missing {col}")

    df["cluster"] = df["cluster"].astype(int)
    df["theme"] = df["theme"].fillna("").astype(str).str.strip()
    df = df[df["theme"] != ""].copy()
    df = df.sort_values("cluster")

    themes_list = [{"cluster": int(r["cluster"]), "theme": r["theme"]} for _, r in df.iterrows()]
    theme_lookup = {int(r["cluster"]): r["theme"] for _, r in df.iterrows()}
    kept_cluster_ids = set(theme_lookup.keys())
    return themes_list, theme_lookup, kept_cluster_ids

def load_ads(csv_path: str) -> pd.DataFrame:
    """
    Required columns: ad_archive_id, ad_creative_bodies, cluster (meta)
    Required columns: cid, text, cluster (bluesky)
    """
    df = pd.read_csv(csv_path)
    req = ["cid", "text", "cluster"] # bluesky
    # req = ["ad_archive_id", "ad_creative_bodies", "cluster"]  # meta
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path} missing {c}")
    # replace cid with ad_archive_id and text with ad_creative_bodies for meta
    df["cid"] = df["cid"].astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["cluster"] = df["cluster"].astype(int)
    return df


def build_batch_prompt(ads_batch: List[Dict[str, str]], themes_list: List[Dict[str, Any]]):
    """
    Ask the model to choose theme
    """
    theme_lines = []
    for idx, row in enumerate(themes_list, start=1):
        # Show only the label
        theme_lines.append(f"{idx}. {row['theme']}")

    ad_lines = []
    for idx, ad in enumerate(ads_batch, start=1):
        # text -> ad_creative_bodies for meta
        # cid -> ad_archive_id for meta
        text_one_line = re.sub(r"\s+", " ", ad["text"]).strip()
        ad_lines.append(f'{idx}. [cid={ad["cid"]}] {text_one_line}')

    k = len(ads_batch)
    prompt = (
        "Choose, for each ad, the single best-fitting theme from the numbered list below.\n"
        "RETURN FORMAT (MUST follow exactly):\n"
        f"- Output EXACTLY {k} lines.\n"
        "- On line i, output ONLY the NUMBER (no text) of the chosen theme for post i.\n"
        "No words, no commas, no explanations—just the number per line.\n\n"
        "Allowed themes:\n" + "\n".join(theme_lines) + "\n\n"
        "Posts:\n" + "\n".join(ad_lines) + "\n\n"
        "Response (numbers only, one per line):\n"
    )
    return prompt

def parse_llm_batch_response(
    raw: str,
    ads_batch: List[Dict[str, str]],
    reverse_lookup: Dict[str, int],
    themes_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Prefer numeric indices (1..N) mapping to themes_list.
    Fallback: try to map a returned text label to a theme via reverse_lookup.
    """
    def _clean_line(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r'^\s*(\[\d+\]|[-*]|\d+[\)\].:])\s*', '', s)  # trim bullets like "1) " or "- "
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    # idx -> cluster & idx -> theme
    idx_to_cluster = {str(i): themes_list[i-1]["cluster"] for i in range(1, len(themes_list)+1)}
    idx_to_theme   = {str(i): themes_list[i-1]["theme"]   for i in range(1, len(themes_list)+1)}

    lines = [ln for ln in (raw or "").splitlines() if ln.strip()]
    # Allow comma-separated single line fallback
    if len(lines) == 1 and ("," in lines[0]):
        parts = [p.strip() for p in lines[0].split(",") if p.strip()]
        if len(parts) > 1:
            lines = parts

    # pad/truncate to #ads
    if len(lines) < len(ads_batch):
        lines += [None] * (len(ads_batch) - len(lines))
    elif len(lines) > len(ads_batch):
        lines = lines[:len(ads_batch)]

    out: Dict[str, Dict[str, Any]] = {}

    for ad, raw_line in zip(ads_batch, lines):
        cleaned = _clean_line(raw_line or "")

        # 1) Numeric index path
        cid = None
        chosen_theme_text = None
        m = re.match(r'^\s*(\d+)\s*$', cleaned) or re.match(r'^\s*(\d+)', cleaned)
        if m:
            idx = m.group(1)
            if idx in idx_to_cluster:
                cid = int(idx_to_cluster[idx])
                chosen_theme_text = idx_to_theme[idx]

        # 2) Fallback: try text match to theme label
        if cid is None and cleaned:
            cleaned2 = strip_md_quotes(cleaned)
            cid_lookup = (reverse_lookup.get(normalize_theme(cleaned2))
                          or reverse_lookup.get(normalize_theme_loose(cleaned2)))
            if cid_lookup is not None:
                cid = int(cid_lookup)
                chosen_theme_text = cleaned2

        if cid is not None:
            # replace cid with ad_archive_id for meta
            out[ad["cid"]] = {
                "chosen_cluster": cid,
                "chosen_theme": chosen_theme_text,
                "llm_label": raw_line,
                "needs_retry": False,
                "match_status": "matched",
                "cleaned_attempt": cleaned,
            }
        else:
            # replace cid with ad_archive_id for meta
            out[ad["cid"]] = {
                "chosen_cluster": None,
                "chosen_theme": None,
                "llm_label": raw_line,
                "needs_retry": True,
                "match_status": "non_matching" if raw_line else "no_output",
                "cleaned_attempt": cleaned or None,
            }

    return out

def print_batch_diagnostics(batch_idx: int,
                            retry_num: int,
                            raw: str,
                            ads_batch: List[Dict[str, str]],
                            batch_map: Dict[str, Dict[str, Any]],
                            theme_lookup: Dict[int, str]):
    if not DEBUG_VERBOSE:
        return

    # replace cid with ad_archive_id for meta
    failed = [(ad["cid"], batch_map.get(ad["cid"], {})) for ad in ads_batch
              if not batch_map.get(ad["cid"], {}).get("chosen_cluster")]
    # replace cid with ad_archive_id for meta
    matched = [(ad["cid"], batch_map.get(ad["cid"], {})) for ad in ads_batch
               if batch_map.get(ad["cid"], {}).get("chosen_cluster")]

    # Show a couple matched examples
    for i, (ad_id, rec) in enumerate(matched[:3], start=1):
        cid = rec.get("chosen_cluster")
        theme = theme_lookup.get(cid)
        print(f"[OK] {ad_id} -> {theme}")

    # Show failures with cleaned label + short ad snippet
    for i, (ad_id, rec) in enumerate(failed, start=1):
        cleaned = rec.get("cleaned_attempt")
        raw_label = rec.get("llm_label")
        ad_text = next((a["text"] for a in ads_batch if a["cid"] == ad_id), "")
        snippet = shorten(ad_text.replace("\n", " "), width=120, placeholder="…")
        print(f"[FAIL] cid={ad_id} cleaned='{cleaned}' raw='{raw_label}'  text='{snippet}'")

def call_llm(messages_or_text,
             max_retries=5,
             base_sleep=1.0,
             max_sleep=10.0):
    if API_KEY in (None, "", "REPLACE_ME"):
        raise RuntimeError("MISTRAL_API_KEY not set; export it before running.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if isinstance(messages_or_text, str):
        payload_messages = [{"role": "user", "content": messages_or_text}]
    else:
        payload_messages = messages_or_text

    base_payload = {
        "model": MODEL_ID,
        "messages": payload_messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.post(API_URL, json=base_payload, headers=headers, timeout=120)
            if r.status_code == 429:
                try:
                    body = r.text
                except Exception:
                    body = "<no-body>"
                print(f"[HTTP 429] {len(body)} bytes: {body[:500]}")
                raise RuntimeError("RATE_LIMITED_429")

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            resp = e.response
            try:
                body = resp.text
            except Exception:
                body = ""
            last_err = f"{type(e).__name__} {resp.status_code}: {body[:1000]}"
            print(f"[HTTP ERROR] status={resp.status_code} body_len={len(body)}\n{body[:1000]}")
            if 400 <= resp.status_code < 500:
                if resp.status_code == 429:
                    raise RuntimeError("RATE_LIMITED_429")
                break
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s payload_tokens~={MAX_TOKENS}")
                time.sleep(jitter)
                continue

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[HTTP EXC] {last_err}")
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s")
                time.sleep(jitter)
                continue

        except Exception as e:
            last_err = f"Unexpected: {e}"
            print(f"[HTTP UNEXPECTED] {last_err}")
            if "RATE_LIMITED_429" in str(e):
                raise
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s")
                time.sleep(jitter)
                continue

    raise RuntimeError(last_err or "unknown error")


def main():
    _ensure_debug_dir()

    # Load themes (CSV) and ads/posts
    themes_list, theme_lookup, kept_cluster_ids = load_cluster_themes(CLUSTER_THEMES_CSV)
    print(f"[LOAD] {len(themes_list)} clusters with themes")

    # Reverse lookup keyed by THEME ONLY (used in fallback matching)
    reverse_lookup: Dict[str, int] = {}
    for item in themes_list:
        cid = int(item["cluster"])
        thm = strip_md_quotes(str(item["theme"]))
        reverse_lookup[normalize_theme(thm)] = cid
        reverse_lookup[normalize_theme_loose(thm)] = cid

    df = load_ads(ADS_WITH_CLUSTERS_PATH)
    print(f"[LOAD] {len(df)} posts")

    df["needs_reassign"] = True
    df_unassigned = df.copy()
    print(f"[INFO] {len(df_unassigned)} ads/posts need reassignment")

    if MAX_ADS_TO_CLASSIFY is not None:
        df_unassigned = df_unassigned.head(MAX_ADS_TO_CLASSIFY).copy()
        print(f"[INFO] limiting to first {len(df_unassigned)} ads for classification")

    llm_assignments: Dict[str, Dict[str, Any]] = {}

    rows = list(df_unassigned.itertuples(index=False))
    total = len(rows)

    PER_BATCH_MAX_RETRIES = 5
    hit_rate_limit = False

    for start in range(0, total, BATCH_SIZE):
        if hit_rate_limit:
            break

        batch_rows = rows[start:start + BATCH_SIZE]
        # ad_archive_id, ad_creative_bodies for meta
        ads_batch = [{"cid": str(r.cid), "text": str(r.text)} for r in batch_rows]
        #ads_batch = [{"ad_archive_id": str(r.cid), "ad_creative_bodies": str(r.text)} for r in batch_rows]  # meta
        batch_idx = start // BATCH_SIZE
        print(f"[BATCH] sending {len(ads_batch)} posts/ads ({start}..{start+len(ads_batch)-1}) to LLM")
        prompt = build_batch_prompt(ads_batch, themes_list)
        _dump_text("prompt", batch_idx, 0, prompt)

        try:
            raw = call_llm(prompt)
            _dump_text("response", batch_idx, 0, raw)
            batch_map = parse_llm_batch_response(raw, ads_batch, reverse_lookup, themes_list)
            llm_assignments.update(batch_map)
            print(f"[BATCH] got {len(batch_map)} assignments back")
            print_batch_diagnostics(batch_idx, 0, raw, ads_batch, batch_map, theme_lookup)

        except RuntimeError as e:
            if str(e) == "RATE_LIMITED_429":
                print("[BATCH] hit 429 rate limit / token budget exhausted. Stopping early and saving progress.")
                hit_rate_limit = True
            else:
                print(f"[BATCH] fatal error for ads/posts {start}-{start+len(ads_batch)-1}: {e}")
            break
        except Exception as e:
            print(f"[BATCH] error for ads/posts {start}-{start+len(ads_batch)-1}: {e}")
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        # Retry only the failures within this batch 

        # replace cid with ad_archive_id for meta
        # replace text with ad_creative_bodies for meta
        id_to_text = {a["cid"]: a["text"] for a in ads_batch}
        failed_ids = [ad_id for ad_id, rec in batch_map.items() if rec.get("chosen_cluster") is None]

        retry_num = 0
        while failed_ids and retry_num < PER_BATCH_MAX_RETRIES and not hit_rate_limit:
            retry_num += 1
            print(f"[BATCH] retry #{retry_num}: reattempting {len(failed_ids)} failed posts")

            # replace cid with ad_archive_id for meta
            # replace text with ad_creative_bodies for meta
            retry_ads_batch = [{"cid": ad_id, "text": id_to_text[ad_id]} for ad_id in failed_ids]
            retry_prompt = build_batch_prompt(retry_ads_batch, themes_list)
            _dump_text("prompt", batch_idx, retry_num, retry_prompt)

            try:
                raw_retry = call_llm(retry_prompt)
                _dump_text("response", batch_idx, retry_num, raw_retry)
                retry_map = parse_llm_batch_response(raw_retry, retry_ads_batch, reverse_lookup, themes_list)

                # Update master map; keep latest attempt
                llm_assignments.update(retry_map)

                # Diagnostics for retry
                print_batch_diagnostics(batch_idx, retry_num, raw_retry, retry_ads_batch, retry_map, theme_lookup)

                # Recompute failed_ids
                failed_ids = [ad_id for ad_id, rec in retry_map.items() if rec.get("chosen_cluster") is None]

            except RuntimeError as e:
                if str(e) == "RATE_LIMITED_429":
                    print("[BATCH] retry hit 429; stopping.")
                    hit_rate_limit = True
                    break
                else:
                    print(f"[BATCH] retry fatal error: {e}")
            except Exception as e:
                print(f"[BATCH] retry parse error: {e}")

            time.sleep(min(10.0, SLEEP_BETWEEN_BATCHES * (2 ** (retry_num - 1))))

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # build final output
    final_rows = []
    attempted_ids = set(llm_assignments.keys())

    for r in df.itertuples(index=False):
        ad_id        = str(r.cid)
        ad_text      = str(r.text)
        orig_cluster = int(r.cluster)
        needs        = ad_id in attempted_ids  

        if not needs:
            final_cluster = orig_cluster
            assign_reason = "kept_original_cluster"
            llm_label     = None
            match_status  = None
            final_theme   = theme_lookup.get(final_cluster)
        else:
            rec = llm_assignments.get(ad_id, {})
            if rec and rec.get("chosen_cluster") is not None:
                final_cluster = int(rec["chosen_cluster"])
                assign_reason = "llm_theme_reassign"
                llm_label     = rec.get("llm_label")
                match_status  = rec.get("match_status")
                final_theme   = theme_lookup.get(final_cluster)
            else:
                final_cluster = None
                assign_reason = "llm_missing"
                llm_label     = rec.get("llm_label")
                match_status  = rec.get("match_status")
                final_theme   = None

        final_rows.append({
            "cid": ad_id, # ad_archive_id for meta
            "text": ad_text, # ad_creative_bodies for meta
            "original_cluster": orig_cluster,
            "final_cluster": final_cluster,
            "assign_reason": assign_reason,
            "needs_reassign": needs,   
            "llm_label": llm_label,
            "match_status": match_status,
            "final_theme": final_theme,
        })

    print("[WRITE] writing outputs")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(OUT_CSV, index=False)

    reassigned_ok = sum(1 for row in final_rows if row["assign_reason"] == "llm_theme_reassign")
    kept_orig     = sum(1 for row in final_rows if row["assign_reason"] == "kept_original_cluster")
    missing       = sum(1 for row in final_rows if row["assign_reason"] == "llm_missing")

    print("=== SUMMARY ===")
    print(f"Total posts/ads: {len(df)}")
    print(f"LLM reassigned using themes: {reassigned_ok}")
    print(f"Kept original cluster: {kept_orig}")
    print(f"Missing / no assignment from LLM: {missing}")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
