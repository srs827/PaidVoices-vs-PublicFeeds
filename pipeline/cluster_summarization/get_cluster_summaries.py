import os, json, time, re, argparse, traceback
import pandas as pd
import httpx
from openai import OpenAI
import requests
import numpy as np

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

# replace with proper paths for meta/bluesky
CSV_PATH = "clustering_code/topk_posts_by_cluster.csv"
COH_JSONL = "coherency_check/coherency_results.jsonl"
OUT_JSONL = "cluster_summaries.jsonl"

MODEL_ID = "mistral-large-2407" 
BASE_URL     = "https://api.mistral.ai/v1"
API_KEY      = os.getenv("MISTRAL_API_KEY")
API_URL  = f"{BASE_URL}/chat/completions"

# settings 
TOPK = 5  # how many ads/posts per cluster to summarize
MAX_AD_CHARS = 1000
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 1024
ENV_LIMIT = 0 # no limit

COHERENT_PAT   = re.compile(r"the cluster is\s*\*?coherent\*?", re.I)
INCOHERENT_PAT = re.compile(r"the cluster is\s*\*?incoherent\*?", re.I)

def is_coherent_from_response(s: str) -> bool | None:
    """
    Returns True/False if the line clearly states coherent/incoherent.
    Returns None if it can't tell (should be rare with your enforced format).
    """
    if not s:
        return None
    if COHERENT_PAT.search(s):
        return True
    if INCOHERENT_PAT.search(s):
        return False
    return None


def truncate(s: str, max_chars=400) -> str:
    s = (s or "").strip()
    return s[:max_chars] + (" …" if len(s) > max_chars else "")

def extract_first_json_relaxed(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON start")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    # If JSON ended prematurely, auto-close braces
    return s[start:] + ("}" * max(depth, 0))

def call_openai_with_retry(messages, retries=3, sleep=1.2):
    last_err = None
    headers = {"Authorization": f"Bearer {API_KEY}"}

    payload_base = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    for attempt in range(1, retries + 1):
        try:
            # 1) Try with response_format=json_object
            payload = {
                **payload_base,
                "response_format": {"type": "json_object"}
            }
            r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code in (400, 404) and "response_format" in r.text.lower():
                # 2) Retry without response_format 
                payload = dict(payload_base)
                r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(sleep * attempt)

    raise RuntimeError(last_err or "unknown error")

def make_summary_prompt(cluster_id, items):
    """
    Build a single text prompt for a cluster.
    items = [
        {"ad_archive_id": "...", "text": "..."},
        ...
    ]
    """

    # replace "posts" with "ads" for meta
    header = (
        "Summarize the following posts by writing 1–3 sentences "
        "(<= 100 words) capturing the common message and intent:\n\n"
    )

    lines = []
    for i, it in enumerate(items, 1):
        # replace "text" with "ad_creative_bodies" for meta
        # replace cid with ad_archive_id for meta
        adid = it.get("text", "")
        txt = truncate(it.get("text", ""), MAX_AD_CHARS)
        lines.append(f"{i}. ** [cid={adid}] {txt} **")

    ending = "\n\nYour summary:\n"

    return header + "\n".join(lines) + ending

def call_model(prompt, retries=3, sleep=1.2):
    last_err = None
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json",
    }

    payload_base = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(API_URL, json=payload_base, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(sleep * attempt)

    raise RuntimeError(last_err or "unknown error")


def main():
    # load coherency results
    coh = []
    with open(COH_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                coh.append({
                    "cluster": int(obj.get("cluster")),
                    "response": obj.get("response", "") or "",
                })
            except Exception:
                continue
    if not coh:
        print("No coherency results found; nothing to summarize.")
        return
    coh_df = pd.DataFrame(coh)
    coh_df["coherent"] = coh_df["response"].map(is_coherent_from_response)
    coh_df = coh_df[coh_df["coherent"].isin([True, False])].copy()
    if coh_df.empty:
        print("No parseable coherency verdicts; nothing to summarize.")
        return

    # filter for coherent clusters only
    coherent_ids = sorted(coh_df.loc[coh_df["coherent"] == True, "cluster"].unique().astype(int))
    if not coherent_ids:
        print("No coherent clusters found; nothing to summarize.")
        return
    
    # load ads/posts and select top-k per cluster
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["cluster", "topk_rank"])
    groups = df.groupby("cluster")
    work = []
    for cid in coherent_ids:
        if cid not in groups.groups:
            continue
        g = groups.get_group(cid).head(TOPK)
        # replace "text" with "ad_creative_bodies" for meta
        items = [{"text": str(r["text"]), "text": str(r["text"])}
                 for _, r in g.iterrows()]
        work.append((cid, items))

    if ENV_LIMIT > 0: 
        work = work[:ENV_LIMIT]
        print(f"Limiting to first {ENV_LIMIT} coherent clusters")
        try:
            pass
        except: 
            pass
    results = []
    for idx, (cid, items) in enumerate(work, 1):
        #msgs = make_summary_messages(cid, items)
        prompt = make_summary_prompt(cid, items)
        try:
            txt = call_model(prompt)
            obj = {
                "cluster": int(cid),
                "summary": txt.strip(),
                "model_used": MODEL_ID
            }
            
        except Exception as e:
            obj = {
                "cluster": int(cid),
                "error": f"request_or_parse_error: {e}"
            }

        results.append(obj)
        time.sleep(0.4)
        if idx % 5 == 0:
            print(f"Processed {idx}/{len(work)} summaries...")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    print(f"Done. Wrote {len(results)} cluster summaries to {OUT_JSONL}")
    try:
        summary_rows = []
        for r in results:
            if "error" in r:
                summary_rows.append({
                    "cluster": int(r.get("cluster")),
                    "summary": None,
                    "error": r.get("error")
                })
            else:
                summary_rows.append({
                    "cluster": int(r.get("cluster")),
                    "summary": r.get("summary"),
                    "error": r.get("error")
                })
        pd.DataFrame(summary_rows).to_csv(OUT_JSONL.replace(".jsonl", ".csv"), index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
