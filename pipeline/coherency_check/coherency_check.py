# save as: run_cluster_coherency.py
import os, json, time, re, traceback, httpx
import pandas as pd
import argparse

# replace with proper path for meta/blusky from 
# corrresponding subfolder in clustering_code
CSV_PATH  = "clustering_code/topk_posts_by_cluster.csv"
OUT_PATH  = "coherency_results.jsonl"

MODEL_ID     = "mistral-large-2407"
BASE_URL     = "https://api.mistral.ai/v1"
API_KEY      = os.getenv("MISTRAL_API_KEY")
API_URL      = f"{BASE_URL}/chat/completions"

TEMPERATURE  = 0.2
TOP_P        = 0.9
MAX_TOKENS   = 512
ENV_LIMIT    = 0      # 0 = no limit
SLEEP_BETWEEN = 0.4   

ONE_LINE_OK = {
    "coherent":  "The cluster is *coherent* because the labels are consistent.",
    "incoherent": "The cluster is *incoherent* because the labels are not consistent.",
}

def extract_one_line_verdict(text: str) -> str:
    t = (text or "").strip()
    if re.search(r"\bcoherent\b", t, re.I) and re.search(r"\bconsistent\b", t, re.I):
        return ONE_LINE_OK["coherent"]
    if re.search(r"\bincoherent\b", t, re.I):
        return ONE_LINE_OK["incoherent"]
    first = t.splitlines()[0].strip()
    m = re.search(r"The cluster is\s*\*(coherent|incoherent)\*", first, re.I)
    if m:
        return ONE_LINE_OK[m.group(1).lower()]
    return ONE_LINE_OK["incoherent"]

def load_fewshot(path: str | None) -> str:
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # separator so the model knows examples are over:
    return "Examples:\n" + text + "\n\n### Prompt\n"

def truncate(s: str, max_chars=1800) -> str:
    """
    Basic truncation/cleanup for paragraphs.
    """
    s = (s or "").strip()
    if len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


def format_cluster_paragraphs(paragraphs):
    """
    paragraphs: list[str] length up to 5
        1: ** Paragraph 1 **
        2: ** Paragraph 2 **
        ...
    """
    lines = []
    for i, p in enumerate(paragraphs, start=1):
        clean_p = re.sub(r"\s+", " ", p).strip()
        clean_p = truncate(clean_p)
        lines.append(f"{i}: ** {clean_p} **")
    return "\n".join(lines)


def make_coherency_prompt(paragraphs, examples):
    """

    Prompt:
    Label each of the 5 paragraphs...
    ...
    Response:
    The cluster is <coherent/incoherent> because ...
    """
    instr = (
        "Label each of the 5 paragraphs below. If the labels are all the same, "
        "the cluster is *coherent*. If the labels differ, the cluster is *incoherent*.\n"
        "Return **exactly one line** and nothing else, in one of these two forms:\n"
        '- The cluster is *coherent* because the labels are consistent.\n'
        '- The cluster is *incoherent* because the labels are not consistent.\n'
        "Do not list per-paragraph labels. Do not add explanations.\n\n"
    )
    return examples + instr + format_cluster_paragraphs(paragraphs) + "\n\nResponse:"


def call_model(prompt, retries=4, sleep=0.8):
    last_err = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "stop": ["\n\n", "\n1.", "\nLabel:"],
    }
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code != 200:
                print(f"[HF Router] HTTP {r.status_code}: {r.text[:800]}")
                r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(sleep * attempt)
    raise RuntimeError(last_err or "unknown error")


def load_clusters(csv_path):
    """
    From topk_ads_by_cluster.csv (topk_posts_by_cluster.csv):
    - Sort by (cluster, topk_rank)
    - Take top 5 ads/posts per cluster
    - Return list[(cluster_id:int, paragraphs:list[str])]
    """
    df = pd.read_csv(csv_path)

    sub = (
        df.sort_values(["cluster", "topk_rank"])
          .groupby("cluster")
          .head(5)
          .reset_index(drop=True)
    )

    clusters = []
    for cid, g in sub.groupby("cluster"):
        # replace "text" with "ad_creative_bodies" for meta 
        paras = g["text"].fillna("").astype(str).tolist()
        # pad/crop to length 5
        paras = (paras + [""] * 5)[:5]
        clusters.append((int(cid), paras))

    return clusters


def main():
    # pull clusters
    clusters = load_clusters(CSV_PATH)

    if ENV_LIMIT > 0:
        clusters = clusters[:ENV_LIMIT]
        print(f"Limiting to {ENV_LIMIT} clusters")

    examples = load_fewshot("coherency_check/coherency_prompting.txt")
    results = []

    for idx, (cid, paras) in enumerate(clusters, start=1):
        prompt = make_coherency_prompt(paras, examples)

        try:
            model_resp = call_model(prompt)
            one_line = extract_one_line_verdict(model_resp)
            obj = {
                "cluster": cid,
                "response": one_line,
                "raw_response": model_resp.strip(),
                "model_used": MODEL_ID,
            }
        except Exception as e:
            obj = {
                "cluster": cid,
                "response": None,
                "error": f"request_error: {e}",
                "model_used": MODEL_ID,
            }

        results.append(obj)

        time.sleep(SLEEP_BETWEEN)
        if idx % 5 == 0:
            print(f"Processed {idx}/{len(clusters)} clusters...")

    # write JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} cluster results to {OUT_PATH}")

    try:
        out_rows = []
        for r in results:
            out_rows.append({
                "cluster": r.get("cluster"),
                "response": r.get("response"),
                "error": r.get("error"),
            })
        pd.DataFrame(out_rows).to_csv(
            OUT_PATH.replace(".jsonl", ".csv"),
            index=False
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
