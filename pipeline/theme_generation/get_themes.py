# save as: run_cluster_themes.py
import os, json, time
import httpx
import pandas as pd

# replace with proper paths for meta/bluesky with the merged summaries
IN_JSONL  = "merged__merged_summaries.jsonl"
OUT_JSONL = "cluster_themes.jsonl"

MODEL_ID = "mistral-large-2407" 
BASE_URL     = "https://api.mistral.ai/v1"
API_KEY      = os.getenv("MISTRAL_API_KEY")
API_URL  = f"{BASE_URL}/chat/completions"

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 64   
ENV_LIMIT = 0     # 0 = no limit


def make_theme_prompt(cluster_id: int, summary_text: str) -> str:
    prompt = (
        "Produce a short theme label in 1–3 words for the given summary. "
        "This theme label should capture the central idea, stance, and/or topic of the text.\n\n"
        "Examples:\n\n"
        "'Against climate policy'\n"
        "'Alternative energy'\n"
        "'Oil lobby messaging'\n\n"
        f"Summary to label: ** {summary_text.strip()} **\n\n"
        "Theme label (1–3 words):"
    )
    return prompt


def call_model(prompt, retries=3, sleep=1.2):
    last_err = None
    headers = {"Authorization": f"Bearer {API_KEY}"}

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


def load_cluster_summaries(path_jsonl):
    """
    Read cluster_summaries_large.jsonl produced by the first script.
    We only keep rows that have a non-empty 'summary'.
    Returns a list of dicts like:
        {"cluster": 42, "summary": "..."}
    """
    rows = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj.get("summary"), str) and obj["summary"].strip():
                rows.append({
                    "cluster": obj.get("cluster"),
                    "summary": obj["summary"].strip(),
                })
    return rows


def main():
    # 1. Load summaries from previous step
    base_rows = load_cluster_summaries(IN_JSONL)
    if not base_rows:
        print("No valid summaries found in cluster_summaries_large.jsonl")
        return

    work = base_rows[:ENV_LIMIT] if ENV_LIMIT > 0 else base_rows
    if ENV_LIMIT > 0:
        print(f"Limiting to first {ENV_LIMIT} clusters for theme labeling")

    results = []
    for idx, row in enumerate(work, 1):
        cid  = row["cluster"]
        summ = row["summary"]

        prompt = make_theme_prompt(cid, summ)

        try:
            theme_text = call_model(prompt)

            obj = {
                "cluster": cid,
                "summary": summ,
                "theme": theme_text.strip(),   # raw model output
                "model_used": MODEL_ID,
            }

        except Exception as e:
            obj = {
                "cluster": cid,
                "summary": summ,
                "theme": None,
                "error": f"request_error: {e}",
                "model_used": MODEL_ID,
            }

        results.append(obj)
        time.sleep(0.4)
        if idx % 5 == 0:
            print(f"Labeled {idx}/{len(work)} clusters...")

    # 2. Write JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(results)} themes to {OUT_JSONL}")

    # 3. Also write CSV
    try:
        out_rows = []
        for r in results:
            out_rows.append({
                "cluster": r.get("cluster"),
                "theme": r.get("theme"),
                "summary": r.get("summary"),
                "error": r.get("error"),
            })
        pd.DataFrame(out_rows).to_csv(
            OUT_JSONL.replace(".jsonl", ".csv"),
            index=False
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
