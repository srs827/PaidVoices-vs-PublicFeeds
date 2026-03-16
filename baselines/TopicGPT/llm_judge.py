import pandas as pd
import os
import time
from openai import OpenAI

# replace input_csv with actual sample path (meta or bluesky)
INPUT_CSV = "data/topicGPT-main/topic_labels/meta.csv"
OUTPUT_CSV = "llm_judge_meta.csv"
BASE_URL = "https://api.together.xyz/v1"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
SLEEP = 0.1

PRINT_PROMPTS = False
PROMPT_PREVIEW_CHARS = 4000


def is_empty(val) -> bool:
    """Treat common 'empty' / placeholder values as empty."""
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    low = s.lower()
    return low in {"nan", "none", "null", "n/a", "na", "[]", "{}", "()", "empty"}


def yesno(reply: str) -> str:
    if not reply:
        return "No"
    return "Yes" if "yes" in reply.strip().lower() else "No"


def _preview(s: str, limit: int | None) -> str:
    if s is None:
        return ""
    if limit is None or len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated {len(s) - limit} chars]"


def ask(client: OpenAI, system_prompt: str, user_prompt: str, *, tag: str = "") -> str:
    if PRINT_PROMPTS:
        header = f"\n===== PROMPT {f'[{tag}] ' if tag else ''}====="
        print(header)
        print("SYSTEM:\n" + system_prompt)
        print("\nUSER:\n" + _preview(user_prompt, PROMPT_PREVIEW_CHARS))
        print("===== END PROMPT =====\n")

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    # replace TEXT_COL with "ad_creative_bodies" for meta
    TEXT_COL = "text" if "text" in df.columns else "ad_text"

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing TOGETHER_API_KEY. Set it with: export TOGETHER_API_KEY='...'"
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    topicgpt_yesno = []


    TopicGPT_SYSTEM = (
        "Evaluate whether the given topic(s) fit the text.\n"
        "If the topic(s) are missing, irrelevant, or they do not reflect the main content, output 'No'.\n"
        "If all topic(s) do reflect the text, output 'Yes'. Output only Yes or No."
    )
    

    for i, row in df.iterrows():
        post_id = row.get("cid", str(i))
        text = row.get(TEXT_COL, "")

        # If the TEXT itself is empty, force all outputs to No (and skip API calls)
        if is_empty(text):
            topicgpt_yesno.append("No")
            print(
                f"[{i+1}/{len(df)}] {post_id} -> "
                f"TopicGPT: No (empty text)"
            )
            continue
        # 1) TopicGPT keywords — force No if empty topics
        topicgpt_topic = row.get("topics", "")
        if is_empty(topicgpt_topic):
            topicgpt_yesno.append("No")
        else:
            up = f"text:\n{text}\n\ntopic_gpt_topics: {topicgpt_topic}"
            topicgpt_yesno.append(yesno(ask(client, TopicGPT_SYSTEM, up, tag=f"{i+1}/TopicGPT")))
            time.sleep(SLEEP)

    df["topicgpt_correct"] = topicgpt_yesno

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAll evaluations complete. Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
