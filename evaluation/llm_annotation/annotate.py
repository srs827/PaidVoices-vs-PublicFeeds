import pandas as pd
import os
import time
from openai import OpenAI

# replace input_csv with actual sample path (meta or bluesky)
INPUT_CSV = "data/sample_500.csv"
OUTPUT_CSV = "llm_annotation_results.csv"
BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/qwen3-235b-a22b"
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

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing FIREWORKS_API_KEY. Set it with: export FIREWORKS_API_KEY='...'"
        )

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    lda_yesno = []
    bert_yesno = []
    llm_theme_yesno = []
    llm_summ_yesno = []

    LDA_SYSTEM = (
        "Evaluate whether the majority of the given keywords fit the text.\n"
        "If the keywords are missing, more than 2 are irrelevant, or they do not reflect the main content, output 'No'.\n"
        "If nearly all the keywords do reflect the text, output 'Yes'. Output only Yes or No."
    )
    BERT_SYSTEM = (
        "Evaluate whether the majority of the given keywords fit the text.\n"
        "If the keywords are missing, more than 2 are irrelevant, or they do not reflect the main content, output 'No'.\n"
        "If nearly all the keywords do reflect the text, output 'Yes'. Output only Yes or No."
    )
    LLM_THEME_SYSTEM = (
        "Evaluate whether the given theme fits the text.\n"
        "If the theme is missing, is highly irrelevant, or doesn't reflect the main content output 'No'.\n"
        "If the theme mostly reflects the text, output 'Yes'. Output only Yes or No."
    )
    LLM_SUMMARY_SYSTEM = (
        "Evaluate whether the given theme fits the text.\n"
        "If the theme is missing, is highly irrelevant, or doesn't reflect the main content output 'No'.\n"
        "If the theme mostly reflects the text, output 'Yes'. Output only Yes or No."
    )

    for i, row in df.iterrows():
        post_id = row.get("cid", str(i))
        text = row.get(TEXT_COL, "")

        # If the TEXT itself is empty, force all outputs to No (and skip API calls)
        if is_empty(text):
            lda_yesno.append("No")
            bert_yesno.append("No")
            llm_theme_yesno.append("No")
            llm_summ_yesno.append("No")
            print(
                f"[{i+1}/{len(df)}] {post_id} -> "
                f"LDA:No BERT:No LLM Theme:No LLM Summary:No (empty text)"
            )
            continue

        # 1) LDA keywords — force No if empty keywords
        lda_kw = row.get("lda_topic_keywords", "")
        if is_empty(lda_kw):
            lda_yesno.append("No")
        else:
            up = f"text:\n{text}\n\nlda_topic_keywords: {lda_kw}"
            lda_yesno.append(yesno(ask(client, LDA_SYSTEM, up, tag=f"{i+1}/LDA")))
            time.sleep(SLEEP)

        # 2) BERT keywords — force No if empty keywords
        bert_kw = row.get("topic_keywords", "")
        if is_empty(bert_kw):
            bert_yesno.append("No")
        else:
            up = f"text:\n{text}\n\nbert_topic_keywords: {bert_kw}"
            bert_yesno.append(yesno(ask(client, BERT_SYSTEM, up, tag=f"{i+1}/BERT")))
            time.sleep(SLEEP)

        # 3) Theme label — force No if empty theme
        theme = row.get("llm_theme_theme", "")
        if is_empty(theme):
            llm_theme_yesno.append("No")
        else:
            up = f"text:\n{text}\n\nllm_theme_theme: {theme}"
            llm_theme_yesno.append(
                yesno(ask(client, LLM_THEME_SYSTEM, up, tag=f"{i+1}/THEME"))
            )
            time.sleep(SLEEP)

        # 4) Summary label — force No if empty summary
        summ = row.get("llm_theme_summ", "")
        if is_empty(summ):
            llm_summ_yesno.append("No")
        else:
            up = f"text:\n{text}\n\nllm_theme_summ: {summ}"
            llm_summ_yesno.append(
                yesno(ask(client, LLM_SUMMARY_SYSTEM, up, tag=f"{i+1}/SUMMARY"))
            )
            time.sleep(SLEEP)

        print(
            f"[{i+1}/{len(df)}] {post_id} -> "
            f"LDA:{lda_yesno[-1]} BERT:{bert_yesno[-1]} "
            f"LLM Theme:{llm_theme_yesno[-1]} LLM Summary:{llm_summ_yesno[-1]}"
        )

    df["lda_correct"] = lda_yesno
    df["bert_correct"] = bert_yesno
    df["llm_theme_correct"] = llm_theme_yesno
    df["llm_summ_correct"] = llm_summ_yesno

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAll evaluations complete. Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()