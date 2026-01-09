from pathlib import Path
import json
import re
import pandas as pd
from typing import List, Tuple, Optional

INPUT_CSV  = "preprocessing/combined_deduped_sbert080.csv"      
INPUT_JSON = None  

# columns of text
CSV_TEXT_COLS = [
    "text",
    #"normalized_text",
    #"ad_creative_link_titles",
    #"ad_creative_link_descriptions",
    #"ad_creative_bodies",
]
# column to id each row
CSV_ID_COL = "cid"    # ad_archive_id for meta  

SAMPLE_N = None

OUTPUT_DIR = "bertopic_out"

# BERTopic hyperparams 

# single words and pairs of words
N_GRAM_RANGE = (1, 2)
# number of documents required to form a topic
MIN_TOPIC_SIZE = 20      
LOW_MEMORY = True           
LANGUAGE = "english"        

def _strip_ws(s: str) -> str:
    s = "" if s is None else str(s)
    # if there is any whitespace we replace it with a single space
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# combine all text columns from this row into one text column for this document
# return all ids from all documents + all texts from all documents as a tuple 
def build_docs_from_csv(path: str | Path,
                        text_cols: List[str],
                        id_col: Optional[str]) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, encoding="utf-8")
    ids, docs = [], []
    for idx, row in df.iterrows():
        parts = []
        for c in text_cols:
            if c in df.columns:
                v = _strip_ws(row.get(c, ""))
                if v:
                    parts.append(v)
        doc = _strip_ws("\n\n".join(parts))
        if not doc:
            continue
        rid = _strip_ws(str(row.get(id_col))) if (id_col and id_col in df.columns) else f"row_{idx}"
        ids.append(rid)
        docs.append(doc)
    return ids, docs

# TopicGPT needs the "text" and "id" columns in a jsonl file
def build_docs_from_jsonl(path: str | Path) -> Tuple[List[str], List[str]]:
    ids, docs = [], []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = str(rec.get("id") or f"doc_{i}")
            txt = _strip_ws(rec.get("text", ""))
            if txt:
                ids.append(rid)
                docs.append(txt)
    return ids, docs

def run_bertopic(ids: List[str], docs: List[str], outdir: Path):
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import plotly.io as pio
    import numpy as np
    import re

    outdir.mkdir(parents=True, exist_ok=True)

    # embeddings
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # vectorizer
    vectorizer_model = CountVectorizer(
        stop_words=LANGUAGE,
        ngram_range=N_GRAM_RANGE,
        min_df=2
    )
    # c-TF-IDF
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        min_topic_size=MIN_TOPIC_SIZE,
        low_memory=LOW_MEMORY,
        verbose=True,
        top_n_words=10,
    )

    # Fit
    topics, probs = topic_model.fit_transform(docs)

    # topic metadata
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(outdir / "topic_info.csv", index=False)

    # topic_id -> "w1, w2, ..."
    def topic_keywords(topic_id: int, topn: int = 10) -> str:
        if topic_id == -1:
            return ""
        words_scores = topic_model.get_topic(topic_id) or []
        return ", ".join([w for (w, _s) in words_scores[:topn]])

    # topic_id -> name
    name_map = {}
    if "Name" in topic_info.columns:
        name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    kw_map = {tid: topic_keywords(tid, topn=10) for tid in topic_info["Topic"].tolist()}

    # per-doc table
    # assigned topic is topics[i]; grab its probability if 'probs' is available
    def assigned_prob(i: int) -> Optional[float]:
        if probs is None:
            return None
        # probs is an array shape (n_docs, n_topics); retrieve prob for assigned topic
        # BERTopic aligns topic  with max-prob topic
        if isinstance(probs, list) and probs and isinstance(probs[0], (list, np.ndarray)):
            p = probs[i]
            t = topics[i]
        try:
            di = topic_model.get_document_info([docs[i]]).iloc[0]
            if "Probability" in di:
                return float(di["Probability"])
        except Exception:
            pass
        # fallback if probs is ndarray
        if isinstance(probs, np.ndarray):
            t = topics[i]
            if t == -1:
                return float(np.nan)
            return float(probs[i, t]) if probs.ndim == 2 and probs.shape[0] == len(docs) else float(np.nan)
        return None

    rows = []
    for i, (rid, txt, t) in enumerate(zip(ids, docs, topics)):
        rows.append({
            "id": rid,
            "text": txt,
            "topic": t,
            "topic_prob": assigned_prob(i),
            "topic_keywords": kw_map.get(t, ""),
            "topic_name": name_map.get(t, "") if name_map else "",
        })

    doc_df = pd.DataFrame(rows)
    doc_df.to_csv(outdir / "doc_topics.csv", index=False)

    topic_model.save(outdir / "bertopic_model")

    try:
        pio.renderers.default = "browser"
    except Exception:
        pass

    topic_model.visualize_topics().write_html(outdir / "topics_overview.html")
    topic_model.visualize_hierarchy().write_html(outdir / "topics_hierarchy.html")
    topic_model.visualize_barchart(top_n_topics=12).write_html(outdir / "topics_barchart.html")
    topic_model.visualize_documents(docs, topics).write_html(outdir / "documents_scatter.html")


    _illegal = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]')
    def _clean(x):
        s = "" if x is None else str(x)
        return _illegal.sub("", s)

    doc_xlsx = doc_df.copy()
    for c in doc_xlsx.select_dtypes(include=["object"]).columns:
        doc_xlsx[c] = doc_xlsx[c].map(_clean)
    try:
        doc_xlsx.to_excel(outdir / "doc_topics.xlsx", index=False)
        topic_info.to_excel(outdir / "topic_info.xlsx", index=False)
    except Exception as e:
        print(f"[WARN] Excel export skipped: {e}")

    # ------------- Summary -------------
    print(f"[BERTopic] topics (incl. -1 outliers): {len(topic_info)}")
    print(f"[BERTopic] saved:")
    print(f"  - {outdir/'topic_info.csv'} / topic_info.xlsx")
    print(f"  - {outdir/'doc_topics.csv'} / doc_topics.xlsx")
    print(f"  - {outdir/'topics_overview.html'}")
    print(f"  - {outdir/'topics_hierarchy.html'}")
    print(f"  - {outdir/'topics_barchart.html'}")
    print(f"  - {outdir/'documents_scatter.html'}")
    print(f"  - {outdir/'bertopic_model'}")

if __name__ == "__main__":
    outdir = Path(OUTPUT_DIR)

    if INPUT_JSON and not INPUT_CSV:
        ids, docs = build_docs_from_jsonl(INPUT_JSON)
    elif INPUT_CSV and not INPUT_JSON:
        ids, docs = build_docs_from_csv(INPUT_CSV, CSV_TEXT_COLS, CSV_ID_COL)
    else:
        raise SystemExit("Configure INPUT_CSV or INPUT_JSON")

    if SAMPLE_N is not None and len(docs) > SAMPLE_N:
        ids, docs = ids[:SAMPLE_N], docs[:SAMPLE_N]
        print(f"[INFO] Using a sample of {len(docs)} docs.")

    # guard against empty docs
    if not docs:
        raise SystemExit("No documents found after preprocessing.")

    run_bertopic(ids, docs, Path(OUTPUT_DIR))
