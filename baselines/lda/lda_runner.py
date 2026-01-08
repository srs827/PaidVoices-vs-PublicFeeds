import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import re 
import openpyxl


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def normalize_text(s: str) -> str:
    """Lowercase, strip URLs/punct, squeeze spaces."""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)            # URLs
    s = re.sub(r"[^a-z\s]", " ", s)                  # keep letters/spaces
    s = re.sub(r"\s+", " ", s).strip()               # strip whitespace to single space
    return s




def combine(row):
    parts = [str(row.get(c, "")) for c in text_cols]
    return normalize_text(" | ".join([p for p in parts if p]))


def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# insert file here
df = pd.read_csv("preprocessing/combined_deduped_sbert080.csv") # replace with proper path meta/bluesky
df2 = df.copy()

text_cols = ["text"]  # or ad_creative_bodies for meta




from gensim import corpora, models

docs = [preprocess(txt) for txt in df2["text"].astype(str).tolist()]

# remove empty 
docs = [d for d in docs if d]

# build dictionary
dictionary = corpora.Dictionary(docs)

#   - keep tokens that appear in at least 5 docs
#   - drop tokens that appear in more than 50% of docs
dictionary.filter_extremes(no_below=5, no_above=0.5)

# 3) Build corpus (BoW for each doc)
corpus = [dictionary.doc2bow(doc) for doc in docs]


# 4) Train LDA
num_topics = 100
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=15,
    random_state=42
)

# Inspect a few topics
for i in range(num_topics):
    print(f"Topic {i}: ", lda_model.print_topic(i, topn=10))

all_topics = lda_model.get_topics()

rows = []
# topn: number of words per topic
for topic_id, terms in lda_model.show_topics(num_topics=-1, num_words=10, formatted=False):
    # terms is a list of (word, weight)
    top_words = [word for word, _ in terms]
    rows.append({"Topic": topic_id, "Keywords": ", ".join(top_words)})

df2 = pd.DataFrame(rows)

# build df
topics_df = pd.DataFrame(rows)
topics_df.to_csv("topics_keywords_lda.csv", index=False)



# 0) Build a lookup of topic_id -> "kw1, kw2, ..." (top n keywords per topic)
TOPN_WORDS = 5
topic_kw = {}
for topic_id, terms in lda_model.show_topics(num_topics=-1, num_words=TOPN_WORDS, formatted=False):
    topic_kw[topic_id] = ", ".join([w for (w, _p) in terms])

docs_aligned = [preprocess(txt) for txt in df["text"].astype(str).tolist()]
corpus_aligned = [dictionary.doc2bow(doc) for doc in docs_aligned]

# for each document, get its topic distribution and take the dominant topic.
dominant_topic = []
dominant_prob  = []

for bow in corpus_aligned:
    dist = lda_model.get_document_topics(bow, minimum_probability=0)  # list[(topic_id, prob)]
    if not dist:  
        dominant_topic.append(-1)
        dominant_prob.append(0.0)
        continue
    # pick argmax
    best_tid, best_p = max(dist, key=lambda tp: tp[1])
    dominant_topic.append(best_tid)
    dominant_prob.append(best_p)

dominant_keywords = [topic_kw.get(tid, "") if tid != -1 else "" for tid in dominant_topic]


df_out = df.copy()
df_out["lda_topic_id"]  = dominant_topic
df_out["lda_topic_prob"] = dominant_prob
df_out["lda_topic_keywords"] = dominant_keywords

cols_to_keep = []
for c in ["cid", "text", "lda_topic_id", "lda_topic_prob", "lda_topic_keywords"]:
    if c in df_out.columns:
        cols_to_keep.append(c)
if cols_to_keep:
    df_out_view = df_out[cols_to_keep]
else:
    df_out_view = df_out  

df_out_view.to_csv("posts_with_lda_topics.csv", index=False) # "ads_with_lda_topics.csv"

topics_rows = [{"Topic": tid, "Keywords": kw} for tid, kw in topic_kw.items()]
topics_df = pd.DataFrame(topics_rows).sort_values("Topic").reset_index(drop=True)
topics_df.to_csv("topics_keywords_lda.csv", index=False)
