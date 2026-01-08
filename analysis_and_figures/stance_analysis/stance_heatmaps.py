import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# change path based on bluesky/meta stance labels in stance_labels
CSV_PATH = "bluesky_stance.csv"  

df = pd.read_csv(CSV_PATH, encoding="latin1")

# for each row and stance we compute Pearson correlation 
def compute_theme_stance_corr(
    df,
    theme_col,
    stance_col="Stance",
    stance_order=("Pro-Climate", "Pro-Energy", "Neutral"),
    top_k=10,
):
 

    sub = df[[theme_col, stance_col]].dropna().copy()

    sub[theme_col] = sub[theme_col].astype(str)

    sub[stance_col] = sub[stance_col].astype(str)

    themes = sorted(sub[theme_col].unique())
    corr_matrix = np.zeros((len(themes), len(stance_order)))

    for i, t in enumerate(themes):
        # one-hot encode theme and stance
        theme_mask = (sub[theme_col] == t).astype(float).to_numpy()
        for j, s in enumerate(stance_order):
            stance_mask = (sub[stance_col] == s).astype(float).to_numpy()
            if theme_mask.std() == 0 or stance_mask.std() == 0:
                # undefined
                corr = 0.0
            else:
                # get pearson correlation of this theme with this stance
                corr = np.corrcoef(theme_mask, stance_mask)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            corr_matrix[i, j] = corr

    corr_df = pd.DataFrame(corr_matrix, index=themes, columns=stance_order)

    # rank by maximum absolute correlation
    max_abs = corr_df.abs().max(axis=1)
    corr_df["__max_abs_corr__"] = max_abs
    corr_df = corr_df.sort_values("__max_abs_corr__", ascending=False).drop(
        columns="__max_abs_corr__"
    )

    # keep only top_k
    corr_df = corr_df.head(top_k)

    return corr_df


stance_order = ("Pro-Climate", "Pro-Energy", "Neutral")
TOP_K = 10  

# adjust names based on meta/bluesky corresponding columns
matrices = [
    ("llm_theme_theme", "Ours: Theme-Based Assignment"),
    ("llm_theme_summ",   "Ours: Summary-Based Assignment"),
    ("bert_topic_name",              "Baseline: BERTopic topic"),
    ("lda_topic_id",               "Baseline: LDA topic"),
]

corr_results = []
for col, title in matrices:
    corr_df = compute_theme_stance_corr(
        df,
        theme_col=col,
        stance_col="Stance",
        stance_order=stance_order,
        top_k=TOP_K,
    )
    corr_results.append((corr_df, title))


global_vmax = max(abs(corr_df.values).max() for corr_df, _ in corr_results)
global_vmax = max(global_vmax, 0.1)
global_vmin = -global_vmax


sns.set_theme(style="white")

def wrap_labels(labels, width=30):
    """Wrap long y-axis labels so they don’t overlap."""
    return [textwrap.fill(str(lab), width=width) for lab in labels]

for corr_df, title in corr_results:
    corr_df_plot = corr_df.copy()


    raw_index = corr_df_plot.index

    # For LDA topics force integer labels 
    if "lda" in title.lower():  
        base_labels = []
        for lab in raw_index:
            try:
                f = float(lab)
                if f.is_integer():
                    base_labels.append(str(int(f)))
                else:
                    base_labels.append(str(f))
            except ValueError:
                base_labels.append(str(lab))
    else:
        # for other plots convert to string
        base_labels = [str(l) for l in raw_index]

    wrapped_index = wrap_labels(base_labels, width=30)
    corr_df_plot.index = wrapped_index

    fig, ax = plt.subplots(figsize=(6, 10))  

    hm = sns.heatmap(
        corr_df_plot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=global_vmin,
        vmax=global_vmax,
        linewidths=0.7,    
        linecolor="black",
        cbar=True,
        cbar_kws={"shrink": 0.7},
        square=False,         
    )

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Stance", fontsize=12)
    ax.set_ylabel("Theme / Topic", fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    safe_title = title.lower().replace(" ", "_").replace(":", "")
    plt.savefig(f"stance_heatmap_{safe_title}.pdf", bbox_inches="tight")
    plt.show()
