# Paid Voices versus Public Feeds: How Climate Narratives Diverge Online

This repository contains code and data for the paper titled "[Paid Voices vs. Public Feeds: Interpretable Cross-Platform Theme
Modeling of Climate Discourse](https://arxiv.org/pdf/2601.13317)".

All code from the theme generation pipeline under "pipeline", which can be modified to work with any target dataset. 
We include all results for our Meta and Bluesky climate-related datasets, with raw Meta and Bluesky data available through the following link: https://osf.io/csp3x/overview?view_only=85708d98978b4c7094e47f8604dd5e1a

The repository also contains all analysis-related code under "analysis_and_figures", such as:
  - Comparison of common themes between Meta and Bluesky
  - Event analysis code and resulting figures
  - Stance analysis code and resulting figures

We provide the evaluation code under "evaluation", where we include the code to provide LLM-judge annotation of the obtained results. 

We provide our full results and final human/llm annotations under "results". 

Each section is broken down as follows:

## Pipeline Code: Details the whole process from initial dataset -> assigned themes 

### Preprocessing: Removes duplicates/similar texts using SentenceBERT
  - preprocessing.py: script to deduplicate target csv
  - bluesky_preprocessing: contains deduplicated bluesky data + embeddings
  - meta_preprocessing: contains deduplicated meta data + embeddings

### Clustering Code: Clusters texts using HDBSCSN
  - get_clusters.py: clustering code
  - bluesky_clusters (meta_clusters): contains 3 files with clustering info for Bluesky (Meta)
    - cluster_summary.csv: contains list of clusters and number of posts (ads) in that cluster
    - posts_with_clusters.csv (ads_with_clusters.csv): list of posts (ads) with their assigned clusters
    - topk_posts_by_cluster.csv: contains top k posts (ads) by probability for each cluster 

### Coherency Checking: Checks coherency of each cluster using LLM (Mistral Large) 
  - Uses the top k posts/ads as examples for each cluster to determine coherency
  - coherency_check.py: code to obtain coherency results
  - coherency_prompting.txt: few-shot examples for LLM
  - bluesky_coherency (meta_coherency): contains conherency_results.csv from running coherency_check.py
    - List of clusters + whether each cluster is coherent or incoherent
      
### Summary Generation: Obtain summaries from each coherent cluster using an LLM (Mistral Large)
  - Prompts LLM to provide a summary for the top k posts/ads in the given cluster.
  - This process is completed for each cluster to obtain a full list of cluster summaries.
  - get_cluster_summaries.py: generate cluster summaries for Meta or Bluesky
  - bluesky_summaries (meta_summaries): contains cluster_summaries.csv with the list of clusters and their summaries
    
### Cluster Merging: Merges similar clusters with best Silhouette Score / DBI balance
  - merge.py: Given the values specified in GRID_SPEC, determines the best merge threshold 
              for SentenceBERT and uses this value to merge similar clusters
  - bluesky_merging (meta_merging): includes 4 files detailing:
    - Removed clusters: merged__removed_clusters.csv
    - The merging key: merged__best_cluster_to_merged.csv
    - The list of merged summaries: merged__merged_summaries.jsonl
    - List of threshold values with silhouette, dbi, number of resulting clusters, number of clusters with 1 summary
      - merged__summary_eval_threhold_sweep.csv
     
### Theme Generation: Generate themes for each merged cluster using an LLM (Mistral Large)
  - Given the merged cluster summary list, obtain a 1-3 word theme for each summary
  - get_themes.py: codes to obtain the list of themes based on merged summaries for Meta or Bluesky
  - bluesky_themes (meta_themes): contains cluster_themes.csv with the list of clusters, themes, and summaries

### Theme Assignment: Assign each post/ad to the proper theme using an LLM (Mistral Large)
  - Prompts LLM to assign each text to the best fitting theme given a post/ad text and the list of all cluster themes
  - assign_themes.py: Assigns all Meta ads or Bluesky posts to the best-fitting theme
  - bluesky_assigned_themes (meta_assigned_themes): contains the list of each unique ad/post and its assigned theme

### Summary (-> Theme) Assignment: Assign each post/ad to the proper summary using an LLM (Mistral Large)
  - Prompts LLM to assign each text to the best fitting summary given a post/ad text and the list of all cluster summaries
  - From the assigned summary, we take the theme assignment given for that summary from the **Theme generation** step (summary -> theme mapping)
  - assign_summaries.py: Assign all Meta ads or Bluesky posts to the best-fitting summary (and implied theme)
  - bluesky_assigned_themes (meta_assigned_themes): contains the list of each unique ad/post and its assigned theme

## Evaluation Code: Contains both the Human judge results on a sample of 500 ads/posts and the LLM judge results for the same sample of 500 ads/posts

### Human Annotation: Human judge results for Meta/Bluesky
  - bluesky_human_annotation.csv: contains the list of 500 samples for Bluesky with human annotation of Yes/No for both baselines and our methods (theme, summary->theme)
  - meta_human_annotation.csv: contains the list of 500 samples for Meta with human annotation of Yes/No for both baselines and our methods (theme, summary->theme)
    - meta_human_annotation_w_metadata.csv: also contains metadata such as impressions, spend, etc.
   
### LLM Annotation: LLM (Qwen3-235b) judge results for Meta/Bluesky and code
  - annotate.py: annotates each post/ad in the sample for both the baselines and our methods with Yes/No based on wither the topic/theme properly fits the text
  - bluesky_llm_annotation (meta_llm_annotation): contains the llm annotation results for Bluesky (Meta) with 4 new Yes/No columns (one for each compared method)

## Analysis Code:

### Event Analysis:
The purpose of the event analysis section is to obtain information on number of posts/ads within a before/after window based upon significant political events.
The chosen example for this work includes Charlie Kirk's death and the 2024 Presidential Election.

#### Folder Contents:
  - events_charts_bluesky: contains all event windows (2, 3, 7, 30 day windows before/after) for number of posts only [Bluesky Dataset]
    - additionally includes bar chart with top 10 themes overall by number of posts for Bluesky
  - events_charts_meta: contains all event windows (2, 3, 7, 30 day windows before/after) for number of ads, impressions, spend [Meta Dataset]
    - additionally includes bar chart with top 10 themes overall by number of ads for Meta  
  - bluesky_event_analysis.py: generates event_charts_bluesky
  - meta_event_anaylsis.py: generates events_charts_meta

### Stance Analysis:
The stance analysis code uses human-annotated stances to generate heatmaps for both the baseline methods and our method's generated themes
to assess whether the inferred themes meaningfully capture underlying climate-related stance. 

#### Folder Contents:
  - stance_labels: contains the human-annotated stance labels for Meta (meta_stance.csv) and Bluesky (bluesky_stance.csv)
  - stance_heatmaps.py: generates the stance heatmaps
  - stance_heatmaps_bluesky: stance heatmap figures for Bluesky baselines + our methods
  - stance_heatmaps_meta: stance heatmap figures for Meta baselines + our methods

### Common Theme Analysis:
This analysis section compares between similar themes found between Meta/Bluesky. It takes a combined JOINT_THEMES list and maps the Meta and Bluesky data with the combined themes.
Then, a UMAP representation is generated for the combined themes, as well as a bar chart comparing the number of Meta ads and Bluesky posts for each combined theme.

#### Folder Contents:
  - theme_comparison_outputs: contains generated UMAP representation (umap_fig.png), theme counts per platform
                              for each combined theme w/ corresponding bar chart, and examples from each combined theme
  - combined_theme_csvs: resulting csv files from mapping common themes between Meta/Bluesky
  - map_common_themes.py: maps common themes to produce combined theme csvs for Meta/Bluesky
      - 1 CSV is generated containing ONLY joint themes, 1 CSV is generated with joint themes replaced, but all other themes the same for both Bluesky and Meta (4 CSVs total)
  - umap_figures.py: generates the joint theme comparison bar chart and UMAP figure based on the joint themes 

## Baselines:
This folder contains the code to obtain LDA and BERTopic baseline topics and keywords. These results for Meta and Bluesky can be found under BERTopic/<bluesky/meta> and lda/<bluesky/meta>. 

## Results:
The results folder contains the full results for the ~20k posts/ads from Bluesky and Meta, with labeled themes and baseline topics, as well as the results for the sample of 500 posts/ads with both human and LLM annotation. 
For the sample of 500 texts, column names mean the following:
  - llm_theme_summ: Theme derived from LLM Summary (-> Theme) assignment portion of the pipeline code
  - llm_theme_theme: Theme derived from LLM theme assignment portion of the pipeline code
  - llm_theme_llm_truth: LLM judge annotation of whether llm_theme_theme properly fits the text
  - llm_summ_llm_truth: LLM judge annotation of whether llm_theme_summ properly fits the text 
  - llm_theme_human_truth: Human judge annotation of whether llm_theme_theme properly fits the text
  - llm_summ_human_truth: Human judge annotation of whether llm_theme_summ properly fits the text 


## Citation:

If you find the code, data, and paper useful in your work, please cite:

```
@article{sudhoff2026paid,
  title={Paid Voices vs. Public Feeds: Interpretable Cross-Platform Theme Modeling of Climate Discourse},
  author={Sudhoff, Samantha and Perumal, Pranav and Wu, Zhaoqing and Islam, Tunazzina},
  journal={arXiv preprint arXiv:2601.13317},
  year={2026}
}

```



