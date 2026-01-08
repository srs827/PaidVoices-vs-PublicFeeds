# Paid Voices versus Public Feeds: How Climate Narratives Diverge Online

This respository contains all code from the theme generation pipeline under "pipeline", which can be modified to work with any target dataset. 
**Note that full results obtained on the Bluesky and Meta data will be made publically available, with access links to be added in the future.**

The repository also contains all analysis-related code under "analysis_and_figures", such as:
  - Comparison of common themes between Meta and Bluesky
  - Event analysis code and resulting figures
  - Stance analysis code and resulting figures

We provide the evaluation code under "evaluation", where we include the code to provide LLM-judge annotation of the obtained results. 

Each section is broken down as follows:

## Pipeline Code:

### Preprocessing:
### Clustering Code:
### Coherency Checking:
### Summary Generation:
### Cluster Merging:
### Theme Generation:

## Evaluation Code:
### Human Annotation:
### LLM Annotation:

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

##### Folder Contents:




