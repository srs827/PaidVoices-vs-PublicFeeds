## Event Analysis
The purpose of the event analysis section is to obtain information on number of posts/ads within a before/after window based upon significant political events.
The chosen example for this work includes Charlie Kirk's death and the 2024 Presidential Election.

### Folder Contents:
  - events_charts_bluesky: contains all event windows (2, 3, 7, 30 day windows before/after) for number of posts only [Bluesky Dataset]
    - additionally includes bar chart with top 10 themes overall by number of posts for Bluesky
  - events_charts_meta: contains all event windows (2, 3, 7, 30 day windows before/after) for number of ads, impressions, spend [Meta Dataset]
    - additionally includes bar chart with top 10 themes overall by number of ads for Meta  
  - bluesky_event_analysis.py: generates event_charts_bluesky
  - meta_event_anaylsis.py: generates events_charts_meta
