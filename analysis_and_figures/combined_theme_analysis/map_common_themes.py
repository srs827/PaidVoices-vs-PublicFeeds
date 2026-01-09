import pandas as pd
import re 

# replace with paths to final results for meta/bluesky
POSTS_CSV_IN = "data/bluesky_data/all_results.csv"      
ADS_CSV_IN   = "data/meta_data/all_results.csv"        
POSTS_CSV_OUT = "bluesky_combined_themes.csv"
ADS_CSV_OUT   = "meta_combined_themes.csv"
POSTS_CSV_OUT_COMMON = "bluesky_combined_themes_common.csv"
ADS_CSV_OUT_COMMON   = "meta_combined_themes_common.csv"

# Columns we’re mapping
# Replace based upon the column which provides assigned theme
# (under theme_assignment folder)
# for meta all_results.csv, provided data has "final_theme_llm_theme"
# for bluesky all_results.csv, provided data has "llm_theme_theme"
POSTS_THEME_COL = "llm_theme_theme"
ADS_THEME_COL   = "final_theme_llm_theme"


def normalize_theme(s: str) -> str:
    """
    Normalize theme strings for robust matching.
    - strip whitespace
    - remove leading/trailing ** and quotes
    - collapse internal spaces
    - lowercase
    """
    if s is None:
        return ""
    s = str(s).strip()
    # strip markdown ** and surrounding quotes
    s = s.strip("*")
    s = s.strip('"“”')
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# mappings to combine themes
posts_mapping_raw = { # bluesky
    "Sustainable luxury fashion": "Sustainable Fashion",
    "Trees for Climate": "Anti-Deforestation",
    "Deforestation & Wildlife Crisis": "Anti-Deforestation",
    "Water pollution crisis": "Water pollution",
    "Marine climate crisis": "Ocean conservation",
    "Oil's Deceptive Greed": "Anti-Big Oil",
    "Clean Energy Push": "Clean energy",
    "Renewable Energy Momentum": "Clean energy",
    "Renewable energy surge": "Clean energy",
    "Resource expansion drilling": "Anti-Drilling",
    "Climate-fueled wildfires": "Climate Wildfires",
}

ads_mapping_raw = { # meta
    "Ethical fashion advocacy": "Sustainable Fashion",
    "Pro-forest climate action": "Anti-Deforestation",
    "Clean water advocacy": "Water pollution",
    "Ocean conservation call": "Ocean conservation",
    "Big Oil accountability": "Anti-Big Oil",
    "Clean energy boom": "Clean energy",
    "Pro-clean energy": "Clean energy",
    "Clean energy advocacy": "Clean energy",
    "Anti-offshore drilling": "Anti-Drilling",
    "Wildfire recovery advocacy": "Climate Wildfires",
}

# Build normalized lookup dicts
posts_mapping = {normalize_theme(k): v for k, v in posts_mapping_raw.items()}
ads_mapping   = {normalize_theme(k): v for k, v in ads_mapping_raw.items()}


def map_theme(value, mapping):
    """
    Map a raw theme value to combined theme.
    If it’s not in the mapping (after normalization), return original.
    """
    if pd.isna(value):
        return value
    key = normalize_theme(value)
    return mapping.get(key, value)


# joint themes
JOINT_THEMES = {
    "Sustainable Fashion",
    "Anti-Deforestation",
    "Water pollution",
    "Ocean conservation",
    "Anti-Big Oil",
    "Clean energy",
    "Anti-Drilling",
    "Climate Wildfires",
}

# load data
posts_df = pd.read_csv(POSTS_CSV_IN, encoding="latin1")
ads_df   = pd.read_csv(ADS_CSV_IN, encoding="latin1")

# Sanity check columns exist
if POSTS_THEME_COL not in posts_df.columns:
    raise ValueError(f"Column '{POSTS_THEME_COL}' not found in posts_df.")
if ADS_THEME_COL not in ads_df.columns:
    raise ValueError(f"Column '{ADS_THEME_COL}' not found in ads_df.")

# apply mappings (overwrite original theme columns)
posts_df[POSTS_THEME_COL] = posts_df[POSTS_THEME_COL].apply(
    lambda x: map_theme(x, posts_mapping)
)
ads_df[ADS_THEME_COL] = ads_df[ADS_THEME_COL].apply(
    lambda x: map_theme(x, ads_mapping)
)

# save full combined versions
posts_df.to_csv(POSTS_CSV_OUT, index=False)
ads_df.to_csv(ADS_CSV_OUT, index=False)

print(f"Saved posts with combined themes to: {POSTS_CSV_OUT}")
print(f"Saved ads with combined themes to:   {ADS_CSV_OUT}")

# filter to only joint themes
posts_common = posts_df[posts_df[POSTS_THEME_COL].isin(JOINT_THEMES)].copy()
ads_common   = ads_df[ads_df[ADS_THEME_COL].isin(JOINT_THEMES)].copy()

posts_common.to_csv(POSTS_CSV_OUT_COMMON, index=False)
ads_common.to_csv(ADS_CSV_OUT_COMMON, index=False)

print(f"Saved posts with ONLY joint themes to: {POSTS_CSV_OUT_COMMON}")
print(f"Saved ads with ONLY joint themes to:   {ADS_CSV_OUT_COMMON}")