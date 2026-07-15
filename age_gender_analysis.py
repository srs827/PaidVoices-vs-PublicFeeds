import pandas as pd
import json

# ── CONFIG ────────────────────────────────────────────────────────────────────
META_ADS_FILE     = "full_meta_results.csv"
SAMPLE_FILE       = "stance_labeled_results.csv"
OUT_AGE_THEME     = "demo_age_theme.csv"
OUT_GENDER_THEME  = "demo_gender_theme.csv"
OUT_AGE_STANCE    = "demo_age_stance.csv"
OUT_GENDER_STANCE = "demo_gender_stance.csv"
# ─────────────────────────────────────────────────────────────────────────────

AGE_ORDER    = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDER_ORDER = ["female", "male", "unknown"]

def parse_demo(raw):
    """Parse demographic_distribution JSON into list of dicts."""
    if pd.isna(raw):
        return []
    try:
        return json.loads(f"[{raw}]")
    except Exception:
        return []

def explode_demo(df, extra_cols):
    """Explode demographic_distribution into one row per (cid, age, gender)."""
    records = []
    for _, row in df.iterrows():
        for d in parse_demo(row["demographic_distribution"]):
            age    = d.get("age", "").strip()
            gender = d.get("gender", "").strip()
            pct    = d.get("percentage", 0.0)
            if not age or not gender:
                continue
            record = {"age": age, "gender": gender, "demo_pct": pct}
            for col in extra_cols:
                record[col] = row[col]
            records.append(record)
    return pd.DataFrame(records)


# ── THEME ANALYSIS: all ads ───────────────────────────────────────────────────
print("Loading full meta ads...")
meta = pd.read_csv(META_ADS_FILE, dtype={"cid": str})[
    ["cid", "final_theme_llm_theme", "demographic_distribution"]
]
meta["final_theme_llm_theme"] = (
    meta["final_theme_llm_theme"].str.replace(r'[\*\"]+', '', regex=True).str.strip()
)
print(f"  {len(meta)} total ads")

demo_long = explode_demo(meta, ["cid", "final_theme_llm_theme"])

# Age x Theme heatmap data
age_theme = (
    demo_long.groupby(["age", "final_theme_llm_theme"])["demo_pct"]
    .sum().reset_index().rename(columns={"demo_pct": "weight"})
)
# Normalise each age group to 100% so rows are comparable
age_totals = age_theme.groupby("age")["weight"].sum()
age_theme["pct"] = (age_theme["weight"] / age_theme["age"].map(age_totals) * 100).round(2)
age_theme = age_theme[age_theme["age"].isin(AGE_ORDER)]
age_theme.to_csv(OUT_AGE_THEME, index=False)
print(f"Age x Theme: {OUT_AGE_THEME}")

# Gender x Theme heatmap data
gender_theme = (
    demo_long.groupby(["gender", "final_theme_llm_theme"])["demo_pct"]
    .sum().reset_index().rename(columns={"demo_pct": "weight"})
)
gender_totals = gender_theme.groupby("gender")["weight"].sum()
gender_theme["pct"] = (gender_theme["weight"] / gender_theme["gender"].map(gender_totals) * 100).round(2)
gender_theme = gender_theme[gender_theme["gender"].isin(GENDER_ORDER)]
gender_theme.to_csv(OUT_GENDER_THEME, index=False)
print(f"Gender × Theme: {OUT_GENDER_THEME}")


# ── STANCE ANALYSIS: 700-row sample ──────────────────────────────────────────
print("\nLoading 700-row sample...")
sample = pd.read_csv(SAMPLE_FILE, dtype={"cid": str})[["cid", "Stance"]]
sample_meta = sample.merge(meta[["cid", "demographic_distribution"]], on="cid", how="left")
stance_long = explode_demo(sample_meta, ["cid", "Stance"])
stance_long = stance_long[stance_long["Stance"].isin(["Pro-Climate", "Pro-Energy"])]

# Age x Stance
age_stance = (
    stance_long.groupby(["age", "Stance"])["demo_pct"]
    .sum().unstack(fill_value=0).reset_index()
)
age_stance.columns.name = None
for col in ["Pro-Climate", "Pro-Energy"]:
    if col not in age_stance.columns:
        age_stance[col] = 0.0
age_stance["total"] = age_stance["Pro-Climate"] + age_stance["Pro-Energy"]
age_stance["pct_pro_climate"] = (age_stance["Pro-Climate"] / age_stance["total"].replace(0, float("nan")) * 100).round(1)
age_stance["pct_pro_energy"]  = (age_stance["Pro-Energy"]  / age_stance["total"].replace(0, float("nan")) * 100).round(1)
age_stance = age_stance[age_stance["age"].isin(AGE_ORDER)]
age_stance["age"] = pd.Categorical(age_stance["age"], categories=AGE_ORDER, ordered=True)
age_stance = age_stance.sort_values("age")
age_stance.to_csv(OUT_AGE_STANCE, index=False)
print(f"Age x Stance: {OUT_AGE_STANCE}")

# Gender x Stance
gender_stance = (
    stance_long.groupby(["gender", "Stance"])["demo_pct"]
    .sum().unstack(fill_value=0).reset_index()
)
gender_stance.columns.name = None
for col in ["Pro-Climate", "Pro-Energy"]:
    if col not in gender_stance.columns:
        gender_stance[col] = 0.0
gender_stance["total"] = gender_stance["Pro-Climate"] + gender_stance["Pro-Energy"]
gender_stance["pct_pro_climate"] = (gender_stance["Pro-Climate"] / gender_stance["total"].replace(0, float("nan")) * 100).round(1)
gender_stance["pct_pro_energy"]  = (gender_stance["Pro-Energy"]  / gender_stance["total"].replace(0, float("nan")) * 100).round(1)
gender_stance = gender_stance[gender_stance["gender"].isin(GENDER_ORDER)]
gender_stance["gender"] = pd.Categorical(gender_stance["gender"], categories=GENDER_ORDER, ordered=True)
gender_stance = gender_stance.sort_values("gender")
gender_stance.to_csv(OUT_GENDER_STANCE, index=False)
print(f"Gender x Stance: {OUT_GENDER_STANCE}")