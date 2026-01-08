import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set(style="whitegrid")
sns.set_context("talk")

# replace with path to final results for bluesky
DATA_PATH = "data/bluesky_data/all_results.csv"
ELECTION_DAY = pd.Timestamp("2024-11-05")
CHARLIE_KIRK_DAY = pd.Timestamp("2025-09-10")

CHARTS_DIR = "events_charts_bluesky"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Choose which column defines llm assigned theme OR baseline topic
THEME_COL = "llm_theme_theme"   # alternatives: "topic_name", etc.

# can be indexed_at or date
TIMESTAMP_COL_CANDIDATES = ["indexed_at", "date"]

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def thousands_formatter(x, pos):
    return f"{x:,.0f}"

def normalize_theme(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.strip('"')
        .str.strip("'")
        .str.strip("*")
        .replace({"nan": np.nan, "None": np.nan})
    )

# load data
def read_csv_robust(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path, encoding="latin1", errors="replace")

df = read_csv_robust(DATA_PATH)

print("Columns in CSV:")
print(df.columns.tolist())

# Require cid
if "cid" not in df.columns:
    raise ValueError("CSV must contain a 'cid' column.")

# Pick timestamp column
ts_col = None
for c in TIMESTAMP_COL_CANDIDATES:
    if c in df.columns:
        ts_col = c
        break
if ts_col is None:
    raise ValueError(f"CSV must contain one of timestamp columns: {TIMESTAMP_COL_CANDIDATES}")

print(f"Using timestamp column: {ts_col}")

# Parse timestamp:
# - indexed_at looks like 2025-08-03T22:14:21.505000Z (UTC)
# - date looks like 2025-08-03
if ts_col == "indexed_at":
    df["date"] = pd.to_datetime(df["indexed_at"], errors="coerce", utc=True).dt.floor("D").dt.tz_localize(None)
else:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

# Drop rows without valid date
before_len = len(df)
df = df.dropna(subset=["date"]).copy()
print(f"Dropped {before_len - len(df)} rows with invalid dates.")

# Theme column
if THEME_COL not in df.columns:
    raise ValueError(f"Theme column '{THEME_COL}' not found. Available: {list(df.columns)}")

df["theme"] = normalize_theme(df[THEME_COL])

# drop missing theme
df = df.dropna(subset=["theme"]).copy()

# Ensure one row per post per day (Bluesky posts should already be 1 per cid)
df = df.drop_duplicates(subset=["cid"]).copy()

print("Top 20 themes:")
print(df["theme"].value_counts().head(20))

# metrics
def compute_before_after_post_metrics(df_posts: pd.DataFrame, event_date: pd.Timestamp, window_days: int):
    """
    BEFORE: [event_date - window_days, event_date - 1]
    AFTER:  [event_date, event_date + window_days]  (includes event day)
    Metric: number of unique posts (cid).
    """
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    dfb = df_posts[df_posts["date"].between(before_start, before_end)].copy()
    dfa = df_posts[df_posts["date"].between(after_start, after_end)].copy()

    return {
        "Before": {"posts": int(dfb["cid"].nunique())},
        "After":  {"posts": int(dfa["cid"].nunique())},
    }

def plot_overall_posts(summary, window_days: int, event_name: str):
    periods = ["Before", "After"]
    values = [summary[p]["posts"] for p in periods]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(periods, values)
    ax.set_title(f"Total Posts: {window_days}-Day Window\n{event_name}")
    ax.set_ylabel("Number of Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:,.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fname = f"{sanitize_filename(event_name)}_{window_days}d_overall_posts.png"
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall posts plot to: {out_path}")

# top-5 themes by before/after posts difference
def get_top5_themes_by_posts_diff(df_posts: pd.DataFrame, event_date: pd.Timestamp, window_days: int):
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    df_before = df_posts[df_posts["date"].between(before_start, before_end)].copy()
    df_after  = df_posts[df_posts["date"].between(after_start, after_end)].copy()

    if df_before.empty and df_after.empty:
        return []

    before_counts = df_before.groupby("theme")["cid"].nunique()
    after_counts  = df_after.groupby("theme")["cid"].nunique()

    all_themes = before_counts.index.union(after_counts.index)
    before_counts = before_counts.reindex(all_themes, fill_value=0)
    after_counts  = after_counts.reindex(all_themes, fill_value=0)

    diffs = (after_counts - before_counts).abs()
    return diffs.sort_values(ascending=False).head(5).index.tolist()

def plot_top5_themes_posts(df_posts: pd.DataFrame, event_date: pd.Timestamp, window_days: int, event_name: str, top_themes):
    if not top_themes:
        print(f"[WARN] No top themes for {event_name} ({window_days}d).")
        return

    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    df_before = df_posts[df_posts["date"].between(before_start, before_end)].copy()
    df_after  = df_posts[df_posts["date"].between(after_start, after_end)].copy()

    before_vals = df_before.groupby("theme")["cid"].nunique().reindex(top_themes, fill_value=0)
    after_vals  = df_after.groupby("theme")["cid"].nunique().reindex(top_themes, fill_value=0)

    plot_df = pd.DataFrame({"Before": before_vals, "After": after_vals}, index=top_themes).T

    fig, ax = plt.subplots(figsize=(13, 5))
    plot_df.plot(kind="bar", ax=ax)

    title = f"Event: {event_date.strftime('%d %b, %Y')} {event_name} ({window_days}-day window)"
    ax.set_title(title)
    ax.set_ylabel("Number of Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"{window_days} days before event", f"{window_days} days after event"],
        rotation=0
    )
    ax.legend(
        title="Themes",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    fname = f"{sanitize_filename(event_name)}_{window_days}d_top5themes_posts.png"
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-5 themes (posts) plot to: {out_path}")

def theme_counts_in_window(df_posts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Returns a Series: theme -> number of unique posts (cid) in [start, end].
    """
    sub = df_posts[df_posts["date"].between(start, end)]
    if sub.empty:
        return pd.Series(dtype=int)
    return sub.groupby("theme")["cid"].nunique().sort_values(ascending=False)


def get_emerging_disappearing_themes(
    df_posts: pd.DataFrame,
    event_date: pd.Timestamp,
    window_days: int,
    min_posts: int = 1,
    top_k: int = 25,
):
    """
    Emerging: Before == 0 and After >= min_posts
    Disappearing: Before >= min_posts and After == 0

    Returns:
        emerging_df, disappearing_df
    Each has columns: theme, before_posts, after_posts, delta
    """
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    before_counts = theme_counts_in_window(df_posts, before_start, before_end)
    after_counts  = theme_counts_in_window(df_posts, after_start, after_end)

    all_themes = before_counts.index.union(after_counts.index)
    before_counts = before_counts.reindex(all_themes, fill_value=0).astype(int)
    after_counts  = after_counts.reindex(all_themes, fill_value=0).astype(int)

    emerging_mask = (before_counts == 0) & (after_counts >= min_posts)
    disappearing_mask = (before_counts >= min_posts) & (after_counts == 0)

    emerging = pd.DataFrame({
        "theme": all_themes,
        "before_posts": before_counts.values,
        "after_posts": after_counts.values,
    })
    emerging = emerging[emerging_mask.values].copy()
    emerging["delta"] = emerging["after_posts"] - emerging["before_posts"]
    emerging = emerging.sort_values(["after_posts", "theme"], ascending=[False, True]).head(top_k)

    disappearing = pd.DataFrame({
        "theme": all_themes,
        "before_posts": before_counts.values,
        "after_posts": after_counts.values,
    })
    disappearing = disappearing[disappearing_mask.values].copy()
    disappearing["delta"] = disappearing["after_posts"] - disappearing["before_posts"]
    disappearing = disappearing.sort_values(["before_posts", "theme"], ascending=[False, True]).head(top_k)

    return emerging, disappearing


def print_theme_change_list(title: str, df_list: pd.DataFrame, max_chars: int = 70):
    """
    Print themes with counts.
    """
    print(title)
    if df_list.empty:
        print("  (none)")
        return

    for _, row in df_list.iterrows():
        t = str(row["theme"])
        if len(t) > max_chars:
            t = t[: max_chars - 3] + "..."
        print(f"  - {t} | before={int(row['before_posts'])}, after={int(row['after_posts'])}, Δ={int(row['delta'])}")


# main analysis
def analyze_event(df_posts: pd.DataFrame, event_date: pd.Timestamp, event_name: str, MIN_POSTS: int = 1):
    print(f"\n===== Analysis: {event_name} on {event_date.date()} =====\n")
    windows = [2, 3, 7, 30]

    for w in windows:
        print(f"\n=== {w}-DAY WINDOW ===")
        summary = compute_before_after_post_metrics(df_posts, event_date, w)
        print(summary)
        plot_overall_posts(summary, w, event_name)

        top_themes = get_top5_themes_by_posts_diff(df_posts, event_date, w)
        print(f"Top themes by posts diff ({w}d): {top_themes}")
        plot_top5_themes_posts(df_posts, event_date, w, event_name, top_themes)

        # Emerging / Disappearing themes
        emerging, disappearing = get_emerging_disappearing_themes(
            df_posts=df_posts,
            event_date=event_date,
            window_days=w,
            min_posts=MIN_POSTS,
            top_k=5,
        )

        print_theme_change_list(f"Emerging themes (Before=0, After≥{MIN_POSTS}) [{w}d]:", emerging)
        print_theme_change_list(f"Disappearing themes (Before≥{MIN_POSTS}, After=0) [{w}d]:", disappearing)

analyze_event(df, ELECTION_DAY, "US Election 2024 (Bluesky)")
analyze_event(df, CHARLIE_KIRK_DAY, "Charlie Kirk Death (Bluesky)")

# Top 10 Themes Overall
plt.figure(figsize=(12, 6))
ax = df["theme"].value_counts().head(10).plot(kind="bar")
ax.set_title("Top 10 Most Common Themes (Bluesky)")
ax.set_ylabel("Number of Posts")
ax.set_xlabel("Theme")
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

fname = "top10_themes_overall_bluesky.png"
out_path = os.path.join(CHARTS_DIR, fname)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved top-10 themes plot to: {out_path}")
