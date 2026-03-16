import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set(style="whitegrid")
sns.set_context("talk")

# update with proper path
DATA_PATH = "data/final_results_nov12_with_topics_and_meta.csv"
ELECTION_DAY = pd.Timestamp("2024-11-05")
WILDFIRE_DAY = pd.Timestamp("2025-01-07")

CHARTS_DIR = "event_charts_meta"
os.makedirs(CHARTS_DIR, exist_ok=True)


def sanitize_filename(s: str) -> str:
    """Make a safe filename from an event name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def thousands_formatter(x, pos):
    """Format y-axis tick labels with thousands separators."""
    return f"{x:,.0f}"


# load data

ads = pd.read_csv(DATA_PATH)

print("Columns in CSV:")
print(ads.columns.tolist())


def parse_meta_range(value):
    """
    Parse strings like 'lower_bound: 9000, upper_bound: 9999'
    into midpoint. If only lower_bound exists, use that.
    If parsing fails, return NaN.
    """
    if pd.isna(value):
        return np.nan
    s = str(value)

    lb_match = re.search(r"lower_bound:\s*([0-9]+)", s)
    ub_match = re.search(r"upper_bound:\s*([0-9]+)", s)

    if lb_match and ub_match:
        lb = int(lb_match.group(1))
        ub = int(ub_match.group(1))
        return (lb + ub) / 2.0

    if lb_match:
        lb = int(lb_match.group(1))
        return float(lb)

    return pd.to_numeric(s, errors="coerce")


def parse_spend_value(value):
    """
    Parse spend, handling Meta ranges and raw numeric-like strings
    """
    if pd.isna(value):
        return np.nan
    s = str(value)

    # If it looks like a Meta range, reuse parse_meta_range
    if "lower_bound" in s or "upper_bound" in s:
        return parse_meta_range(s)

    # Otherwise, strip non-numeric (keep digits and '.')
    cleaned = re.sub(r"[^\d.]", "", s)
    if cleaned == "":
        return np.nan
    return pd.to_numeric(cleaned, errors="coerce")


# impressions
if "impressions" in ads.columns:
    ads["impressions_numeric"] = ads["impressions"].apply(parse_meta_range)
else:
    print("WARNING: 'impressions' column not found; setting impressions_numeric = 0.")
    ads["impressions_numeric"] = 0.0

spend_candidates = [c for c in ads.columns if "spend" in c.lower()]
if spend_candidates:
    SPEND_COL = spend_candidates[0]
    print(f"Using spend column: {SPEND_COL}")
    ads["spend_numeric"] = ads[SPEND_COL].apply(parse_spend_value)
else:
    print("WARNING: No spend-like column found; setting spend_numeric = 0.")
    SPEND_COL = None
    ads["spend_numeric"] = 0.0

# Fill NaNs with 0 for numeric metrics
ads["impressions_numeric"] = ads["impressions_numeric"].fillna(0.0)
ads["spend_numeric"] = ads["spend_numeric"].fillna(0.0)

ads["ad_delivery_start_time"] = pd.to_datetime(ads["ad_delivery_start_time"])
ads["ad_delivery_stop_time"] = pd.to_datetime(ads["ad_delivery_stop_time"])

# Normalize theme strings
ads["theme"] = (
    ads["final_theme_llm_theme"]
    .astype(str)
    .str.strip()
    .str.strip('"')
    .str.strip("'")
    .str.strip('*')
    .str.strip('"')
)

print("Unique normalized themes (top 20):")
print(ads["theme"].value_counts().head(20))

# Remove missing/invalid dates
ads = ads.dropna(subset=["ad_delivery_start_time", "ad_delivery_stop_time"])
ads = ads[ads["ad_delivery_stop_time"] >= ads["ad_delivery_start_time"]].copy()

# daily expansion
date_lists = [
    list(pd.date_range(start, stop, freq="D"))
    for start, stop in zip(ads["ad_delivery_start_time"], ads["ad_delivery_stop_time"])
]
ads["date"] = date_lists

df_days = ads.explode("date").reset_index(drop=True)
df_days = df_days[df_days["date"].notna()].copy()
df_days["date"] = pd.to_datetime(df_days["date"])

id_candidates = [c for c in ads.columns if "ad_archive_id" in c or c == "ad_id"]
if id_candidates:
    AD_ID_COL = id_candidates[0]
    print(f"Using ad id column: {AD_ID_COL}")
else:
    print("WARNING: No ad id column found; will treat each row as unique ad.")
    AD_ID_COL = None


# metrics

def compute_before_after_metrics(df_days, event_date, window_days):
    """
    Compute before vs after metrics for:
        - # ads
        - total impressions
        - total spend
    within a window of +/- window_days around event_date.
    BEFORE: [event_date - window_days, event_date - 1]
    AFTER:  [event_date, event_date + window_days]  (includes event day)
    """

    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    before_mask = df_days["date"].between(before_start, before_end)
    after_mask  = df_days["date"].between(after_start, after_end)

    dfb = df_days[before_mask].copy()
    dfa = df_days[after_mask].copy()

    # Ensure numeric
    for d in (dfb, dfa):
        d["impressions_numeric"] = pd.to_numeric(
            d["impressions_numeric"], errors="coerce"
        ).fillna(0.0)
        d["spend_numeric"] = pd.to_numeric(
            d["spend_numeric"], errors="coerce"
        ).fillna(0.0)

    summary = {}

    for period, d in [("Before", dfb), ("After", dfa)]:
        if d.empty:
            summary[period] = {
                "ads": 0,
                "total_impressions": 0.0,
                "total_spend": 0.0,
            }
        else:
            if AD_ID_COL is not None:
                # One row per ad id within this window
                grouped = d.groupby(AD_ID_COL).agg({
                    "impressions_numeric": "max",
                    "spend_numeric": "max",
                })
            else:
                # Fallback: treat each row as its own "ad"
                grouped = d[["impressions_numeric", "spend_numeric"]]

            summary[period] = {
                "ads": grouped.shape[0],
                "total_impressions": float(grouped["impressions_numeric"].sum()),
                "total_spend": float(grouped["spend_numeric"].sum()),
            }

    return summary


def plot_metrics(summary, window_days, event_name):
    """
    Plot Total Number of Ads, Total Impressions, Total Spend
    for Before vs After, and save to file.
    """
    periods = ["Before", "After"]

    metric_defs = [
        ("ads", "Total Number of Ads"),
        ("total_impressions", "Total Impressions"),
        ("total_spend", "Total Spend"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (metric_key, metric_title) in zip(axes, metric_defs):
        values = [summary[p][metric_key] for p in periods]

        bars = ax.bar(periods, values)
        ax.set_title(metric_title)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle(
        f"{window_days}-Day Window Before/After: {event_name}",
        fontsize=16,
        y=1.03,
    )
    plt.tight_layout()

    fname = f"{sanitize_filename(event_name)}_{window_days}d_overall_metrics.png"
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot to: {out_path}")


def get_top5_themes_by_ads_diff(df_days, event_date, window_days):
    """
    Return the list of top-5 themes with the largest absolute
    before/after difference in number of ads,
    for the given event and window.
    """

    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    df_before = df_days[df_days["date"].between(before_start, before_end)].copy()
    df_after  = df_days[df_days["date"].between(after_start, after_end)].copy()

    if df_before.empty and df_after.empty:
        return []

    # Deduplicate by ad id so each ad is counted once per window
    if AD_ID_COL is not None:
        df_before_ads = df_before.drop_duplicates(subset=[AD_ID_COL])
        df_after_ads  = df_after.drop_duplicates(subset=[AD_ID_COL])
    else:
        df_before_ads = df_before
        df_after_ads  = df_after

    # # of unique ads per theme in each window
    if AD_ID_COL is not None:
        before_ads = df_before_ads.groupby("theme")[AD_ID_COL].nunique()
        after_ads  = df_after_ads.groupby("theme")[AD_ID_COL].nunique()
    else:
        before_ads = df_before_ads.groupby("theme").size().astype(float)
        after_ads  = df_after_ads.groupby("theme").size().astype(float)

    all_themes = before_ads.index.union(after_ads.index)
    before_ads = before_ads.reindex(all_themes, fill_value=0.0)
    after_ads  = after_ads.reindex(all_themes, fill_value=0.0)

    diffs = (after_ads - before_ads).abs()
    top_themes = diffs.sort_values(ascending=False).head(5).index.tolist()
    return top_themes


# top 5 themes by before/after difference

def plot_top5_themes_by_metric_window(
    df_days,
    event_date: pd.Timestamp,
    window_days: int,
    event_name: str,
    metric: str,          # "ads", "impressions", or "spend"
    metric_label: str,    # label for y-axis
    top_themes,           # list of themes chosen based on ads diff
):
    """
    For a given event + window and a fixed set of top_themes (based on
    ads difference), plot BEFORE vs AFTER for the requested metric:
      - metric in {"ads", "impressions", "spend"}
      - x-axis: Before / After
      - bars: the same top_themes in all three metric plots.
    """

    if not top_themes:
        print(f"[WARN] No top themes for {event_name} ({window_days}d, metric={metric})")
        return

    # Same windows as above
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    df_before = df_days[df_days["date"].between(before_start, before_end)].copy()
    df_after  = df_days[df_days["date"].between(after_start, after_end)].copy()

    # Deduplicate by ad id so each ad is counted once per window
    if AD_ID_COL is not None:
        df_before_ads = df_before.drop_duplicates(subset=[AD_ID_COL])
        df_after_ads  = df_after.drop_duplicates(subset=[AD_ID_COL])
    else:
        df_before_ads = df_before
        df_after_ads  = df_after

    def agg_per_theme(d: pd.DataFrame) -> pd.Series:
        if d.empty:
            return pd.Series(dtype=float)

        if metric == "ads":
            if AD_ID_COL is not None:
                return d.groupby("theme")[AD_ID_COL].nunique()
            else:
                return d.groupby("theme").size().astype(float)
        elif metric == "impressions":
            return d.groupby("theme")["impressions_numeric"].sum()
        elif metric == "spend":
            return d.groupby("theme")["spend_numeric"].sum()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    before_vals_all = agg_per_theme(df_before_ads)
    after_vals_all  = agg_per_theme(df_after_ads)

    # Restrict to top_themes and fill missing with 0
    before_top = before_vals_all.reindex(top_themes, fill_value=0.0)
    after_top  = after_vals_all.reindex(top_themes, fill_value=0.0)

    # Build dataframe for grouped bar plot:
    # index = ["Before", "After"], columns = themes
    plot_df = pd.DataFrame(
        {
            "Before": before_top,
            "After": after_top,
        },
        index=top_themes,
    ).T   # shape (2, n_themes)

    fig, ax = plt.subplots(figsize=(13, 5))
    plot_df.plot(kind="bar", ax=ax)

    # Title & labels
    title = (
        f"Event: {event_date.strftime('%d %b, %Y')} {event_name} "
        f"({window_days}-day window)"
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric_label)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # X tick labels
    ax.set_xticklabels(
        [f"{window_days} days before event", f"{window_days} days after event"],
        rotation=0
    )

    # Legend on middle-right
    ax.legend(
        title="Themes",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()

    fname = f"{sanitize_filename(event_name)}_{window_days}d_top5themes_{metric}.png"
    out_path = os.path.join(CHARTS_DIR, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved top-5 themes by {metric} plot to: {out_path}")


# ── Emerging / Disappearing themes ──────────────────────────────────────────

def theme_counts_in_window(
    df_days: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Returns a Series: theme -> number of unique ads in [start, end].
    Deduplicates by AD_ID_COL when available; otherwise counts rows.
    """
    sub = df_days[df_days["date"].between(start, end)]
    if sub.empty:
        return pd.Series(dtype=int)

    if AD_ID_COL is not None:
        sub = sub.drop_duplicates(subset=[AD_ID_COL])
        return sub.groupby("theme")[AD_ID_COL].nunique().sort_values(ascending=False)
    else:
        return sub.groupby("theme").size().sort_values(ascending=False)


def get_emerging_disappearing_themes(
    df_days: pd.DataFrame,
    event_date: pd.Timestamp,
    window_days: int,
    min_ads: int = 1,
    top_k: int = 25,
):
    """
    Emerging:     Before == 0 AND After >= min_ads
    Disappearing: Before >= min_ads AND After == 0

    Returns:
        emerging_df, disappearing_df
    Each DataFrame has columns: theme, before_ads, after_ads, delta
    """
    before_start = event_date - pd.Timedelta(days=window_days)
    before_end   = event_date - pd.Timedelta(days=1)
    after_start  = event_date
    after_end    = event_date + pd.Timedelta(days=window_days)

    before_counts = theme_counts_in_window(df_days, before_start, before_end)
    after_counts  = theme_counts_in_window(df_days, after_start, after_end)

    all_themes = before_counts.index.union(after_counts.index)
    before_counts = before_counts.reindex(all_themes, fill_value=0).astype(int)
    after_counts  = after_counts.reindex(all_themes, fill_value=0).astype(int)

    emerging_mask     = (before_counts == 0) & (after_counts >= min_ads)
    disappearing_mask = (before_counts >= min_ads) & (after_counts == 0)

    emerging = pd.DataFrame({
        "theme":       all_themes,
        "before_ads":  before_counts.values,
        "after_ads":   after_counts.values,
    })
    emerging = emerging[emerging_mask.values].copy()
    emerging["delta"] = emerging["after_ads"] - emerging["before_ads"]
    emerging = emerging.sort_values(
        ["after_ads", "theme"], ascending=[False, True]
    ).head(top_k)

    disappearing = pd.DataFrame({
        "theme":       all_themes,
        "before_ads":  before_counts.values,
        "after_ads":   after_counts.values,
    })
    disappearing = disappearing[disappearing_mask.values].copy()
    disappearing["delta"] = disappearing["after_ads"] - disappearing["before_ads"]
    disappearing = disappearing.sort_values(
        ["before_ads", "theme"], ascending=[False, True]
    ).head(top_k)

    return emerging, disappearing


def print_theme_change_list(title: str, df_list: pd.DataFrame, max_chars: int = 70):
    """Print emerging/disappearing themes with before/after ad counts."""
    print(title)
    if df_list.empty:
        print("  (none)")
        return
    for _, row in df_list.iterrows():
        t = str(row["theme"])
        if len(t) > max_chars:
            t = t[: max_chars - 3] + "..."
        print(
            f"  - {t} | before={int(row['before_ads'])}, "
            f"after={int(row['after_ads'])}, Δ={int(row['delta'])}"
        )


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_event(df_days, event_date, event_name, MIN_ADS: int = 1):
    print(f"\n===== Analysis: {event_name} on {event_date.date()} =====\n")

    # windows (2d, 3d, 1wk, 1mo)
    windows = [2, 3, 7, 30]

    for w in windows:
        print(f"\n=== {w}-DAY WINDOW ===")
        summary = compute_before_after_metrics(df_days, event_date, w)
        print(summary)  
        plot_metrics(summary, w, event_name)

        # Choose themes once based on ads-only before/after diff
        top_themes = get_top5_themes_by_ads_diff(df_days, event_date, w)
        print(f"Top themes by ads diff ({w}d): {top_themes}")

        if not top_themes:
            continue

        # Use the same themes for ads, impressions, and spend
        plot_top5_themes_by_metric_window(
            df_days, event_date, w, event_name,
            metric="ads",
            metric_label="Number of Ads",
            top_themes=top_themes,
        )
        plot_top5_themes_by_metric_window(
            df_days, event_date, w, event_name,
            metric="impressions",
            metric_label="Total Impressions",
            top_themes=top_themes,
        )
        plot_top5_themes_by_metric_window(
            df_days, event_date, w, event_name,
            metric="spend",
            metric_label="Total Spend",
            top_themes=top_themes,
        )

        # Emerging / Disappearing themes
        emerging, disappearing = get_emerging_disappearing_themes(
            df_days=df_days,
            event_date=event_date,
            window_days=w,
            min_ads=MIN_ADS,
            top_k=5,
        )

        print_theme_change_list(
            f"Emerging themes (Before=0, After≥{MIN_ADS}) [{w}d]:", emerging
        )
        print_theme_change_list(
            f"Disappearing themes (Before≥{MIN_ADS}, After=0) [{w}d]:", disappearing
        )


analyze_event(df_days, ELECTION_DAY, "US Election 2024 (Meta)")
analyze_event(df_days, WILDFIRE_DAY, "January 2025 California Wildfires (Meta)")

# top 10 themes overall

plt.figure(figsize=(12, 6))
ax = ads["theme"].value_counts().head(10).plot(kind="bar")
ax.set_title("Top 10 Most Common Themes (Meta)")
ax.set_ylabel("Number of Ads")
ax.set_xlabel("Theme")
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

fname = "top10_themes_overall.png"
out_path = os.path.join(CHARTS_DIR, fname)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved top-10 themes plot to: {out_path}")
