import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
PALETTE   = ["#6C63FF", "#00D4AA", "#FF6B9D", "#FFB347", "#64DFDF",
             "#A78BFA", "#34D399", "#F472B6"]
BG        = "#0D0F1A"
CARD      = "#141728"
TEXT      = "#F0F2FF"
TEXT_MUTED= "#8A8FAD"
GRID      = "#252840"

rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   "#6C63FF",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "legend.facecolor":  CARD,
    "legend.edgecolor":  GRID,
})

NUMERIC_COLS = [
    "duration", "impressions", "clicks", "leads",
    "conversions", "revenue", "acquisition_cost",
    "roi", "engagement_score",
]


# ─────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────
def load_and_prep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ─────────────────────────────────────────────
# EDA REPORTS
# ─────────────────────────────────────────────
def univariate_analysis(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  UNIVARIATE SUMMARY STATISTICS")
    print("═" * 60)
    print(df[NUMERIC_COLS].describe().round(3).to_string())

    present = [c for c in NUMERIC_COLS if c in df.columns]
    n = len(present)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    fig.suptitle("Univariate Distributions", fontsize=16, color="#6C63FF",
                 fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(present):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i],
                     color=PALETTE[i % len(PALETTE)], edgecolor="none", alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def campaign_rankings(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  TOP 5 CAMPAIGNS BY ROI")
    print("═" * 60)
    top = df.nlargest(5, "roi")[["campaign_id", "campaign_type", "roi", "revenue", "conversions"]]
    print(top.to_string(index=False))

    print("\n" + "═" * 60)
    print("  BOTTOM 5 CAMPAIGNS BY ROI")
    print("═" * 60)
    bot = df.nsmallest(5, "roi")[["campaign_id", "campaign_type", "roi", "revenue", "conversions"]]
    print(bot.to_string(index=False))

    print("\n  TOP 5 BY REVENUE")
    print(df.nlargest(5, "revenue")[["campaign_id", "campaign_type", "revenue", "roi"]].to_string(index=False))


def segment_analysis(df: pd.DataFrame) -> None:
    segments = {
        "Campaign Type":   "campaign_type",
        "Target Audience": "target_audience",
        "Language":        "language",
    }
    for label, col in segments.items():
        if col not in df.columns:
            continue
        print(f"\n  Avg ROI by {label}:")
        print(df.groupby(col)["roi"].mean().sort_values(ascending=False).round(3).to_string())

    # Bar charts
    fig, axes = plt.subplots(1, len(segments), figsize=(18, 5))
    fig.suptitle("Average ROI by Segment", fontsize=15, color="#6C63FF", fontweight="bold")

    for ax, (label, col) in zip(axes, segments.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df.groupby(col)["roi"].mean().sort_values()
        bars = ax.barh(data.index, data.values,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(data))],
                       edgecolor="none", height=0.6)
        ax.set_title(label)
        ax.set_xlabel("Avg ROI")
        ax.grid(axis="x")

    plt.tight_layout()
    plt.show()


def channel_effectiveness(df: pd.DataFrame) -> None:
    if "channel_used" not in df.columns:
        return
    channels = df.copy()
    channels["channel_list"] = channels["channel_used"].str.split(", ")
    exploded = channels.explode("channel_list")
    ch_perf = (
        exploded.groupby("channel_list")
        .agg(roi=("roi", "mean"), revenue=("revenue", "sum"),
             conversions=("conversions", "sum"), impressions=("impressions", "sum"))
        .round(2)
        .sort_values("roi", ascending=False)
    )
    print("\n  Channel Effectiveness:")
    print(ch_perf.to_string())


def correlation_heatmap(df: pd.DataFrame) -> None:
    present = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[present].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(260, 10, as_cmap=True),
        linewidths=0.5, linecolor=GRID,
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax.set_title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.show()


def outlier_analysis(df: pd.DataFrame) -> None:
    present = [c for c in NUMERIC_COLS if c in df.columns]

    fig, ax = plt.subplots(figsize=(16, 6))
    data = [df[c].dropna() for c in present]
    bp = ax.boxplot(data, patch_artist=True, notch=True, vert=True,
                    medianprops=dict(color="#FFB347", linewidth=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([c.replace("_", "\n") for c in present], fontsize=9)
    ax.set_title("Outlier Detection — Boxplots", fontsize=14)
    plt.tight_layout()
    plt.show()

    q1, q3 = df["roi"].quantile(0.25), df["roi"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df["roi"] < q1 - 1.5 * iqr) | (df["roi"] > q3 + 1.5 * iqr)]
    print(f"\n  ROI outliers detected: {len(outliers)}")


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_eda(path: str = "nykaa_campaign_data.csv") -> None:
    df = load_and_prep(path)
    univariate_analysis(df)
    campaign_rankings(df)
    segment_analysis(df)
    channel_effectiveness(df)
    correlation_heatmap(df)
    outlier_analysis(df)


if __name__ == "__main__":
    run_eda()