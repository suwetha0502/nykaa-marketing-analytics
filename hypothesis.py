import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations

# ─────────────────────────────────────────────
# GLOBAL STYLE  (matches eda.py)
# ─────────────────────────────────────────────
PALETTE    = ["#6C63FF", "#00D4AA", "#FF6B9D", "#FFB347", "#64DFDF",
              "#A78BFA", "#34D399", "#F472B6"]
BG         = "#0D0F1A"
CARD       = "#141728"
TEXT       = "#F0F2FF"
TEXT_MUTED = "#8A8FAD"
GRID       = "#252840"
SIG_COLOR  = "#00D4AA"
INSIG_COLOR= "#FF6B9D"

rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  "#6C63FF",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "xtick.color":      TEXT_MUTED,
    "ytick.color":      TEXT_MUTED,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "legend.facecolor": CARD,
    "legend.edgecolor": GRID,
})

ALPHA = 0.05  # significance level


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _ttest(g1: pd.Series, g2: pd.Series) -> tuple[float, float]:
    """Welch's t-test, returns (t_stat, p_value)."""
    return stats.ttest_ind(g1.dropna(), g2.dropna(), equal_var=False)


def _sig_label(p: float) -> str:
    if p < 0.001: return "★★★ Highly Significant"
    if p < 0.01:  return "★★  Significant"
    if p < ALPHA: return "★   Significant"
    return "✗   Not Significant"


def _sig_color(p: float) -> str:
    return SIG_COLOR if p < ALPHA else INSIG_COLOR


def _print_header(title: str) -> None:
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)


# ─────────────────────────────────────────────
# A/B TESTING FUNCTIONS
# ─────────────────────────────────────────────
def test_campaign_types(df: pd.DataFrame) -> pd.DataFrame:
    _print_header("A/B TESTING — Campaign Types vs ROI")
    types = df["campaign_type"].dropna().unique()
    records = []
    for a, b in combinations(types, 2):
        g1 = df[df["campaign_type"] == a]["roi"]
        g2 = df[df["campaign_type"] == b]["roi"]
        t, p = _ttest(g1, g2)
        label = _sig_label(p)
        print(f"  {a:20s} vs {b:20s} | p={p:.4f}  {label}")
        records.append({"Group A": a, "Group B": b, "p-value": round(p, 4),
                         "Significant": p < ALPHA, "Verdict": label})
    return pd.DataFrame(records)


def test_languages(df: pd.DataFrame) -> pd.DataFrame:
    _print_header("A/B TESTING — Languages vs ROI")
    langs = df["language"].dropna().unique()
    records = []
    for a, b in combinations(langs, 2):
        g1 = df[df["language"] == a]["roi"]
        g2 = df[df["language"] == b]["roi"]
        t, p = _ttest(g1, g2)
        label = _sig_label(p)
        print(f"  {a:15s} vs {b:15s} | p={p:.4f}  {label}")
        records.append({"Lang A": a, "Lang B": b, "p-value": round(p, 4),
                         "Significant": p < ALPHA, "Verdict": label})
    return pd.DataFrame(records)


def test_duration(df: pd.DataFrame) -> dict:
    _print_header("A/B TESTING — Short vs Long Duration")
    median_dur = df["duration"].median()
    df = df.copy()
    df["duration_group"] = np.where(df["duration"] <= median_dur, "Short", "Long")

    short = df[df["duration_group"] == "Short"]["roi"]
    long  = df[df["duration_group"] == "Long"]["roi"]
    t, p  = _ttest(short, long)
    label = _sig_label(p)
    print(f"  Short (n={len(short)}) vs Long (n={len(long)}) | p={p:.4f}  {label}")
    return {"p_value": round(p, 4), "significant": p < ALPHA, "df_with_group": df}


# ─────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────
def summary_tables(df: pd.DataFrame, df_dur: pd.DataFrame) -> None:
    for col, label in [("campaign_type", "Campaign Type"),
                       ("language",       "Language"),
                       ("duration_group", "Duration Group")]:
        src = df_dur if col == "duration_group" else df
        if col not in src.columns:
            continue
        _print_header(f"Mean ROI by {label}")
        tbl = src.groupby(col)["roi"].agg(["mean", "count", "std"]).round(3)
        print(tbl.to_string())


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_campaign_types(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    types = sorted(df["campaign_type"].dropna().unique())
    data  = [df[df["campaign_type"] == t]["roi"].dropna() for t in types]
    bp = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, v in enumerate(bp["bodies"]):
        v.set_facecolor(PALETTE[i % len(PALETTE)])
        v.set_alpha(0.7)
    bp["cmedians"].set_color("#FFB347")
    bp["cmedians"].set_linewidth(2)
    ax.set_xticks(range(1, len(types) + 1))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_title("ROI Distribution by Campaign Type")
    ax.set_ylabel("ROI")
    plt.tight_layout()
    plt.show()


def plot_languages(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    order = df.groupby("language")["roi"].median().sort_values(ascending=False).index
    sns.boxplot(x="language", y="roi", data=df, order=order,
                palette=PALETTE, ax=ax, linewidth=1.2, flierprops=dict(marker=".", color=TEXT_MUTED))
    ax.set_title("ROI Distribution by Language")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.show()


def plot_duration(df_dur: pd.DataFrame) -> None:
    if "duration_group" not in df_dur.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (grp, color) in zip(axes, [("Short", PALETTE[0]), ("Long", PALETTE[1])]):
        subset = df_dur[df_dur["duration_group"] == grp]["roi"].dropna()
        sns.histplot(subset, kde=True, ax=ax, color=color, edgecolor="none", alpha=0.8)
        ax.axvline(subset.mean(), color="#FFB347", linestyle="--", linewidth=1.5,
                   label=f"Mean: {subset.mean():.2f}")
        ax.set_title(f"{grp} Duration — ROI")
        ax.legend()
    plt.suptitle("ROI: Short vs Long Duration Campaigns", color="#6C63FF",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_p_value_summary(results_df: pd.DataFrame, title: str) -> None:
    if results_df.empty:
        return
    results_df = results_df.copy().sort_values("p-value")
    pair_labels = results_df.apply(
        lambda r: f"{r.iloc[0]} vs {r.iloc[1]}", axis=1
    )
    colors = [SIG_COLOR if sig else INSIG_COLOR for sig in results_df["Significant"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(results_df) * 0.5)))
    bars = ax.barh(pair_labels, results_df["p-value"], color=colors, edgecolor="none", height=0.5)
    ax.axvline(ALPHA, color="#FFB347", linestyle="--", linewidth=1.5, label=f"α = {ALPHA}")
    ax.set_title(f"p-values — {title}")
    ax.set_xlabel("p-value")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_hypothesis_tests(path: str = "nykaa_campaign_data.csv") -> None:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    ct_results = test_campaign_types(df)
    lg_results = test_languages(df)
    dur_result  = test_duration(df)
    df_dur = dur_result["df_with_group"]

    summary_tables(df, df_dur)

    plot_campaign_types(df)
    plot_languages(df)
    plot_duration(df_dur)
    plot_p_value_summary(ct_results, "Campaign Types")
    plot_p_value_summary(lg_results, "Languages")

    print("\n✅ A/B Testing Completed!")


if __name__ == "__main__":
    run_hypothesis_tests()