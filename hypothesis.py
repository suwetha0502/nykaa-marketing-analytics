"""
Hypothesis testing module for Nykaa Marketing Analytics.
Provides a HypothesisTester class that app.py can import and use.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

import matplotlib
matplotlib.use('Agg')  # headless — safe for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns


PALETTE = ['#6C63FF', '#00D4AA', '#FF6B9D', '#FFB347', '#64DFDF',
           '#A78BFA', '#34D399', '#F472B6']
SIG_COLOR   = '#00D4AA'
INSIG_COLOR = '#FF6B9D'
ALPHA       = 0.05


def _apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor': '#0D0F1A',
        'axes.facecolor':   '#141728',
        'axes.edgecolor':   '#252840',
        'axes.labelcolor':  '#F0F2FF',
        'axes.titlecolor':  '#6C63FF',
        'xtick.color':      '#8A8FAD',
        'ytick.color':      '#8A8FAD',
        'text.color':       '#F0F2FF',
        'grid.color':       '#252840',
        'grid.linestyle':   '--',
        'grid.alpha':       0.5,
        'legend.facecolor': '#141728',
        'legend.edgecolor': '#252840',
    })


def _cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    """Cohen's d effect size."""
    g1, g2 = g1.dropna(), g2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((len(g1) - 1) * g1.std() ** 2 + (len(g2) - 1) * g2.std() ** 2)
        / (len(g1) + len(g2) - 2)
    )
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std != 0 else 0.0


class HypothesisTester:
    """Runs A/B hypothesis tests on campaign data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── Campaign Type Tests ──────────────────────────────────────────────────

    def test_campaign_types(self) -> pd.DataFrame:
        """
        Pairwise Welch t-tests on ROI across campaign types.
        Returns a DataFrame with columns:
          group1, group2, p_value, significant, effect_size,
          group1_mean, group2_mean, difference
        """
        if 'campaign_type' not in self.df.columns or 'roi' not in self.df.columns:
            return pd.DataFrame()

        types = self.df['campaign_type'].dropna().unique()
        records = []
        for a, b in combinations(types, 2):
            g1 = self.df[self.df['campaign_type'] == a]['roi']
            g2 = self.df[self.df['campaign_type'] == b]['roi']
            if len(g1) < 2 or len(g2) < 2:
                continue
            _, p = stats.ttest_ind(g1.dropna(), g2.dropna(), equal_var=False)
            records.append({
                'group1':       a,
                'group2':       b,
                'p_value':      round(float(p), 4),
                'significant':  bool(p < ALPHA),
                'effect_size':  round(_cohens_d(g1, g2), 3),
                'group1_mean':  round(float(g1.mean()), 3),
                'group2_mean':  round(float(g2.mean()), 3),
                'difference':   round(float(g1.mean() - g2.mean()), 3),
            })
        return pd.DataFrame(records)

    # ── Language Tests ───────────────────────────────────────────────────────

    def test_languages(self) -> pd.DataFrame:
        """
        Pairwise Welch t-tests on ROI across languages.
        Returns same schema as test_campaign_types().
        """
        if 'language' not in self.df.columns or 'roi' not in self.df.columns:
            return pd.DataFrame()

        langs = self.df['language'].dropna().unique()
        records = []
        for a, b in combinations(langs, 2):
            g1 = self.df[self.df['language'] == a]['roi']
            g2 = self.df[self.df['language'] == b]['roi']
            if len(g1) < 2 or len(g2) < 2:
                continue
            _, p = stats.ttest_ind(g1.dropna(), g2.dropna(), equal_var=False)
            records.append({
                'group1':      a,
                'group2':      b,
                'p_value':     round(float(p), 4),
                'significant': bool(p < ALPHA),
                'effect_size': round(_cohens_d(g1, g2), 3),
                'group1_mean': round(float(g1.mean()), 3),
                'group2_mean': round(float(g2.mean()), 3),
                'difference':  round(float(g1.mean() - g2.mean()), 3),
            })
        return pd.DataFrame(records)

    # ── Duration Test ────────────────────────────────────────────────────────

    def test_duration_groups(self) -> dict:
        """
        Welch t-test: short-duration vs long-duration campaigns by ROI.
        Returns a dict with keys: p_value, significant, effect_size,
          group1_mean (Short), group2_mean (Long)
        """
        if 'duration' not in self.df.columns or 'roi' not in self.df.columns:
            return {'error': 'Missing duration or roi column'}

        median_dur = self.df['duration'].median()
        short = self.df[self.df['duration'] <= median_dur]['roi']
        long_ = self.df[self.df['duration'] >  median_dur]['roi']

        if len(short) < 2 or len(long_) < 2:
            return {'error': 'Insufficient data'}

        _, p = stats.ttest_ind(short.dropna(), long_.dropna(), equal_var=False)
        return {
            'p_value':      round(float(p), 4),
            'significant':  bool(p < ALPHA),
            'effect_size':  round(_cohens_d(short, long_), 3),
            'group1_mean':  round(float(short.mean()), 3),  # Short
            'group2_mean':  round(float(long_.mean()), 3),  # Long
        }

    # ── Visualisation ────────────────────────────────────────────────────────

    def plot_test_results(self, results_df: pd.DataFrame, title: str = ''):
        """Bar chart of p-values — returns a matplotlib Figure."""
        if results_df is None or results_df.empty:
            return None

        _apply_dark_style()
        df = results_df.copy().sort_values('p_value')
        labels = df.apply(lambda r: f"{r['group1']} vs {r['group2']}", axis=1)
        colors = [SIG_COLOR if s else INSIG_COLOR for s in df['significant']]

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
        ax.barh(labels, df['p_value'], color=colors, edgecolor='none', height=0.5)
        ax.axvline(ALPHA, color='#FFB347', linestyle='--', linewidth=1.5,
                   label=f'α = {ALPHA}')
        ax.set_title(f'p-values — {title}')
        ax.set_xlabel('p-value')
        ax.legend()
        plt.tight_layout()
        return fig


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('nykaa_campaign_data.csv')
    tester = HypothesisTester(df)
    print(tester.test_campaign_types())
    print(tester.test_duration_groups())
    print(tester.test_languages())