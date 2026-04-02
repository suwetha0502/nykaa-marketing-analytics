"""
Exploratory Data Analysis module for Nykaa Marketing Analytics.
Provides an EDAnalyzer class that app.py can import and use.
Matplotlib rcParams are only set inside methods, not at import time.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


PALETTE = ['#6C63FF', '#00D4AA', '#FF6B9D', '#FFB347', '#64DFDF',
           '#A78BFA', '#34D399', '#F472B6']

NUMERIC_COLS = [
    'duration', 'impressions', 'clicks', 'leads',
    'conversions', 'revenue', 'acquisition_cost',
    'roi', 'engagement_score',
]


def _apply_dark_style():
    """Apply dark matplotlib theme (called lazily inside methods)."""
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


class EDAnalyzer:
    """Exploratory data analysis for campaign data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── Outlier Analysis ─────────────────────────────────────────────────────

    def outlier_analysis(self):
        """Return (fig, outlier_stats_dict) for numeric columns."""
        _apply_dark_style()
        present = [c for c in NUMERIC_COLS if c in self.df.columns]
        if not present:
            return None, {}

        fig, ax = plt.subplots(figsize=(16, 6))
        data = [self.df[c].dropna() for c in present]
        bp = ax.boxplot(
            data, patch_artist=True, notch=False, vert=True,
            medianprops=dict(color='#FFB347', linewidth=2),
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([c.replace('_', '\n') for c in present], fontsize=9)
        ax.set_title('Outlier Detection — Boxplots', fontsize=14)
        plt.tight_layout()

        # Build outlier stats dict
        outlier_stats = {}
        for col in present:
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.df[col] < q1 - 1.5 * iqr) | (self.df[col] > q3 + 1.5 * iqr)
            count = int(mask.sum())
            outlier_stats[col] = {
                'count': count,
                'percentage': count / len(self.df) * 100 if len(self.df) > 0 else 0,
            }

        return fig, outlier_stats

    # ── Time Series Analysis ─────────────────────────────────────────────────

    def time_series_analysis(self):
        """Return (plotly_fig, daily_metrics_df) or (None, None)."""
        if 'date' not in self.df.columns or not PLOTLY_AVAILABLE:
            return None, None

        daily = (
            self.df.groupby('date')
            .agg(
                avg_roi=('roi', 'mean'),
                total_revenue=('revenue', 'sum'),
                campaigns=('roi', 'count'),
            )
            .reset_index()
        )

        fig = px.line(
            daily, x='date', y='avg_roi',
            title='Average ROI Over Time',
            labels={'avg_roi': 'Average ROI', 'date': 'Date'},
            color_discrete_sequence=['#667eea'],
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        return fig, daily

    # ── Correlation Heatmap ──────────────────────────────────────────────────

    def correlation_heatmap(self):
        """Return matplotlib figure of the correlation matrix."""
        _apply_dark_style()
        present = [c for c in NUMERIC_COLS if c in self.df.columns]
        if len(present) < 2:
            return None

        corr = self.df[present].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            corr, ax=ax, annot=True, fmt='.2f',
            cmap=sns.diverging_palette(260, 10, as_cmap=True),
            linewidths=0.5, linecolor='#252840',
            square=True,
        )
        ax.set_title('Correlation Matrix', fontsize=14)
        plt.tight_layout()
        return fig

    # ── Univariate Analysis ──────────────────────────────────────────────────

    def univariate_analysis(self):
        """Return matplotlib figure of distribution histograms."""
        _apply_dark_style()
        present = [c for c in NUMERIC_COLS if c in self.df.columns]
        if not present:
            return None

        cols_per_row = 3
        rows = (len(present) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(present):
            sns.histplot(
                self.df[col].dropna(), kde=True, ax=axes[i],
                color=PALETTE[i % len(PALETTE)], edgecolor='none', alpha=0.85,
            )
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_xlabel('')

        for j in range(len(present), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('nykaa_campaign_data.csv')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    analyzer = EDAnalyzer(df)
    fig_out, stats = analyzer.outlier_analysis()
    if fig_out:
        plt.show()