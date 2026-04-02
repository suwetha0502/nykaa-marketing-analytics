"""
Performance analysis module for Nykaa Marketing Analytics.
Provides a PerformanceAnalyzer class that app.py can import and use.
All tkinter / GUI code is guarded behind __main__.
"""

import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PerformanceAnalyzer:
    """Analyzes campaign performance metrics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── Top Campaigns ────────────────────────────────────────────────────────

    def get_top_campaigns(self, metric: str = 'roi', n: int = 10) -> pd.DataFrame:
        """Return top-n campaigns sorted by *metric* descending."""
        if metric not in self.df.columns:
            return pd.DataFrame()
        cols = [c for c in ['campaign_id', metric, 'revenue', 'acquisition_cost']
                if c in self.df.columns]
        return self.df[cols].sort_values(metric, ascending=False).head(n)

    # ── Channel Mix ──────────────────────────────────────────────────────────

    def analyze_channel_mix(self) -> pd.DataFrame:
        """
        Explode comma-separated channel_used and compute per-channel stats.
        Returns a DataFrame with columns: Channel, Avg ROI, Total Revenue,
          Total Conversions, Campaigns.
        Falls back to one-hot encoded channel columns if channel_used is absent.
        """
        if 'channel_used' in self.df.columns:
            expanded = self.df.copy()
            expanded['channel_list'] = expanded['channel_used'].str.split(',')
            exploded = expanded.explode('channel_list')
            exploded['channel_list'] = exploded['channel_list'].str.strip()
            exploded = exploded[exploded['channel_list'].notna() &
                                (exploded['channel_list'] != '')]

            if 'roi' not in exploded.columns:
                return pd.DataFrame()

            agg = (
                exploded.groupby('channel_list')
                .agg(
                    **{'Avg ROI':          ('roi',         'mean'),
                       'Total Revenue':    ('revenue',     'sum') if 'revenue'     in exploded.columns else ('roi', 'count'),
                       'Total Conversions':('conversions', 'sum') if 'conversions' in exploded.columns else ('roi', 'count'),
                       'Campaigns':        ('roi',         'count')}
                )
                .reset_index()
                .rename(columns={'channel_list': 'Channel'})
                .sort_values('Avg ROI', ascending=False)
            )
            return agg

        # Fallback: binary one-hot channel columns
        possible = [
            col for col in self.df.columns
            if col not in {'campaign_id', 'roi', 'revenue', 'spend', 'clicks',
                           'impressions', 'leads', 'conversions', 'acquisition_cost'}
            and self.df[col].dropna().isin([0, 1]).all()
        ]
        if not possible or 'roi' not in self.df.columns:
            return pd.DataFrame()

        records = []
        for ch in possible:
            subset = self.df[self.df[ch] == 1]
            records.append({
                'Channel':    ch,
                'Avg ROI':    round(subset['roi'].mean(), 3),
                'Campaigns':  len(subset),
            })
        return (
            pd.DataFrame(records)
            .dropna()
            .sort_values('Avg ROI', ascending=False)
        )

    # ── Duration Effect ──────────────────────────────────────────────────────

    def analyze_duration_effect(self):
        """
        Return (plotly_fig, duration_perf_df) or (None, None).
        """
        if 'duration' not in self.df.columns or 'roi' not in self.df.columns:
            return None, None
        if not PLOTLY_AVAILABLE:
            return None, None

        bins = [0, 7, 14, 30, 60, float('inf')]
        labels = ['1-7d', '8-14d', '15-30d', '31-60d', '60d+']
        df_copy = self.df.copy()
        df_copy['duration_bucket'] = pd.cut(
            df_copy['duration'], bins=bins, labels=labels
        )
        perf = (
            df_copy.groupby('duration_bucket', observed=True)
            .agg(
                avg_roi=('roi', 'mean'),
                campaigns=('roi', 'count'),
            )
            .reset_index()
        )

        fig = px.bar(
            perf, x='duration_bucket', y='avg_roi',
            title='Average ROI by Campaign Duration',
            labels={'duration_bucket': 'Duration', 'avg_roi': 'Average ROI'},
            color='avg_roi',
            color_continuous_scale='Viridis',
            text='campaigns',
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )
        return fig, perf

    # ── Revenue Efficiency ───────────────────────────────────────────────────

    def analyze_revenue_efficiency(self):
        """
        Return (plotly_fig, summary_dict) or (None, None).
        summary_dict keys: total_profit, break_even_rate, avg_profit_margin
        """
        rev_col  = 'revenue'
        cost_col = 'acquisition_cost'

        if rev_col not in self.df.columns or cost_col not in self.df.columns:
            return None, None
        if not PLOTLY_AVAILABLE:
            return None, None

        df_copy = self.df.copy()
        df_copy['profit'] = df_copy[rev_col] - df_copy[cost_col]
        df_copy['profit_margin'] = np.where(
            df_copy[rev_col] == 0, 0,
            df_copy['profit'] / df_copy[rev_col]
        )

        fig = px.histogram(
            df_copy, x='profit',
            title='Profit Distribution',
            color_discrete_sequence=['#6C63FF'],
            nbins=40,
        )
        fig.add_vline(
            x=0, line_dash='dash', line_color='#FF6B9D',
            annotation_text='Break-even', annotation_font_color='#FF6B9D',
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
        )

        total      = len(df_copy)
        profitable = int((df_copy['profit'] >= 0).sum())
        summary = {
            'total_profit':      round(float(df_copy['profit'].sum()), 2),
            'break_even_rate':   round(profitable / total * 100, 1) if total > 0 else 0,
            'avg_profit_margin': round(float(df_copy['profit_margin'].mean()), 4),
        }
        return fig, summary


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'processed_campaign_data.csv'
    df = pd.read_csv(path)
    analyzer = PerformanceAnalyzer(df)
    print(analyzer.get_top_campaigns())
    print(analyzer.analyze_channel_mix())