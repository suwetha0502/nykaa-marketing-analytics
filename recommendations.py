"""
Recommendations module for Nykaa Marketing Analytics.
Provides a RecommendationEngine class that app.py can import and use.
All tkinter / PuLP GUI code is guarded behind __main__.
"""

import pandas as pd
import numpy as np


class RecommendationEngine:
    """Generates strategic recommendations from campaign data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ── ROI Distribution ─────────────────────────────────────────────────────

    def analyze_roi_distribution(self) -> dict:
        """
        Return summary stats about the ROI distribution.
        Keys: mean_roi, median_roi, high_performers, low_performers,
              high_performer_revenue, std_roi
        """
        if 'roi' not in self.df.columns:
            return {}

        mean_roi = float(self.df['roi'].mean())
        high_mask = self.df['roi'] > mean_roi
        low_mask  = self.df['roi'] <= mean_roi

        rev_col = 'revenue' if 'revenue' in self.df.columns else None

        return {
            'mean_roi':              round(mean_roi, 3),
            'median_roi':            round(float(self.df['roi'].median()), 3),
            'std_roi':               round(float(self.df['roi'].std()), 3),
            'high_performers':       int(high_mask.sum()),
            'low_performers':        int(low_mask.sum()),
            'high_performer_revenue': (
                round(float(self.df.loc[high_mask, rev_col].sum()), 2)
                if rev_col else 0
            ),
        }

    # ── Budget Recommendations ───────────────────────────────────────────────

    def generate_budget_recommendations(self) -> list:
        """
        Return a list of recommendation dicts with keys:
          category, recommendation, impact, action
        """
        if 'roi' not in self.df.columns:
            return []

        recs = []
        mean_roi = float(self.df['roi'].mean())

        # High ROI campaigns
        high = self.df[self.df['roi'] > mean_roi]
        recs.append({
            'category':       'Scale Up',
            'recommendation': f'Increase budget for {len(high)} high-ROI campaigns',
            'impact':         'High',
            'action':         f'Campaigns with ROI > {mean_roi:.2f} — allocate 20-30% more spend',
        })

        # Low ROI campaigns
        low = self.df[self.df['roi'] <= mean_roi]
        recs.append({
            'category':       'Cut Spend',
            'recommendation': f'Reduce or pause {len(low)} under-performing campaigns',
            'impact':         'Medium',
            'action':         'Reallocate budget from low-ROI campaigns to high-ROI ones',
        })

        # Best campaign type
        if 'campaign_type' in self.df.columns:
            type_roi = self.df.groupby('campaign_type')['roi'].mean()
            best_type  = type_roi.idxmax()
            worst_type = type_roi.idxmin()
            recs.append({
                'category':       'Campaign Type',
                'recommendation': f"Prioritise '{best_type}' campaign type",
                'impact':         'High',
                'action':         f"'{best_type}' has the highest avg ROI ({type_roi[best_type]:.2f})",
            })
            recs.append({
                'category':       'Campaign Type',
                'recommendation': f"Review '{worst_type}' campaign type",
                'impact':         'Medium',
                'action':         f"'{worst_type}' has the lowest avg ROI ({type_roi[worst_type]:.2f})",
            })

        # Spend efficiency
        if 'acquisition_cost' in self.df.columns and 'revenue' in self.df.columns:
            total_spend = float(self.df['acquisition_cost'].sum())
            total_rev   = float(self.df['revenue'].sum())
            overall_roas = total_rev / total_spend if total_spend > 0 else 0
            recs.append({
                'category':       'Portfolio',
                'recommendation': f'Overall ROAS = {overall_roas:.2f}x',
                'impact':         'Informational',
                'action':         (
                    'Target channels with ROAS > portfolio average to maximise revenue'
                ),
            })

        return recs

    # ── Budget Allocation Table ──────────────────────────────────────────────

    def budget_allocation(self, total_budget: float = 100_000) -> pd.DataFrame:
        """ROI-weighted budget allocation across campaign types."""
        if 'roi' not in self.df.columns:
            return pd.DataFrame()

        group_col = 'campaign_type' if 'campaign_type' in self.df.columns else None
        if group_col:
            agg = (
                self.df.groupby(group_col)
                .agg(Avg_ROI=('roi', 'mean'),
                     Total_Revenue=('revenue', 'sum') if 'revenue' in self.df.columns
                     else ('roi', 'count'))
                .reset_index()
            )
            agg['Weight']           = agg['Avg_ROI'] / agg['Avg_ROI'].sum()
            agg['Allocated_Budget'] = (agg['Weight'] * total_budget).round(2)
            return agg.sort_values('Allocated_Budget', ascending=False)

        # No campaign_type: allocate per-campaign
        temp = self.df.copy()
        temp['Weight']           = temp['roi'] / temp['roi'].sum()
        temp['Allocated_Budget'] = (temp['Weight'] * total_budget).round(2)
        cols = [c for c in ['campaign_id', 'roi', 'Allocated_Budget']
                if c in temp.columns]
        return temp[cols].sort_values('Allocated_Budget', ascending=False)

    # ── What-If Analysis ─────────────────────────────────────────────────────

    def what_if_analysis(self, spend_uplift: float = 1.20) -> pd.DataFrame:
        """Simulate expected revenue with a spend uplift factor."""
        required = {'acquisition_cost', 'revenue', 'roas'}
        if not required.issubset(self.df.columns):
            return pd.DataFrame()

        temp = self.df.copy()
        temp['new_spend']        = (temp['acquisition_cost'] * spend_uplift).round(2)
        temp['expected_revenue'] = (temp['new_spend'] * temp['roas']).round(2)
        temp['revenue_delta']    = (temp['expected_revenue'] - temp['revenue']).round(2)

        cols = [c for c in
                ['campaign_id', 'acquisition_cost', 'new_spend',
                 'revenue', 'expected_revenue', 'revenue_delta']
                if c in temp.columns]
        return temp[cols].head(20)


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    try:
        from pulp import LpMaximize, LpProblem, LpVariable, lpSum
        PULP_AVAILABLE = True
    except ImportError:
        PULP_AVAILABLE = False

    path = sys.argv[1] if len(sys.argv) > 1 else 'processed_campaign_data.csv'
    df = pd.read_csv(path)

    engine = RecommendationEngine(df)
    print("ROI Analysis:", engine.analyze_roi_distribution())
    print("\nBudget Recommendations:")
    for r in engine.generate_budget_recommendations():
        print(f"  [{r['category']}] {r['recommendation']}")
    print("\nBudget Allocation:")
    print(engine.budget_allocation())