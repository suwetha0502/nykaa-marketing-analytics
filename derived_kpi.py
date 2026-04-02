"""
Derived KPI module for Nykaa Marketing Analytics.
Provides a KPIGenerator class that app.py can import and use.
"""

import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import MultiLabelBinarizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ── Standalone pipeline helpers ───────────────────────────────────────────────

def _create_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    imp = df.get('impressions', pd.Series(0, index=df.index))
    clk = df.get('clicks', pd.Series(0, index=df.index))
    cost = df.get('acquisition_cost', pd.Series(0, index=df.index))
    rev = df.get('revenue', pd.Series(0, index=df.index))
    conv = df.get('conversions', pd.Series(0, index=df.index))
    leads = df.get('leads', pd.Series(0, index=df.index))

    df['ctr'] = np.where(imp == 0, 0, clk / imp)
    df['cpc'] = np.where(clk == 0, 0, cost / clk)
    df['conversion_rate'] = np.where(clk == 0, 0, conv / clk)
    df['cpl'] = np.where(leads == 0, 0, cost / leads)
    df['roas'] = np.where(cost == 0, 0, rev / cost)
    df['spend'] = cost
    return df


def _encode_channels(df: pd.DataFrame) -> pd.DataFrame:
    if 'channel_used' not in df.columns or not SKLEARN_AVAILABLE:
        return df
    df = df.copy()
    channel_series = df['channel_used'].fillna('').apply(
        lambda x: [i.strip() for i in x.split(',') if i.strip()]
    )
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(channel_series)
    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_, index=df.index)
    return pd.concat([df, encoded_df], axis=1)


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    imp = df.get('impressions', pd.Series(0, index=df.index))
    eng = df.get('engagement_score', pd.Series(0, index=df.index))
    rev = df.get('revenue', pd.Series(0, index=df.index))
    conv = df.get('conversions', pd.Series(0, index=df.index))

    df['engagement_efficiency'] = np.where(imp == 0, 0, eng / imp)
    df['revenue_per_conversion'] = np.where(conv == 0, 0, rev / conv)

    if 'roi' in df.columns:
        df['roi_category'] = pd.cut(
            df['roi'],
            bins=[-np.inf, 0, 1, 3, np.inf],
            labels=['Loss', 'Low', 'Medium', 'High'],
        )
    return df


# ── KPIGenerator class ────────────────────────────────────────────────────────

class KPIGenerator:
    """Generates derived KPIs and extra features from a campaign DataFrame."""

    def __init__(self, df: pd.DataFrame = None):
        self.df = df.copy() if df is not None else None

    def set_data(self, df: pd.DataFrame) -> 'KPIGenerator':
        self.df = df.copy()
        return self

    def create_kpis(self) -> 'KPIGenerator':
        self.df = _create_kpis(self.df)
        return self

    def encode_channels(self) -> 'KPIGenerator':
        self.df = _encode_channels(self.df)
        return self

    def add_extra_features(self) -> 'KPIGenerator':
        self.df = _add_extra_features(self.df)
        return self

    def run_full_pipeline(self) -> pd.DataFrame:
        """Run all KPI generation steps and return the processed DataFrame."""
        self.create_kpis()
        self.encode_channels()
        self.add_extra_features()
        return self.df


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'nykaa_campaign_data.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'processed_campaign_data.csv'

    raw = pd.read_csv(input_path)
    gen = KPIGenerator(raw)
    processed = gen.run_full_pipeline()

    print("\n✅ Final Dataset Preview:")
    print(processed.head())

    processed.to_csv(output_path, index=False)
    print(f"\n💾 Processed data saved to: {output_path}")