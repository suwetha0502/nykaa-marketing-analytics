"""
Data cleaning module for Nykaa Marketing Analytics.
Provides a DataCleaner class that app.py can import and use.
"""

import pandas as pd


class DataCleaner:
    """Handles all data cleaning operations."""

    NUMERIC_COLS = [
        'duration', 'impressions', 'clicks', 'leads', 'conversions',
        'revenue', 'acquisition_cost', 'roi', 'engagement_score',
    ]
    TEXT_COLS = [
        'campaign_id', 'campaign_type', 'target_audience', 'channel_used',
        'language', 'customer_segment',
    ]

    def __init__(self):
        self.df = None

    def clean_column_names(self):
        """Lowercase + underscore column names."""
        if self.df is not None:
            self.df.columns = [
                col.strip().lower().replace(' ', '_') for col in self.df.columns
            ]

    def convert_dates(self):
        """Parse date column."""
        if self.df is not None and 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(
                self.df['date'], format='%d-%m-%Y', errors='coerce'
            )

    def convert_numeric(self):
        """Coerce numeric columns."""
        if self.df is not None:
            for col in self.NUMERIC_COLS:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def clean_text_columns(self):
        """Strip whitespace from text columns."""
        if self.df is not None:
            for col in self.TEXT_COLS:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.strip()

    def handle_missing_values(self):
        """Fill missing values with median (numeric) or mode (text)."""
        if self.df is not None:
            for col in self.NUMERIC_COLS:
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            for col in self.TEXT_COLS:
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def remove_duplicates(self):
        """Drop duplicate rows."""
        if self.df is not None:
            self.df = self.df.drop_duplicates()

    def run_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps and return the cleaned DataFrame."""
        self.df = df
        self.clean_column_names()
        self.convert_dates()
        self.convert_numeric()
        self.clean_text_columns()
        self.handle_missing_values()
        self.remove_duplicates()
        return self.df


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv('nykaa_campaign_data.csv')
    cleaner = DataCleaner()
    df_clean = cleaner.run_full_pipeline(df)
    print("Shape:", df_clean.shape)
    print("\nMissing Values:\n", df_clean.isnull().sum())
    print("\nData Types:\n", df_clean.dtypes)
    df_clean.to_csv('nykaa_campaign_data_cleaned.csv', index=False)
    print("\nCleaned file saved as: nykaa_campaign_data_cleaned.csv")