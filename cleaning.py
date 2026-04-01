import pandas as pd

df = pd.read_csv('nykaa_campaign_data.csv')
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
numeric_cols = ['duration', 'impressions', 'clicks', 'leads', 'conversions', 
                'revenue', 'acquisition_cost', 'roi', 'engagement_score']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
text_cols = ['campaign_id', 'campaign_type', 'target_audience', 'channel_used', 
             'language', 'customer_segment']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
for col in text_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
df = df.drop_duplicates()

print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
df.to_csv('nykaa_campaign_data_cleaned.csv', index=False)
print("\nCleaned file saved as: nykaa_campaign_data_cleaned.csv")