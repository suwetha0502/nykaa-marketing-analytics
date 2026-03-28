import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('nykaa_campaign_data_cleaned.csv')
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['duration', 'impressions', 'clicks', 'leads', 'conversions', 
                'revenue', 'acquisition_cost', 'roi', 'engagement_score']
print("Univariate Summary Statistics:")
print(df.describe())
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

print("\nTop 5 Campaigns by ROI:")
print(df.nlargest(5, 'roi')[['campaign_id', 'campaign_type', 'roi', 'revenue', 'conversions']])

print("\nBottom 5 Campaigns by ROI:")
print(df.nsmallest(5, 'roi')[['campaign_id', 'campaign_type', 'roi', 'revenue', 'conversions']])

print("\nTop 5 Campaigns by Revenue:")
print(df.nlargest(5, 'revenue')[['campaign_id', 'campaign_type', 'revenue', 'roi']])

print("\nAverage ROI by Campaign Type:")
print(df.groupby('campaign_type')['roi'].mean().sort_values(ascending=False))

print("\nAverage ROI by Target Audience:")
print(df.groupby('target_audience')['roi'].mean().sort_values(ascending=False))

print("\nAverage ROI by Language:")
print(df.groupby('language')['roi'].mean().sort_values(ascending=False))

plt.figure(figsize=(12, 5))
df.groupby('campaign_type')['roi'].mean().sort_values().plot(kind='bar')
plt.title('Average ROI by Campaign Type')
plt.ylabel('ROI')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 5))
df.groupby('target_audience')['roi'].mean().sort_values().plot(kind='bar')
plt.title('Average ROI by Target Audience')
plt.ylabel('ROI')
plt.xticks(rotation=45)
plt.show()

df['channel_list'] = df['channel_used'].str.split(', ')
channels = df.explode('channel_list')
channel_perf = channels.groupby('channel_list').agg({
    'roi': 'mean',
    'revenue': 'sum',
    'conversions': 'sum',
    'impressions': 'sum'
}).round(2)
print("\nChannel Effectiveness:")
print(channel_perf.sort_values('roi', ascending=False))

plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title('Boxplot for Outlier Detection')
plt.show()

Q1 = df['roi'].quantile(0.25)
Q3 = df['roi'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['roi'] < (Q1 - 1.5 * IQR)) | (df['roi'] > (Q3 + 1.5 * IQR))]
print(f"\nNumber of ROI outliers detected: {len(outliers)}")