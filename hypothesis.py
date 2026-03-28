import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('nykaa_campaign_data_cleaned.csv')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("A/B Testing - Statistical Significance\n")
campaign_types = df['campaign_type'].unique()
for i in range(len(campaign_types)):
    for j in range(i+1, len(campaign_types)):
        group1 = df[df['campaign_type'] == campaign_types[i]]['roi']
        group2 = df[df['campaign_type'] == campaign_types[j]]['roi']
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        print(f"{campaign_types[i]} vs {campaign_types[j]} → p-value = {p_value:.4f}  {'Significant' if p_value < 0.05 else 'Not Significant'}")

languages = df['language'].unique()
for i in range(len(languages)):
    for j in range(i+1, len(languages)):
        group1 = df[df['language'] == languages[i]]['roi']
        group2 = df[df['language'] == languages[j]]['roi']
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        print(f"{languages[i]} vs {languages[j]} → p-value = {p_value:.4f}  {'Significant' if p_value < 0.05 else 'Not Significant'}")

median_duration = df['duration'].median()
df['duration_group'] = np.where(df['duration'] <= median_duration, 'Short', 'Long')

short = df[df['duration_group'] == 'Short']['roi']
long = df[df['duration_group'] == 'Long']['roi']
t_stat, p_value = stats.ttest_ind(short, long, equal_var=False)
print(f"\nShort Duration vs Long Duration → p-value = {p_value:.4f}  {'Significant' if p_value < 0.05 else 'Not Significant'}")

print("\nMean ROI by Campaign Type:")
print(df.groupby('campaign_type')['roi'].agg(['mean', 'count', 'std']).round(3))

print("\nMean ROI by Language:")
print(df.groupby('language')['roi'].agg(['mean', 'count', 'std']).round(3))

print("\nMean ROI by Duration Group:")
print(df.groupby('duration_group')['roi'].agg(['mean', 'count', 'std']).round(3))

plt.figure(figsize=(12, 5))
sns.boxplot(x='campaign_type', y='roi', data=df)
plt.title('ROI Distribution by Campaign Type')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='language', y='roi', data=df)
plt.title('ROI Distribution by Language')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='duration_group', y='roi', data=df)
plt.title('ROI Distribution: Short vs Long Duration')
plt.show()

print("\nA/B Testing Completed!")