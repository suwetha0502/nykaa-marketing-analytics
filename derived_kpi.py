import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import tkinter as tk
from tkinter import ttk




def load_data(path):
    df = pd.read_csv(path)
    return df



def create_kpis(df):
    df['CTR'] = np.where(df['Impressions'] == 0, 0, df['Clicks'] / df['Impressions'])

    df['CPC'] = np.where(df['Clicks'] == 0, 0, df['Acquisition_Cost'] / df['Clicks'])

    df['Conversion_Rate'] = np.where(df['Clicks'] == 0, 0, df['Conversions'] / df['Clicks'])

    df['CPL'] = np.where(df['Leads'] == 0, 0, df['Acquisition_Cost'] / df['Leads'])

    df['ROAS'] = np.where(df['Acquisition_Cost'] == 0, 0, df['Revenue'] / df['Acquisition_Cost'])

    df['Spend'] = df['Acquisition_Cost']

    return df



def encode_channels(df):
    df['Channel_Used'] = df['Channel_Used'].fillna('').apply(
        lambda x: [i.strip() for i in x.split(',') if i.strip() != '']
    )

    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df['Channel_Used'])

    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_, index=df.index)

    df = pd.concat([df, encoded_df], axis=1)

    return df



def add_extra_features(df):
    df['Engagement_Efficiency'] = np.where(
        df['Impressions'] == 0, 0, df['Engagement_Score'] / df['Impressions']
    )

    df['Revenue_per_Conversion'] = np.where(
        df['Conversions'] == 0, 0, df['Revenue'] / df['Conversions']
    )

    df['ROI_Category'] = pd.cut(
        df['ROI'],
        bins=[-np.inf, 0, 1, 3, np.inf],
        labels=['Loss', 'Low', 'Medium', 'High']
    )

    return df


def save_data(df, path):
    df.to_csv(path, index=False)



def run_pipeline(input_path, output_path=None):
    df = load_data(input_path)

    df = create_kpis(df)
    df = encode_channels(df)
    df = add_extra_features(df)

    print("\n✅ Final Dataset Preview:")
    print(df.head())

    if output_path:
        save_data(df, output_path)
        print(f"\n💾 Processed data saved to: {output_path}")

    return df


def run_pipeline_gui(input_path):
    df = load_data(input_path)
    df = create_kpis(df)
    df = encode_channels(df)
    df = add_extra_features(df)

    # Create GUI window
    root = tk.Tk()
    root.title("Marketing Data Preview")
    root.geometry("1000x500")

    # Create table
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame)
    tree.pack(fill="both", expand=True)

    # Add scrollbar
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    # Set columns
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    # Insert rows (limit for performance)
    for i, row in df.head(100).iterrows():
        tree.insert("", "end", values=list(row))

    root.mainloop()


# ENTRY POINT
if __name__ == "__main__":
    run_pipeline_gui("nykaa_campaign_data.csv")
    run_pipeline(
        input_path="nykaa_campaign_data.csv",
        output_path="processed_campaign_data.csv"
    )