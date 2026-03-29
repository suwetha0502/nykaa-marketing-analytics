import pandas as pd
import plotly.express as px
import tkinter as tk
from tkinter import ttk
from derived_kpi import load_data, create_kpis, encode_channels, add_extra_features


def load_processed_data(path):
    return pd.read_csv(path)



def ranking_segmentation(df):
    print("\n📊 Top Campaigns by ROI:\n")
    top = df.sort_values(by="ROI", ascending=False).head(10)
    print(top[["Campaign_ID", "ROI", "Revenue", "Spend"]])

    fig = px.bar(top, x="Campaign_ID", y="ROI", title="Top Campaigns by ROI")
    fig.show()



def best_channel_mix(df):
    channel_cols = df.select_dtypes(include='number').columns.tolist()

    possible_channels = [col for col in df.columns if col not in [
        'Campaign_ID','ROI','Revenue','Spend','Clicks','Impressions',
        'Leads','Conversions','Acquisition_Cost'
    ] and df[col].dropna().isin([0,1]).all()]

    print("\n📊 Channel Performance (Avg ROI):\n")
    for ch in possible_channels:
        avg_roi = df[df[ch] == 1]["ROI"].mean()
        print(f"{ch}: {avg_roi:.2f}")



def duration_vs_performance(df):
    fig = px.scatter(
        df,
        x="Duration",
        y="ROI",
        size="Revenue",
        title="Duration vs ROI"
    )
    fig.show()



def break_even_analysis(df):
    df['Profit'] = df['Revenue'] - df['Spend']

    print("\n📊 Break-even Campaigns:\n")
    breakeven = df[df['Profit'] >= 0]
    print(breakeven[["Campaign_ID", "Profit"]].head(10))

    fig = px.histogram(df, x="Profit", title="Profit Distribution")
    fig.show()



def run_analysis(path):
    df = load_processed_data(path)

    ranking_segmentation(df)
    best_channel_mix(df)
    duration_vs_performance(df)
    break_even_analysis(df)




def show_table(df, title="Data"):
    window = tk.Toplevel()
    window.title(title)
    window.geometry("1100x500")

    frame = ttk.Frame(window)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame)
    tree.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    for _, row in df.head(100).iterrows():
        tree.insert("", "end", values=list(row))


def run_gui_pipeline(input_path):
    df = load_data(input_path)
    df = create_kpis(df)
    df = encode_channels(df)
    df = add_extra_features(df)

    root = tk.Tk()
    root.title("Marketing Analytics Dashboard")
    root.geometry("400x300")

    ttk.Label(root, text="Select View", font=("Arial", 14)).pack(pady=10)

    # Button 1: Clean Dataset
    ttk.Button(root, text="View Clean Dataset",
               command=lambda: show_table(df, "Clean Dataset")).pack(pady=5)

    # Button 2: Top ROI Campaigns
    def show_top_roi():
        top = df.sort_values(by="ROI", ascending=False).head(10)
        show_table(top[["Campaign_ID", "ROI", "Revenue", "Spend"]], "Top ROI Campaigns")

    ttk.Button(root, text="Top ROI Campaigns",
               command=show_top_roi).pack(pady=5)

    # Button 3: Break-even
    def show_breakeven():
        temp = df.copy()
        temp["Profit"] = temp["Revenue"] - temp["Spend"]
        be = temp[temp["Profit"] >= 0]
        show_table(be[["Campaign_ID", "Profit"]], "Break-even Campaigns")

    ttk.Button(root, text="Break-even Campaigns",
               command=show_breakeven).pack(pady=5)

    root.mainloop()



if __name__ == "__main__":
    run_analysis("processed_campaign_data.csv")
    run_gui_pipeline("nykaa_campaign_data.csv")