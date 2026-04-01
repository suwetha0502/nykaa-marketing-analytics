import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import tkinter as tk
from tkinter import ttk, font
import tkinter.messagebox as messagebox


# ─────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────
BG_DARK    = "#0D0F1A"
BG_CARD    = "#141728"
BG_SIDEBAR = "#0A0C16"
ACCENT1    = "#6C63FF"   # violet
ACCENT2    = "#00D4AA"   # teal
ACCENT3    = "#FF6B9D"   # pink
TEXT_PRI   = "#F0F2FF"
TEXT_SEC   = "#8A8FAD"
BORDER     = "#252840"
ROW_ALT    = "#181A2E"


# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"]             = np.where(df["Impressions"] == 0, 0, df["Clicks"] / df["Impressions"])
    df["CPC"]             = np.where(df["Clicks"] == 0, 0, df["Acquisition_Cost"] / df["Clicks"])
    df["Conversion_Rate"] = np.where(df["Clicks"] == 0, 0, df["Conversions"] / df["Clicks"])
    df["CPL"]             = np.where(df["Leads"] == 0, 0, df["Acquisition_Cost"] / df["Leads"])
    df["ROAS"]            = np.where(df["Acquisition_Cost"] == 0, 0, df["Revenue"] / df["Acquisition_Cost"])
    df["Spend"]           = df["Acquisition_Cost"]
    return df


def encode_channels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Channel_Used"] = df["Channel_Used"].fillna("").apply(
        lambda x: [i.strip() for i in x.split(",") if i.strip()]
    )
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df["Channel_Used"])
    encoded_df = pd.DataFrame(encoded, columns=mlb.classes_, index=df.index)
    return pd.concat([df, encoded_df], axis=1)


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Engagement_Efficiency"]  = np.where(df["Impressions"] == 0, 0, df["Engagement_Score"] / df["Impressions"])
    df["Revenue_per_Conversion"] = np.where(df["Conversions"] == 0, 0, df["Revenue"] / df["Conversions"])
    df["ROI_Category"] = pd.cut(
        df["ROI"],
        bins=[-np.inf, 0, 1, 3, np.inf],
        labels=["Loss", "Low", "Medium", "High"]
    )
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def run_pipeline(input_path: str, output_path: str | None = None) -> pd.DataFrame:
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


# ─────────────────────────────────────────────
# STYLED TABLE WIDGET
# ─────────────────────────────────────────────
def build_table(parent: tk.Widget, df: pd.DataFrame) -> None:
    """Render a sleek dark-themed treeview table inside *parent*."""
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "KPI.Treeview",
        background=BG_CARD,
        foreground=TEXT_PRI,
        rowheight=30,
        fieldbackground=BG_CARD,
        bordercolor=BORDER,
        borderwidth=0,
        font=("Courier New", 11),
    )
    style.configure(
        "KPI.Treeview.Heading",
        background=ACCENT1,
        foreground=TEXT_PRI,
        font=("Courier New", 11, "bold"),
        relief="flat",
    )
    style.map("KPI.Treeview", background=[("selected", ACCENT1)])
    style.map("KPI.Treeview.Heading", background=[("active", ACCENT2)])

    frame = tk.Frame(parent, bg=BG_DARK)
    frame.pack(fill="both", expand=True, padx=16, pady=10)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    hsb = ttk.Scrollbar(frame, orient="horizontal")

    tree = ttk.Treeview(
        frame,
        style="KPI.Treeview",
        yscrollcommand=vsb.set,
        xscrollcommand=hsb.set,
        selectmode="browse",
    )
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    cols = list(df.columns)
    tree["columns"] = cols
    tree["show"] = "headings"
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=max(120, len(col) * 10), anchor="center")

    for idx, (_, row) in enumerate(df.head(100).iterrows()):
        tag = "odd" if idx % 2 else "even"
        tree.insert("", "end", values=list(row), tags=(tag,))

    tree.tag_configure("odd",  background=ROW_ALT)
    tree.tag_configure("even", background=BG_CARD)


# ─────────────────────────────────────────────
# MAIN GUI
# ─────────────────────────────────────────────
def run_pipeline_gui(input_path: str) -> None:
    df = load_data(input_path)
    df = create_kpis(df)
    df = encode_channels(df)
    df = add_extra_features(df)

    root = tk.Tk()
    root.title("KPI Pipeline — Marketing Intelligence")
    root.configure(bg=BG_DARK)
    root.geometry("1200x720")
    root.minsize(900, 560)

    # ── Header bar ───────────────────────────
    header = tk.Frame(root, bg=BG_SIDEBAR, height=64)
    header.pack(fill="x")
    header.pack_propagate(False)

    tk.Label(
        header,
        text="◈  KPI PIPELINE",
        bg=BG_SIDEBAR, fg=ACCENT1,
        font=("Courier New", 18, "bold"),
    ).pack(side="left", padx=24, pady=14)

    tk.Label(
        header,
        text=f"{len(df):,} records loaded",
        bg=BG_SIDEBAR, fg=TEXT_SEC,
        font=("Courier New", 11),
    ).pack(side="right", padx=24)

    # ── Stat badges ──────────────────────────
    badge_bar = tk.Frame(root, bg=BG_DARK)
    badge_bar.pack(fill="x", padx=16, pady=10)

    stats = [
        ("Avg ROI",    f"{df['ROI'].mean():.2f}",         ACCENT1),
        ("Avg ROAS",   f"{df['ROAS'].mean():.2f}",         ACCENT2),
        ("Avg CTR",    f"{df['CTR'].mean()*100:.2f}%",     ACCENT3),
        ("High ROI%",  f"{(df['ROI_Category']=='High').mean()*100:.1f}%", "#FFB347"),
    ]
    for label, value, color in stats:
        card = tk.Frame(badge_bar, bg=BG_CARD, padx=20, pady=12, relief="flat")
        card.pack(side="left", padx=8, ipadx=4)
        tk.Label(card, text=value, bg=BG_CARD, fg=color,
                 font=("Courier New", 20, "bold")).pack()
        tk.Label(card, text=label, bg=BG_CARD, fg=TEXT_SEC,
                 font=("Courier New", 10)).pack()

    # ── Tab strip ────────────────────────────
    tab_bar = tk.Frame(root, bg=BG_SIDEBAR, height=44)
    tab_bar.pack(fill="x")
    tab_bar.pack_propagate(False)

    content_area = tk.Frame(root, bg=BG_DARK)
    content_area.pack(fill="both", expand=True)

    views = {
        "Full Dataset":   df,
        "KPIs Only":      df[["Campaign_ID", "CTR", "CPC", "Conversion_Rate", "CPL", "ROAS", "Spend"]],
        "Enriched":       df[["Campaign_ID", "Engagement_Efficiency", "Revenue_per_Conversion", "ROI_Category"]],
    }

    active_tab = {"name": None}

    def switch_tab(name: str, btn_ref: dict) -> None:
        if active_tab["name"] == name:
            return
        active_tab["name"] = name
        for b_name, b_widget in btn_ref.items():
            b_widget.config(
                bg=ACCENT1 if b_name == name else BG_SIDEBAR,
                fg=TEXT_PRI,
            )
        for w in content_area.winfo_children():
            w.destroy()
        build_table(content_area, views[name])

    btn_refs: dict[str, tk.Label] = {}
    for v_name in views:
        btn = tk.Label(
            tab_bar,
            text=v_name,
            bg=BG_SIDEBAR, fg=TEXT_SEC,
            font=("Courier New", 11),
            padx=18, pady=10, cursor="hand2",
        )
        btn.pack(side="left")
        btn_refs[v_name] = btn
        btn.bind("<Button-1>", lambda e, n=v_name: switch_tab(n, btn_refs))

    switch_tab("Full Dataset", btn_refs)
    root.mainloop()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline_gui("nykaa_campaign_data.csv")
    run_pipeline(
        input_path="nykaa_campaign_data.csv",
        output_path="processed_campaign_data.csv",
    )