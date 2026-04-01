import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk
from derived_kpi import load_data, create_kpis, encode_channels, add_extra_features


# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
BG_DARK    = "#0D0F1A"
BG_CARD    = "#141728"
BG_SIDEBAR = "#0A0C16"
ACCENT1    = "#6C63FF"
ACCENT2    = "#00D4AA"
ACCENT3    = "#FF6B9D"
TEXT_PRI   = "#F0F2FF"
TEXT_SEC   = "#8A8FAD"
BORDER     = "#252840"
ROW_ALT    = "#181A2E"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#141728",
        plot_bgcolor="#0D0F1A",
        font=dict(color="#F0F2FF", family="Courier New"),
        title_font=dict(size=16, color="#6C63FF"),
        xaxis=dict(gridcolor="#252840", zeroline=False),
        yaxis=dict(gridcolor="#252840", zeroline=False),
        colorway=["#6C63FF", "#00D4AA", "#FF6B9D", "#FFB347", "#64DFDF"],
    )
)


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ranking_segmentation(df: pd.DataFrame) -> None:
    top = df.sort_values("ROI", ascending=False).head(10)
    print("\n📊 Top Campaigns by ROI:\n")
    print(top[["Campaign_ID", "ROI", "Revenue", "Spend"]])

    fig = px.bar(
        top, x="Campaign_ID", y="ROI",
        title="Top 10 Campaigns by ROI",
        color="ROI", color_continuous_scale="Viridis",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker_line_width=0)
    fig.show()


def best_channel_mix(df: pd.DataFrame) -> None:
    possible_channels = [
        col for col in df.columns
        if col not in {"Campaign_ID", "ROI", "Revenue", "Spend", "Clicks",
                       "Impressions", "Leads", "Conversions", "Acquisition_Cost"}
        and df[col].dropna().isin([0, 1]).all()
    ]
    results = {ch: df[df[ch] == 1]["ROI"].mean() for ch in possible_channels}
    ch_df = pd.DataFrame(results.items(), columns=["Channel", "Avg_ROI"]).dropna()

    print("\n📊 Channel Performance (Avg ROI):\n")
    print(ch_df.sort_values("Avg_ROI", ascending=False).to_string(index=False))

    if not ch_df.empty:
        fig = px.bar(
            ch_df.sort_values("Avg_ROI"),
            x="Avg_ROI", y="Channel", orientation="h",
            title="Average ROI by Channel",
            color="Avg_ROI", color_continuous_scale="Teal",
            template=PLOTLY_TEMPLATE,
        )
        fig.show()


def duration_vs_performance(df: pd.DataFrame) -> None:
    fig = px.scatter(
        df, x="Duration", y="ROI", size="Revenue",
        title="Campaign Duration vs ROI  (bubble = Revenue)",
        color="ROI", color_continuous_scale="Plasma",
        template=PLOTLY_TEMPLATE,
        opacity=0.75,
    )
    fig.update_traces(marker_line_width=0)
    fig.show()


def break_even_analysis(df: pd.DataFrame) -> None:
    temp = df.copy()
    temp["Profit"] = temp["Revenue"] - temp["Spend"]
    breakeven = temp[temp["Profit"] >= 0]

    print("\n📊 Break-even Campaigns:\n")
    print(breakeven[["Campaign_ID", "Profit"]].head(10).to_string(index=False))

    fig = px.histogram(
        temp, x="Profit",
        title="Profit Distribution",
        color_discrete_sequence=[ACCENT1],
        template=PLOTLY_TEMPLATE,
        nbins=40,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=ACCENT3,
                  annotation_text="Break-even", annotation_font_color=ACCENT3)
    fig.show()


def run_analysis(path: str) -> None:
    df = load_processed_data(path)
    ranking_segmentation(df)
    best_channel_mix(df)
    duration_vs_performance(df)
    break_even_analysis(df)


# ─────────────────────────────────────────────
# STYLED TABLE HELPER
# ─────────────────────────────────────────────
def _apply_tree_style(tag: str = "PA") -> None:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        f"{tag}.Treeview",
        background=BG_CARD,
        foreground=TEXT_PRI,
        rowheight=30,
        fieldbackground=BG_CARD,
        borderwidth=0,
        font=("Courier New", 11),
    )
    style.configure(
        f"{tag}.Treeview.Heading",
        background=ACCENT1,
        foreground=TEXT_PRI,
        font=("Courier New", 11, "bold"),
        relief="flat",
    )
    style.map(f"{tag}.Treeview", background=[("selected", ACCENT2)])


def show_table(df: pd.DataFrame, title: str = "Data", parent: tk.Misc | None = None) -> None:
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("1160x520")
    win.configure(bg=BG_DARK)

    # Header
    hdr = tk.Frame(win, bg=BG_SIDEBAR, height=52)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text=f"◈  {title}", bg=BG_SIDEBAR, fg=ACCENT1,
             font=("Courier New", 14, "bold")).pack(side="left", padx=20, pady=12)
    tk.Label(hdr, text=f"{len(df):,} rows", bg=BG_SIDEBAR, fg=TEXT_SEC,
             font=("Courier New", 10)).pack(side="right", padx=20)

    _apply_tree_style()
    frame = tk.Frame(win, bg=BG_DARK)
    frame.pack(fill="both", expand=True, padx=14, pady=12)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    hsb = ttk.Scrollbar(frame, orient="horizontal")
    tree = ttk.Treeview(frame, style="PA.Treeview",
                        yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)
    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=max(120, len(col) * 10), anchor="center")
    for idx, (_, row) in enumerate(df.head(100).iterrows()):
        tree.insert("", "end", values=list(row),
                    tags=("odd" if idx % 2 else "even",))
    tree.tag_configure("odd",  background=ROW_ALT)
    tree.tag_configure("even", background=BG_CARD)


# ─────────────────────────────────────────────
# MAIN DASHBOARD GUI
# ─────────────────────────────────────────────
def run_gui_pipeline(input_path: str) -> None:
    df = load_data(input_path)
    df = create_kpis(df)
    df = encode_channels(df)
    df = add_extra_features(df)

    root = tk.Tk()
    root.title("Performance Analysis — Marketing Intelligence")
    root.configure(bg=BG_DARK)
    root.geometry("520x560")
    root.resizable(False, False)

    # Header
    hdr = tk.Frame(root, bg=BG_SIDEBAR, height=68)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="◈  PERFORMANCE ANALYSIS",
             bg=BG_SIDEBAR, fg=ACCENT1,
             font=("Courier New", 16, "bold")).pack(side="left", padx=24, pady=18)

    # KPI strip
    kpi_bar = tk.Frame(root, bg=BG_DARK)
    kpi_bar.pack(fill="x", padx=16, pady=14)
    kpis = [
        ("Campaigns", f"{len(df):,}", ACCENT1),
        ("Avg ROI",   f"{df['ROI'].mean():.2f}", ACCENT2),
        ("Avg ROAS",  f"{df['ROAS'].mean():.2f}", ACCENT3),
    ]
    for label, val, color in kpis:
        card = tk.Frame(kpi_bar, bg=BG_CARD, padx=16, pady=10)
        card.pack(side="left", padx=6)
        tk.Label(card, text=val,   bg=BG_CARD, fg=color,
                 font=("Courier New", 18, "bold")).pack()
        tk.Label(card, text=label, bg=BG_CARD, fg=TEXT_SEC,
                 font=("Courier New", 9)).pack()

    # Divider
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=16)

    # Button definitions
    def btn_section(parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text, bg=BG_DARK, fg=TEXT_SEC,
                 font=("Courier New", 9, "bold")).pack(anchor="w", padx=24, pady=(14, 2))

    def add_btn(parent: tk.Widget, label: str, icon: str,
                cmd, color: str = ACCENT1) -> None:
        f = tk.Frame(parent, bg=BG_CARD, cursor="hand2")
        f.pack(fill="x", padx=16, pady=4)
        accent_bar = tk.Frame(f, bg=color, width=4)
        accent_bar.pack(side="left", fill="y")
        inner = tk.Frame(f, bg=BG_CARD, pady=12, padx=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(inner, text=f"{icon}  {label}", bg=BG_CARD, fg=TEXT_PRI,
                 font=("Courier New", 12, "bold"), anchor="w").pack(fill="x")
        for widget in (f, inner, accent_bar):
            widget.bind("<Button-1>", lambda e, c=cmd: c())
            widget.bind("<Enter>", lambda e, fr=f: fr.config(bg="#1E2135"))
            widget.bind("<Leave>", lambda e, fr=f: fr.config(bg=BG_CARD))

    btn_section(root, "TABLES")
    add_btn(root, "Full Dataset",     "◉", lambda: show_table(df, "Full Dataset", root),           ACCENT1)
    add_btn(root, "Top ROI Campaigns","▲", lambda: show_table(
        df.sort_values("ROI", ascending=False).head(10)[["Campaign_ID", "ROI", "Revenue", "Spend"]],
        "Top ROI Campaigns", root), ACCENT2)

    def show_breakeven():
        temp = df.copy()
        temp["Profit"] = temp["Revenue"] - temp["Spend"]
        show_table(temp[temp["Profit"] >= 0][["Campaign_ID", "Profit"]], "Break-even Campaigns", root)

    add_btn(root, "Break-even Campaigns", "◆", show_breakeven, ACCENT3)

    btn_section(root, "CHARTS")
    add_btn(root, "ROI Ranking Chart",       "📊", lambda: ranking_segmentation(df),   ACCENT1)
    add_btn(root, "Duration vs ROI Scatter", "📈", lambda: duration_vs_performance(df), ACCENT2)
    add_btn(root, "Profit Distribution",     "💰", lambda: break_even_analysis(df),     ACCENT3)

    root.mainloop()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_analysis("processed_campaign_data.csv")
    run_gui_pipeline("nykaa_campaign_data.csv")