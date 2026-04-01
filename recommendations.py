import pandas as pd
import tkinter as tk
from tkinter import ttk
import math

try:
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
BG_DARK    = "#0D0F1A"
BG_CARD    = "#141728"
BG_SIDEBAR = "#0A0C16"
ACCENT1    = "#6C63FF"   # violet  — primary
ACCENT2    = "#00D4AA"   # teal    — positive
ACCENT3    = "#FF6B9D"   # pink    — warning / reduce
ACCENT4    = "#FFB347"   # amber   — neutral highlight
TEXT_PRI   = "#F0F2FF"
TEXT_SEC   = "#8A8FAD"
BORDER     = "#252840"
ROW_ALT    = "#181A2E"
HOVER      = "#1E2135"


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def budget_allocation(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("Campaign_Type")
          .agg(Avg_ROI=("ROI", "mean"), Total_Revenue=("Revenue", "sum"),
               Total_Spend=("Spend", "sum"))
          .reset_index()
    )
    result["Suggested_%"] = (result["Avg_ROI"] / result["Avg_ROI"].sum() * 100).round(2)
    return result.sort_values("Suggested_%", ascending=False)


def what_if_analysis(df: pd.DataFrame, uplift: float = 1.20) -> pd.DataFrame:
    temp = df.copy()
    temp["New_Spend"]        = (temp["Spend"] * uplift).round(2)
    temp["Expected_Revenue"] = (temp["New_Spend"] * temp["ROAS"]).round(2)
    temp["Revenue_Delta"]    = (temp["Expected_Revenue"] - temp["Revenue"]).round(2)
    return temp[["Campaign_ID", "Spend", "New_Spend",
                 "Revenue", "Expected_Revenue", "Revenue_Delta"]].head(20)


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    mean_roi = df["ROI"].mean()
    high = df[df["ROI"] > mean_roi]
    low  = df[df["ROI"] < mean_roi]
    best_type  = df.groupby("Campaign_Type")["ROI"].mean().idxmax()
    worst_type = df.groupby("Campaign_Type")["ROI"].mean().idxmin()

    insights = [
        ("🟢 Scale Up",    f"Increase budget for {len(high)} campaigns with above-average ROI ({mean_roi:.2f})"),
        ("🔴 Cut Spend",   f"Reduce spend on {len(low)} under-performing campaigns"),
        ("🏆 Best Type",   f"'{best_type}' yields the highest average ROI — prioritise it"),
        ("⚠️  Worst Type", f"'{worst_type}' is the lowest-performing campaign type"),
        ("📊 Avg ROI",     f"Portfolio average ROI = {mean_roi:.3f}"),
        ("💰 Total Spend", f"Total spend = ${df['Spend'].sum():,.0f}"),
        ("📈 Total Rev",   f"Total revenue = ${df['Revenue'].sum():,.0f}"),
    ]
    return pd.DataFrame(insights, columns=["Tag", "Recommendation"])


def simple_optimization(df: pd.DataFrame, total_budget: float = 100_000) -> pd.DataFrame:
    temp = df.copy()
    temp["Weight"]           = temp["ROI"] / temp["ROI"].sum()
    temp["Allocated_Budget"] = (temp["Weight"] * total_budget).round(2)
    return (
        temp[["Campaign_ID", "ROI", "Allocated_Budget"]]
          .sort_values("Allocated_Budget", ascending=False)
    )


def optimize_with_pulp(df: pd.DataFrame, total_budget: float = 100_000) -> pd.DataFrame:
    if not PULP_AVAILABLE:
        return pd.DataFrame({"Message": ["PuLP not installed — run: pip install pulp"]})

    temp = df.copy().reset_index(drop=True)
    campaigns = temp["Campaign_ID"].tolist()
    temp["ROI_norm"] = temp["ROI"] / temp["ROI"].max()

    prob = LpProblem("Budget_Optimization", LpMaximize)
    budget_vars = {
        c: LpVariable(f"b_{i}", lowBound=1_000, upBound=total_budget * 0.3)
        for i, c in enumerate(campaigns)
    }
    prob += lpSum(temp.loc[i, "ROI_norm"] * budget_vars[campaigns[i]]
                  for i in range(len(campaigns)))
    prob += lpSum(budget_vars[c] for c in campaigns) == total_budget
    prob.solve()

    rows = [(c, round(budget_vars[c].value() or 0, 2)) for c in campaigns]
    return pd.DataFrame(rows, columns=["Campaign_ID", "Allocated_Budget"])\
             .sort_values("Allocated_Budget", ascending=False)


# ─────────────────────────────────────────────
# STYLED TABLE WIDGET
# ─────────────────────────────────────────────
def _apply_tree_style(tag: str) -> None:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(f"{tag}.Treeview",
                    background=BG_CARD, foreground=TEXT_PRI, rowheight=30,
                    fieldbackground=BG_CARD, borderwidth=0,
                    font=("Courier New", 11))
    style.configure(f"{tag}.Treeview.Heading",
                    background=ACCENT1, foreground=TEXT_PRI,
                    font=("Courier New", 11, "bold"), relief="flat")
    style.map(f"{tag}.Treeview", background=[("selected", ACCENT2)])


def show_table(parent_frame: tk.Widget, df: pd.DataFrame) -> None:
    """Clear *parent_frame* and render a styled treeview."""
    for w in parent_frame.winfo_children():
        w.destroy()

    _apply_tree_style("REC")
    outer = tk.Frame(parent_frame, bg=BG_DARK)
    outer.pack(fill="both", expand=True, padx=16, pady=12)

    vsb = ttk.Scrollbar(outer, orient="vertical")
    hsb = ttk.Scrollbar(outer, orient="horizontal")
    tree = ttk.Treeview(outer, style="REC.Treeview",
                        yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)
    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    cols = list(df.columns)
    tree["columns"] = cols
    tree["show"]    = "headings"
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=max(140, len(col) * 11), anchor="center")
    for idx, (_, row) in enumerate(df.head(50).iterrows()):
        tree.insert("", "end", values=list(row),
                    tags=("odd" if idx % 2 else "even",))
    tree.tag_configure("odd",  background=ROW_ALT)
    tree.tag_configure("even", background=BG_CARD)


# ─────────────────────────────────────────────
# SIDEBAR BUTTON FACTORY
# ─────────────────────────────────────────────
def _sidebar_btn(sidebar: tk.Widget, label: str, icon: str,
                 cmd, accent: str = ACCENT1) -> None:
    f = tk.Frame(sidebar, bg=BG_SIDEBAR, cursor="hand2")
    f.pack(fill="x", padx=10, pady=3)

    bar = tk.Frame(f, bg=accent, width=3)
    bar.pack(side="left", fill="y")

    inner = tk.Frame(f, bg=BG_SIDEBAR, pady=12, padx=12)
    inner.pack(side="left", fill="x", expand=True)

    tk.Label(inner, text=icon, bg=BG_SIDEBAR, fg=accent,
             font=("Courier New", 14)).pack(side="left")
    tk.Label(inner, text=f"  {label}", bg=BG_SIDEBAR, fg=TEXT_PRI,
             font=("Courier New", 11, "bold"), anchor="w").pack(side="left", fill="x")

    def on_enter(e):
        f.config(bg=HOVER)
        inner.config(bg=HOVER)
    def on_leave(e):
        f.config(bg=BG_SIDEBAR)
        inner.config(bg=BG_SIDEBAR)

    for w in (f, inner, bar):
        w.bind("<Button-1>", lambda e, c=cmd: c())
        w.bind("<Enter>", on_enter)
        w.bind("<Leave>", on_leave)


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
def run_dashboard(path: str) -> None:
    df = load_data(path)

    root = tk.Tk()
    root.title("Marketing Decision Intelligence")
    root.configure(bg=BG_DARK)

    try:
        import ctypes
        user32 = ctypes.windll.user32
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        root.geometry(f"{w}x{h}")
    except Exception:
        root.geometry("1400x860")

    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # ── Sidebar ───────────────────────────────
    sidebar = tk.Frame(root, bg=BG_SIDEBAR, width=260)
    sidebar.grid(row=0, column=0, sticky="nsew")
    sidebar.pack_propagate(False)

    logo = tk.Frame(sidebar, bg=BG_SIDEBAR, height=72)
    logo.pack(fill="x")
    tk.Label(logo, text="◈", bg=BG_SIDEBAR, fg=ACCENT1,
             font=("Courier New", 22, "bold")).pack(side="left", padx=18, pady=16)
    tk.Label(logo, text="DECISION\nINTELLIGENCE", bg=BG_SIDEBAR, fg=TEXT_PRI,
             font=("Courier New", 9, "bold"), justify="left").pack(side="left")

    tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)

    # Mini KPI strip in sidebar
    kpi_frame = tk.Frame(sidebar, bg=BG_SIDEBAR)
    kpi_frame.pack(fill="x", padx=14, pady=10)
    for label, val, color in [
        ("Campaigns", f"{len(df):,}", TEXT_PRI),
        ("Avg ROI",   f"{df['ROI'].mean():.2f}", ACCENT2),
        ("Total Rev", f"${df['Revenue'].sum()/1e6:.1f}M", ACCENT4),
    ]:
        row = tk.Frame(kpi_frame, bg=BG_SIDEBAR)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=BG_SIDEBAR, fg=TEXT_SEC,
                 font=("Courier New", 9), width=12, anchor="w").pack(side="left")
        tk.Label(row, text=val, bg=BG_SIDEBAR, fg=color,
                 font=("Courier New", 11, "bold")).pack(side="right")

    tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=4)
    tk.Label(sidebar, text="ANALYSIS", bg=BG_SIDEBAR, fg=TEXT_SEC,
             font=("Courier New", 9, "bold")).pack(anchor="w", padx=22, pady=(8, 2))

    # ── Main content area ─────────────────────
    main = tk.Frame(root, bg=BG_DARK)
    main.grid(row=0, column=1, sticky="nsew")
    main.grid_rowconfigure(1, weight=1)
    main.grid_columnconfigure(0, weight=1)

    # Top header bar
    top_bar = tk.Frame(main, bg=BG_CARD, height=64)
    top_bar.grid(row=0, column=0, sticky="ew")
    top_bar.pack_propagate(False)

    title_var = tk.StringVar(value="Select an analysis from the sidebar")
    tk.Label(top_bar, textvariable=title_var, bg=BG_CARD, fg=TEXT_PRI,
             font=("Courier New", 15, "bold")).pack(side="left", padx=24, pady=18)

    status_var = tk.StringVar(value="Ready")
    tk.Label(top_bar, textvariable=status_var, bg=BG_CARD, fg=TEXT_SEC,
             font=("Courier New", 10)).pack(side="right", padx=24)

    # Content frame
    content = tk.Frame(main, bg=BG_DARK)
    content.grid(row=1, column=0, sticky="nsew")

    def run_view(label: str, fn):
        title_var.set(label)
        status_var.set("Computing…")
        root.update_idletasks()
        result_df = fn()
        show_table(content, result_df)
        status_var.set(f"{len(result_df)} rows")

    # Sidebar buttons
    _sidebar_btn(sidebar, "Budget Allocation",  "◉",
                 lambda: run_view("💰 Budget Allocation",
                                  lambda: budget_allocation(df)), ACCENT2)
    _sidebar_btn(sidebar, "What-If Analysis",   "▲",
                 lambda: run_view("📈 What-If Analysis (+20% Spend)",
                                  lambda: what_if_analysis(df)), ACCENT4)
    _sidebar_btn(sidebar, "Recommendations",    "◆",
                 lambda: run_view("🧠 Strategic Recommendations",
                                  lambda: generate_recommendations(df)), ACCENT1)
    _sidebar_btn(sidebar, "Simple Optimisation","⚡",
                 lambda: run_view("⚡ ROI-Weighted Optimisation",
                                  lambda: simple_optimization(df)), ACCENT3)
    _sidebar_btn(sidebar, "LP Optimisation",    "🤖",
                 lambda: run_view("🤖 Linear Programme (PuLP)",
                                  lambda: optimize_with_pulp(df)),
                 "#A78BFA" if PULP_AVAILABLE else TEXT_SEC)

    tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=14, pady=10)
    pulp_status = "● PuLP available" if PULP_AVAILABLE else "○ PuLP not installed"
    pulp_color  = ACCENT2 if PULP_AVAILABLE else ACCENT3
    tk.Label(sidebar, text=pulp_status, bg=BG_SIDEBAR, fg=pulp_color,
             font=("Courier New", 9)).pack(anchor="w", padx=22)

    root.mainloop()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_dashboard("processed_campaign_data.csv")