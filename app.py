"""
Nykaa Marketing Intelligence Platform — Full Interactive Dashboard
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cleaning import DataCleaner
from derived_kpi import KPIGenerator
from eda import EDAnalyzer
from hypothesis import HypothesisTester
from performance_analysis import PerformanceAnalyzer
from predictive_modeling import PredictiveModeler, XGB_AVAILABLE, CAT_AVAILABLE
from recommendations import RecommendationEngine

st.set_page_config(
    page_title="Nykaa Marketing Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d0f1a; }
.block-container { padding-top: 1.5rem; }

.page-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #f093fb 100%);
    padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
}
.page-header h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 0.95rem; }

.kpi-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #242840 100%);
    border: 1px solid #2e3350; border-radius: 12px; padding: 1rem 1.2rem;
    border-left: 4px solid #667eea;
}
.kpi-val  { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
.kpi-lbl  { color: #8892b0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; margin: 0; }
.kpi-delta{ font-size: 0.78rem; margin: 0; }

.section-card {
    background: #13152a; border: 1px solid #252840;
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
}

.insight-pill {
    display:inline-block; background: rgba(102,126,234,.15);
    border: 1px solid rgba(102,126,234,.4); color: #a9b8ff;
    border-radius: 20px; padding: .2rem .8rem; font-size:.8rem; margin:.2rem;
}

.sig-badge   { background:#00d4aa22; border:1px solid #00d4aa; color:#00d4aa;
               border-radius:8px; padding:.15rem .6rem; font-size:.78rem; }
.insig-badge { background:#ff6b9d22; border:1px solid #ff6b9d; color:#ff6b9d;
               border-radius:8px; padding:.15rem .6rem; font-size:.78rem; }

.stTabs [data-baseweb="tab-list"] {
    background:#13152a; border-radius:12px; padding:.4rem; gap:.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; padding:.4rem 1rem; color:#8892b0; font-weight:500;
}
.stTabs [aria-selected="true"] { background:#667eea22; color:#a9b8ff; }

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label { color:#a9b8ff !important; font-weight:500; }

.stButton>button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white; border:none; border-radius:8px;
    font-weight:600; transition:all .2s;
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 4px 15px rgba(102,126,234,.4); }

.stMetric { background:#13152a; border-radius:10px; padding:.8rem; border:1px solid #252840; }

div.stAlert { background:#13152a; border-left:4px solid #667eea; }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8d0e7', family='Inter'),
    colorway=['#667eea','#00d4aa','#ff6b9d','#ffb347','#64dfdf','#a78bfa'],
    xaxis=dict(gridcolor='#252840', zeroline=False),
    yaxis=dict(gridcolor='#252840', zeroline=False),
    legend=dict(bgcolor='#13152a', bordercolor='#252840'),
    margin=dict(t=50, b=40, l=40, r=20)
)


def apply_dark(fig, height=420):
    fig.update_layout(**DARK_LAYOUT, height=height)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, color="#667eea"):
    delta_html = f'<p class="kpi-delta" style="color:{color}">{delta}</p>' if delta else ''
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <p class="kpi-lbl">{label}</p>
        <p class="kpi-val">{value}</p>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎯 Nykaa Marketing Intelligence")
        st.markdown("---")
        uploaded = st.file_uploader("📁 Upload Campaign CSV", type=['csv'])

        df = None
        df_processed = None

        if uploaded:
            with st.spinner("Cleaning data…"):
                raw = pd.read_csv(uploaded)
                cleaner = DataCleaner()
                df = cleaner.run_full_pipeline(raw)
            st.success(f"✅ {len(df):,} rows loaded")

            st.markdown("---")
            st.markdown("### ⚙️ Pipeline")
            if st.button("🚀 Run Full KPI Analysis", width="stretch"):
                with st.spinner("Generating KPIs…"):
                    gen = KPIGenerator(df)
                    df_processed = gen.run_full_pipeline()
                st.session_state['df_processed'] = df_processed
                st.success("✅ KPIs ready!")
                st.rerun()

            st.markdown("---")
            st.markdown("### 📋 Data Quality")
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Cols", len(df.columns))
            miss = int(df.isnull().sum().sum())
            c1.metric("Missing", miss if miss else "None")
            c2.metric("Dupes", int(df.duplicated().sum()))

            with st.expander("📌 Columns"):
                st.write(list(df.columns))

        return df, st.session_state.get('df_processed')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview(df, dfp):
    st.markdown("### 📋 Data Overview")
    view = st.radio("Select view", ["Summary Statistics", "Data Types & Missing Values",
                                    "Raw Data Preview", "Column Distribution"],
                    horizontal=True)

    if view == "Summary Statistics":
        num_cols = dfp.select_dtypes(include=np.number).columns if dfp is not None else df.select_dtypes(include=np.number).columns
        src = dfp if dfp is not None else df
        st.dataframe(src[num_cols].describe().T.style.format("{:.3f}"), width="stretch")

    elif view == "Data Types & Missing Values":
        src = dfp if dfp is not None else df
        info = pd.DataFrame({
            'Type': src.dtypes.astype(str),
            'Non-Null Count': src.notnull().sum(),
            'Null Count': src.isnull().sum(),
            'Null %': (src.isnull().sum() / len(src) * 100).round(2),
            'Unique Values': src.nunique()
        })
        st.dataframe(info, width="stretch")

    elif view == "Raw Data Preview":
        src = dfp if dfp is not None else df
        n = st.slider("Rows to show", 10, min(500, len(src)), 50)
        st.dataframe(src.head(n), width="stretch")
        csv = src.to_csv(index=False)
        st.download_button("📥 Download Data", csv, "nykaa_processed.csv", "text/csv")

    elif view == "Column Distribution":
        src = dfp if dfp is not None else df
        num_cols = list(src.select_dtypes(include=np.number).columns)
        cat_cols = list(src.select_dtypes(include='object').columns)
        col_type = st.selectbox("Column type", ["Numeric", "Categorical"])
        if col_type == "Numeric" and num_cols:
            col = st.selectbox("Choose column", num_cols)
            fig = px.histogram(src, x=col, nbins=40,
                               title=f"Distribution of {col}",
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(apply_dark(fig), width="stretch")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean",    f"{src[col].mean():.3f}")
            c2.metric("Median",  f"{src[col].median():.3f}")
            c3.metric("Std Dev", f"{src[col].std():.3f}")
            c4.metric("Skewness",f"{src[col].skew():.3f}")
        elif col_type == "Categorical" and cat_cols:
            col = st.selectbox("Choose column", cat_cols)
            vc = src[col].value_counts().head(20)
            fig = px.bar(vc, title=f"Value Counts — {col}",
                         labels={'value':'Count','index':col},
                         color=vc.values,
                         color_continuous_scale='Purples')
            st.plotly_chart(apply_dark(fig), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — KPI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def tab_kpi(dfp):
    st.markdown("### 📊 Key Performance Indicators")

    # Top metrics row
    cols = st.columns(4)
    kpis = [
        ("Total Campaigns",  f"{len(dfp):,}",                         None, "#667eea"),
        ("Total Revenue",    f"${dfp.get('revenue', pd.Series([0])).sum():,.0f}", None, "#00d4aa"),
        ("Total Ad Spend",   f"${dfp.get('acquisition_cost', pd.Series([0])).sum():,.0f}", None, "#ffb347"),
        ("Avg ROI",          f"{dfp.get('roi', pd.Series([0])).mean():.2f}x",    None, "#ff6b9d"),
    ]
    for i,(lbl,val,dlt,clr) in enumerate(kpis):
        with cols[i]: kpi_card(lbl, val, dlt, clr)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    kpis2 = [
        ("Avg ROAS",         f"{dfp.get('roas', pd.Series([0])).mean():.2f}x",  None, "#a78bfa"),
        ("Avg CTR",          f"{dfp.get('ctr',  pd.Series([0])).mean()*100:.2f}%", None, "#64dfdf"),
        ("Avg Conv. Rate",   f"{dfp.get('conversion_rate', pd.Series([0])).mean()*100:.2f}%", None, "#00d4aa"),
        ("Avg CPC",          f"${dfp.get('cpc', pd.Series([0])).mean():.2f}",   None, "#ffb347"),
    ]
    for i,(lbl,val,dlt,clr) in enumerate(kpis2):
        with cols2[i]: kpi_card(lbl, val, dlt, clr)

    st.markdown("---")
    view = st.selectbox("📈 Choose KPI Chart", [
        "ROI Distribution", "ROI by Campaign Type", "ROI by Target Audience",
        "ROI by Language", "ROAS Distribution", "CTR vs Conversion Rate",
        "Revenue vs Spend Scatter", "KPI Correlation Heatmap"
    ])

    if view == "ROI Distribution" and 'roi' in dfp.columns:
        fig = px.histogram(dfp, x='roi', nbins=40,
                           title="Campaign ROI Distribution",
                           color_discrete_sequence=['#667eea'],
                           marginal='box')
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "ROI by Campaign Type" and all(c in dfp.columns for c in ['campaign_type','roi']):
        agg = dfp.groupby('campaign_type')['roi'].agg(['mean','std','count']).reset_index()
        agg.columns = ['Campaign Type','Avg ROI','Std Dev','Count']
        fig = px.bar(agg, x='Campaign Type', y='Avg ROI',
                     error_y='Std Dev', color='Avg ROI',
                     color_continuous_scale='Viridis',
                     text='Count', title="Avg ROI by Campaign Type")
        fig.update_traces(texttemplate='n=%{text}', textposition='outside')
        st.plotly_chart(apply_dark(fig), width="stretch")
        st.dataframe(agg, width="stretch")

    elif view == "ROI by Target Audience" and all(c in dfp.columns for c in ['target_audience','roi']):
        agg = dfp.groupby('target_audience')['roi'].mean().sort_values(ascending=True)
        fig = px.bar(agg, orientation='h', title="Avg ROI by Target Audience",
                     color=agg.values, color_continuous_scale='Teal')
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "ROI by Language" and all(c in dfp.columns for c in ['language','roi']):
        agg = dfp.groupby('language')['roi'].mean().sort_values(ascending=False)
        fig = px.bar(agg, title="Avg ROI by Language",
                     color=agg.values, color_continuous_scale='Purples')
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "ROAS Distribution" and 'roas' in dfp.columns:
        fig = px.histogram(dfp, x='roas', nbins=40, marginal='violin',
                           title="ROAS Distribution",
                           color_discrete_sequence=['#00d4aa'])
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "CTR vs Conversion Rate" and all(c in dfp.columns for c in ['ctr','conversion_rate']):
        color_col = 'campaign_type' if 'campaign_type' in dfp.columns else None
        fig = px.scatter(dfp, x='ctr', y='conversion_rate',
                         color=color_col, opacity=0.7,
                         title="CTR vs Conversion Rate",
                         labels={'ctr':'CTR','conversion_rate':'Conversion Rate'})
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "Revenue vs Spend Scatter" and all(c in dfp.columns for c in ['revenue','acquisition_cost']):
        color_col = 'roi_category' if 'roi_category' in dfp.columns else None
        size_col  = 'conversions' if 'conversions' in dfp.columns else None
        fig = px.scatter(dfp, x='acquisition_cost', y='revenue',
                         color=color_col, size=size_col,
                         title="Revenue vs Ad Spend",
                         labels={'acquisition_cost':'Ad Spend ($)','revenue':'Revenue ($)'})
        st.plotly_chart(apply_dark(fig), width="stretch")

    elif view == "KPI Correlation Heatmap":
        num_cols = dfp.select_dtypes(include=np.number).columns
        corr = dfp[num_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title="KPI Correlation Matrix")
        st.plotly_chart(apply_dark(fig, height=600), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EDA
# ─────────────────────────────────────────────────────────────────────────────
def tab_eda(dfp):
    st.markdown("### 🔍 Exploratory Data Analysis")
    analyzer = EDAnalyzer(dfp)

    analysis = st.selectbox("Choose Analysis", [
        "Univariate Distribution",
        "Segment Analysis (ROI by Group)",
        "Channel Effectiveness",
        "Top / Bottom Campaigns",
        "Correlation Heatmap",
        "Outlier Detection"
    ])

    num_cols = [c for c in ['duration','impressions','clicks','leads','conversions',
                             'revenue','acquisition_cost','roi','engagement_score','ctr',
                             'cpc','roas','conversion_rate'] if c in dfp.columns]

    if analysis == "Univariate Distribution":
        col = st.selectbox("Select metric", num_cols)
        c1, c2 = st.columns([2,1])
        with c1:
            fig = px.histogram(dfp, x=col, nbins=40, marginal='box',
                               title=f"Distribution of {col.upper()}",
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(apply_dark(fig), width="stretch")
        with c2:
            st.markdown("**Statistics**")
            stats = dfp[col].describe()
            for k,v in stats.items():
                st.metric(k, f"{v:.3f}")

    elif analysis == "Segment Analysis (ROI by Group)":
        seg_options = [c for c in ['campaign_type','target_audience','language','customer_segment'] if c in dfp.columns]
        if not seg_options:
            st.warning("No categorical segment columns found.")
            return
        seg   = st.selectbox("Group by", seg_options)
        metric= st.selectbox("Metric", [c for c in ['roi','revenue','roas','ctr','conversion_rate'] if c in dfp.columns])
        chart = st.radio("Chart type", ["Bar","Box","Violin"], horizontal=True)

        if chart == "Bar":
            agg = dfp.groupby(seg)[metric].mean().sort_values(ascending=False)
            fig = px.bar(agg, title=f"Avg {metric.upper()} by {seg}",
                         color=agg.values, color_continuous_scale='Viridis')
        elif chart == "Box":
            fig = px.box(dfp, x=seg, y=metric, color=seg,
                         title=f"{metric.upper()} Distribution by {seg}")
        else:
            fig = px.violin(dfp, x=seg, y=metric, color=seg, box=True,
                            title=f"{metric.upper()} Violin by {seg}")
        st.plotly_chart(apply_dark(fig), width="stretch")

        st.markdown("**Group Summary**")
        st.dataframe(dfp.groupby(seg)[metric].agg(['mean','median','std','count'])
                     .round(3).sort_values('mean', ascending=False), width="stretch")

    elif analysis == "Channel Effectiveness":
        if 'channel_used' not in dfp.columns:
            st.info("No `channel_used` column found.")
            return
        exp = dfp.copy()
        exp['channel_list'] = exp['channel_used'].str.split(',')
        exp = exp.explode('channel_list')
        exp['channel_list'] = exp['channel_list'].str.strip()
        exp = exp[exp['channel_list'].notna() & (exp['channel_list'] != '')]

        metric = st.selectbox("Metric", [c for c in ['roi','revenue','roas','conversions'] if c in exp.columns])
        agg = exp.groupby('channel_list')[metric].agg(['mean','sum','count']).reset_index()
        agg.columns = ['Channel', f'Avg {metric}', f'Total {metric}', 'Campaigns']
        agg = agg.sort_values(f'Avg {metric}', ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(agg, x='Channel', y=f'Avg {metric}',
                         title=f"Avg {metric.upper()} by Channel",
                         color=f'Avg {metric}', color_continuous_scale='Teal')
            st.plotly_chart(apply_dark(fig), width="stretch")
        with c2:
            fig2 = px.pie(agg, values='Campaigns', names='Channel',
                          title="Campaign Count by Channel",
                          hole=0.45)
            st.plotly_chart(apply_dark(fig2), width="stretch")
        st.dataframe(agg, width="stretch")

    elif analysis == "Top / Bottom Campaigns":
        metric = st.selectbox("Rank by", [c for c in ['roi','revenue','roas','conversions','ctr'] if c in dfp.columns])
        n = st.slider("Show top/bottom N", 5, 30, 10)
        top = dfp.nlargest(n, metric)
        bot = dfp.nsmallest(n, metric)

        c1, c2 = st.columns(2)
        id_col = 'campaign_id' if 'campaign_id' in dfp.columns else dfp.columns[0]
        with c1:
            st.markdown(f"#### 🏆 Top {n} by {metric.upper()}")
            fig = px.bar(top, x=id_col, y=metric, color=metric,
                         color_continuous_scale='Greens',
                         title=f"Top {n} Campaigns")
            st.plotly_chart(apply_dark(fig), width="stretch")
            st.dataframe(top[[id_col, metric]].reset_index(drop=True), width="stretch")
        with c2:
            st.markdown(f"#### 📉 Bottom {n} by {metric.upper()}")
            fig2 = px.bar(bot, x=id_col, y=metric, color=metric,
                          color_continuous_scale='Reds',
                          title=f"Bottom {n} Campaigns")
            st.plotly_chart(apply_dark(fig2), width="stretch")
            st.dataframe(bot[[id_col, metric]].reset_index(drop=True), width="stretch")

    elif analysis == "Correlation Heatmap":
        all_num = list(dfp.select_dtypes(include=np.number).columns)
        selected = st.multiselect("Select metrics to include", all_num,
                                  default=[c for c in ['roi','revenue','roas','ctr','conversion_rate',
                                                        'acquisition_cost','duration','impressions'] if c in all_num])
        if len(selected) >= 2:
            corr = dfp[selected].corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1, title="Correlation Heatmap")
            st.plotly_chart(apply_dark(fig, height=600), width="stretch")
        else:
            st.warning("Select at least 2 metrics.")

    elif analysis == "Outlier Detection":
        col = st.selectbox("Metric for outlier analysis", num_cols)
        q1, q3 = dfp[col].quantile(0.25), dfp[col].quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = dfp[(dfp[col] < lb) | (dfp[col] > ub)]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(dfp, y=col, title=f"Boxplot — {col.upper()}",
                         color_discrete_sequence=['#667eea'], points='outliers')
            st.plotly_chart(apply_dark(fig), width="stretch")
        with c2:
            st.metric("Total Outliers", len(outliers))
            st.metric("Outlier %", f"{len(outliers)/len(dfp)*100:.1f}%")
            st.metric("Lower Bound", f"{lb:.3f}")
            st.metric("Upper Bound", f"{ub:.3f}")
        if not outliers.empty:
            st.markdown("**Outlier Rows**")
            id_col = 'campaign_id' if 'campaign_id' in dfp.columns else dfp.columns[0]
            show_cols = [c for c in [id_col, col, 'campaign_type', 'revenue'] if c in dfp.columns]
            st.dataframe(outliers[show_cols].reset_index(drop=True), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PERFORMANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def tab_performance(dfp):
    st.markdown("### 🎯 Performance Analysis")
    pa = PerformanceAnalyzer(dfp)

    analysis = st.selectbox("Choose Analysis", [
        "Campaign Ranking & Segmentation",
        "Channel Mix Analysis",
        "Duration vs Performance",
        "Break-Even Analysis",
        "Revenue Efficiency"
    ])

    if analysis == "Campaign Ranking & Segmentation":
        metric = st.selectbox("Rank by", [c for c in ['roi','revenue','roas','conversions','ctr','cpc'] if c in dfp.columns])
        n = st.slider("Top N campaigns", 5, 50, 10)
        top = pa.get_top_campaigns(metric, n)
        if not top.empty:
            id_col = 'campaign_id' if 'campaign_id' in dfp.columns else dfp.columns[0]
            fig = px.bar(top, x=id_col, y=metric,
                         color=metric, color_continuous_scale='Viridis',
                         title=f"Top {n} Campaigns by {metric.upper()}")
            st.plotly_chart(apply_dark(fig), width="stretch")

            if 'campaign_type' in dfp.columns:
                seg = dfp.groupby('campaign_type')[metric].agg(['mean','sum','count']).reset_index()
                seg.columns = ['Campaign Type', f'Avg {metric}', f'Total {metric}', 'Count']
                fig2 = px.treemap(seg, path=['Campaign Type'],
                                  values=f'Total {metric}', color=f'Avg {metric}',
                                  color_continuous_scale='RdYlGn',
                                  title=f"Segment Treemap — {metric.upper()}")
                st.plotly_chart(apply_dark(fig2, 380), width="stretch")
            st.dataframe(top.reset_index(drop=True), width="stretch")

    elif analysis == "Channel Mix Analysis":
        ch = pa.analyze_channel_mix()
        if ch.empty:
            st.info("No channel data found.")
            return
        metric_col = ch.columns[1]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(ch.head(15), x='Channel', y=metric_col,
                         color=metric_col, color_continuous_scale='Teal',
                         title="Channel ROI Effectiveness")
            st.plotly_chart(apply_dark(fig), width="stretch")
        with c2:
            if 'Campaigns' in ch.columns:
                fig2 = px.pie(ch, values='Campaigns', names='Channel',
                              title="Campaign Distribution by Channel", hole=0.4)
                st.plotly_chart(apply_dark(fig2), width="stretch")
        st.dataframe(ch, width="stretch")

    elif analysis == "Duration vs Performance":
        fig, dur_perf = pa.analyze_duration_effect()
        if fig:
            st.plotly_chart(fig, width="stretch")
            if dur_perf is not None:
                st.dataframe(dur_perf, width="stretch")
        else:
            # Fallback scatter
            if all(c in dfp.columns for c in ['duration','roi']):
                color_col = 'campaign_type' if 'campaign_type' in dfp.columns else None
                fig2 = px.scatter(dfp, x='duration', y='roi',
                                  color=color_col, opacity=0.65,
                                  size='revenue' if 'revenue' in dfp.columns else None,
                                  title="Duration vs ROI",
                                  trendline='ols')
                st.plotly_chart(apply_dark(fig2), width="stretch")

    elif analysis == "Break-Even Analysis":
        if not all(c in dfp.columns for c in ['revenue','acquisition_cost']):
            st.warning("Need revenue and acquisition_cost columns.")
            return
        df2 = dfp.copy()
        df2['profit'] = df2['revenue'] - df2['acquisition_cost']
        df2['profit_margin'] = df2['profit'] / df2['revenue'].replace(0, np.nan)

        c1,c2,c3 = st.columns(3)
        profitable = (df2['profit'] >= 0).sum()
        c1.metric("Profitable Campaigns", f"{profitable:,}")
        c2.metric("Break-Even Rate", f"{profitable/len(df2)*100:.1f}%")
        c3.metric("Total Profit", f"${df2['profit'].sum():,.0f}")

        c_a, c_b = st.columns(2)
        with c_a:
            fig = px.histogram(df2, x='profit', nbins=40,
                               title="Profit Distribution",
                               color_discrete_sequence=['#00d4aa'])
            fig.add_vline(x=0, line_dash='dash', line_color='#ff6b9d',
                          annotation_text="Break-even")
            st.plotly_chart(apply_dark(fig), width="stretch")
        with c_b:
            fig2 = px.scatter(df2, x='acquisition_cost', y='profit',
                              color='profit', color_continuous_scale='RdYlGn',
                              title="Spend vs Profit", opacity=0.7)
            fig2.add_hline(y=0, line_dash='dash', line_color='#ff6b9d')
            st.plotly_chart(apply_dark(fig2), width="stretch")

    elif analysis == "Revenue Efficiency":
        fig, summary = pa.analyze_revenue_efficiency()
        if fig:
            st.plotly_chart(fig, width="stretch")
        if summary:
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Profit",     f"${summary['total_profit']:,.0f}")
            c2.metric("Break-Even Rate",  f"{summary['break_even_rate']:.1f}%")
            c3.metric("Avg Profit Margin",f"{summary['avg_profit_margin']*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — A/B TESTING
# ─────────────────────────────────────────────────────────────────────────────
def tab_hypothesis(dfp):
    st.markdown("### 📊 A/B Hypothesis Testing")
    ht = HypothesisTester(dfp)

    test_type = st.selectbox("Choose Test", [
        "Campaign Type Comparison",
        "Language Comparison",
        "Short vs Long Duration",
        "Custom A/B Groups"
    ])

    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)

    if test_type == "Campaign Type Comparison":
        results = ht.test_campaign_types()
        if results.empty:
            st.info("Need campaign_type and roi columns.")
            return
        _show_test_results(results, "Campaign Type", alpha)
        fig = ht.plot_test_results(results, "Campaign Type Tests")
        if fig: st.pyplot(fig, width="stretch")

    elif test_type == "Language Comparison":
        results = ht.test_languages()
        if results.empty:
            st.info("Need language and roi columns.")
            return
        _show_test_results(results, "Language", alpha)
        fig = ht.plot_test_results(results, "Language Tests")
        if fig: st.pyplot(fig, width="stretch")

    elif test_type == "Short vs Long Duration":
        r = ht.test_duration_groups()
        if 'error' in r:
            st.warning(r['error'])
            return
        sig = r['p_value'] < alpha
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("p-value", f"{r['p_value']:.4f}")
        c2.metric("Significant", "✅ Yes" if sig else "❌ No")
        c3.metric("Short Duration ROI", f"{r['group1_mean']:.3f}")
        c4.metric("Long Duration ROI",  f"{r['group2_mean']:.3f}")

        if 'duration' in dfp.columns and 'roi' in dfp.columns:
            med = dfp['duration'].median()
            df2 = dfp.copy()
            df2['Duration Group'] = np.where(df2['duration'] <= med, 'Short', 'Long')
            fig = px.box(df2, x='Duration Group', y='roi', color='Duration Group',
                         title="ROI: Short vs Long Duration Campaigns")
            st.plotly_chart(apply_dark(fig), width="stretch")

    elif test_type == "Custom A/B Groups":
        cat_cols = [c for c in dfp.select_dtypes(include='object').columns]
        if not cat_cols:
            st.info("No categorical columns available.")
            return
        grp_col = st.selectbox("Group column", cat_cols)
        metric   = st.selectbox("Metric to test", [c for c in ['roi','revenue','roas','conversion_rate'] if c in dfp.columns])
        groups   = dfp[grp_col].dropna().unique().tolist()
        g1 = st.selectbox("Group A", groups, index=0)
        g2 = st.selectbox("Group B", groups, index=min(1, len(groups)-1))

        if g1 != g2:
            from scipy import stats as scipy_stats
            s1 = dfp[dfp[grp_col] == g1][metric].dropna()
            s2 = dfp[dfp[grp_col] == g2][metric].dropna()
            _, p = scipy_stats.ttest_ind(s1, s2, equal_var=False)
            sig = p < alpha
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("p-value", f"{p:.4f}")
            c2.metric("Significant", "✅ Yes" if sig else "❌ No")
            c3.metric(f"{g1} Avg {metric}", f"{s1.mean():.3f}")
            c4.metric(f"{g2} Avg {metric}", f"{s2.mean():.3f}")

            fig = go.Figure()
            for grp, data, color in [(g1, s1,'#667eea'),(g2, s2,'#00d4aa')]:
                fig.add_trace(go.Violin(y=data, name=grp, box_visible=True,
                                        line_color=color, fillcolor=color,
                                        opacity=0.6, meanline_visible=True))
            fig.update_layout(**DARK_LAYOUT, title=f"{g1} vs {g2} — {metric.upper()}")
            st.plotly_chart(fig, width="stretch")


def _show_test_results(results, label, alpha):
    sig_count = results['significant'].sum() if 'significant' in results.columns else 0
    c1,c2 = st.columns(2)
    c1.metric("Pairs Tested", len(results))
    c2.metric("Significant Pairs", sig_count)
    st.dataframe(
        results.style.map(
            lambda v: 'color:#00d4aa' if v is True else ('color:#ff6b9d' if v is False else ''),
            subset=['significant'] if 'significant' in results.columns else []
        ), width="stretch"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — PREDICTIVE MODELING
# ─────────────────────────────────────────────────────────────────────────────
def tab_modeling(dfp):
    st.markdown("### 🤖 Predictive Modeling")

    available_targets = [c for c in ['revenue','conversions','leads','roi'] if c in dfp.columns]
    if not available_targets:
        st.warning("No suitable target columns found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("🎯 Target variable to predict", available_targets)
    with c2:
        model_options = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']
        if XGB_AVAILABLE: model_options.append('XGBoost')
        if CAT_AVAILABLE: model_options.append('CatBoost')
        selected = st.multiselect("🤖 Models to train", model_options,
                                  default=['Random Forest'])

    tune = st.checkbox("⚙️ Hyperparameter Tuning (slower)", value=False)

    if not selected:
        st.info("Please select at least one model.")
        return

    if st.button("🚀 Train Models", width="stretch"):
        with st.spinner(f"Training {', '.join(selected)} to predict {target}…"):
            modeler = PredictiveModeler(dfp)
            out = modeler.run_full_pipeline(target=target,
                                           selected_models=selected,
                                           tune_models=tune)

        if 'error' in out:
            st.error(out['error'])
            return

        st.success("✅ Training complete!")

        # Model comparison table
        st.markdown("#### 📊 Model Comparison")
        rows = [{'Model': name, 'R² Score': m['r2'], 'RMSE': m['rmse'], 'MAE': m['mae']}
                for name, m in out['results'].items()]
        comp_df = pd.DataFrame(rows).sort_values('R² Score', ascending=False)
        st.dataframe(comp_df.style.highlight_max(subset=['R² Score'], color='#00d4aa33')
                     .highlight_min(subset=['RMSE','MAE'], color='#00d4aa33'),
                     width="stretch")

        best = out['best_model']
        best_m = out['results'][best]
        st.markdown(f"#### 🏆 Best Model: **{best}** (R² = {best_m['r2']:.4f})")

        # Actual vs Predicted + Residuals
        actual = best_m['actual']
        preds  = best_m['predictions']
        residuals = actual - preds

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Actual vs Predicted', 'Residual Distribution'])
        fig.add_trace(go.Scatter(x=actual, y=preds, mode='markers',
                                 marker=dict(color='#667eea', size=6, opacity=0.65),
                                 name='Predictions'), row=1, col=1)
        mn, mx = min(actual.min(), preds.min()), max(actual.max(), preds.max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                 mode='lines', line=dict(dash='dash', color='#ffb347'),
                                 name='Perfect fit'), row=1, col=1)
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                   marker_color='#00d4aa', name='Residuals'), row=1, col=2)
        fig.update_layout(**DARK_LAYOUT, height=420, showlegend=True)
        st.plotly_chart(fig, width="stretch")

        # Feature Importance
        if out['feature_importance'] is not None:
            st.markdown("#### 🔑 Feature Importance")
            fi = out['feature_importance']
            n_feat = st.slider("Number of features to show", 5, min(30, len(fi)), 15)
            fig_fi = px.bar(fi.head(n_feat), x='importance', y='feature',
                            orientation='h', color='importance',
                            color_continuous_scale='Viridis',
                            title=f"Top {n_feat} Features — {best}")
            st.plotly_chart(apply_dark(fig_fi, 450), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def tab_recommendations(dfp):
    st.markdown("### 💡 Strategic Recommendations")
    engine = RecommendationEngine(dfp)

    rec_section = st.selectbox("Choose Analysis", [
        "ROI Distribution & Portfolio Health",
        "Budget Allocation",
        "What-If Spend Analysis",
        "Strategic Recommendations",
        "Export Report"
    ])

    if rec_section == "ROI Distribution & Portfolio Health":
        roi_a = engine.analyze_roi_distribution()
        if not roi_a:
            st.warning("Need roi column.")
            return
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg ROI",          f"{roi_a['mean_roi']:.3f}")
        c2.metric("Median ROI",       f"{roi_a['median_roi']:.3f}")
        c3.metric("High Performers",  roi_a['high_performers'])
        c4.metric("Low Performers",   roi_a['low_performers'])

        if roi_a.get('high_performer_revenue', 0):
            st.metric("Revenue from High-ROI Campaigns",
                      f"${roi_a['high_performer_revenue']:,.0f}")

        if 'roi' in dfp.columns:
            fig = px.histogram(dfp, x='roi', nbins=40, marginal='box',
                               title="ROI Distribution — Portfolio View",
                               color_discrete_sequence=['#667eea'])
            mean_roi = dfp['roi'].mean()
            fig.add_vline(x=mean_roi, line_dash='dash', line_color='#ffb347',
                          annotation_text=f"Avg: {mean_roi:.2f}")
            st.plotly_chart(apply_dark(fig), width="stretch")

        if 'roi_category' in dfp.columns:
            vc = dfp['roi_category'].value_counts()
            fig2 = px.pie(values=vc.values, names=vc.index,
                          title="ROI Category Breakdown", hole=0.45,
                          color_discrete_sequence=['#00d4aa','#667eea','#ffb347','#ff6b9d'])
            st.plotly_chart(apply_dark(fig2), width="stretch")

    elif rec_section == "Budget Allocation":
        total_budget = st.slider("Total Budget ($)", 10_000, 1_000_000, 100_000, 10_000,
                                 format="$%d")
        alloc = engine.budget_allocation(total_budget)
        if alloc.empty:
            st.info("Need roi column for allocation.")
            return

        st.dataframe(alloc, width="stretch")

        if 'Avg_ROI' in alloc.columns and 'Allocated_Budget' in alloc.columns:
            label_col = alloc.columns[0]
            fig = px.bar(alloc, x=label_col, y='Allocated_Budget',
                         color='Avg_ROI', color_continuous_scale='Greens',
                         title=f"ROI-Weighted Budget Allocation (Total ${total_budget:,})")
            st.plotly_chart(apply_dark(fig), width="stretch")

    elif rec_section == "What-If Spend Analysis":
        uplift = st.slider("Spend Increase %", 5, 100, 20, 5) / 100 + 1
        st.info(f"Showing projected revenue if spend is increased by **{(uplift-1)*100:.0f}%**")
        df_wi = engine.what_if_analysis(uplift)
        if df_wi.empty:
            st.warning("Need acquisition_cost, revenue and roas columns.")
            return

        total_current  = dfp.get('revenue', pd.Series([0])).sum()
        total_expected = df_wi['expected_revenue'].sum() if 'expected_revenue' in df_wi.columns else 0
        total_delta    = df_wi['revenue_delta'].sum() if 'revenue_delta' in df_wi.columns else 0

        c1,c2,c3 = st.columns(3)
        c1.metric("Current Revenue",  f"${total_current:,.0f}")
        c2.metric("Projected Revenue",f"${total_expected:,.0f}")
        c3.metric("Revenue Uplift",   f"${total_delta:,.0f}",
                  delta=f"+{total_delta/total_current*100:.1f}%" if total_current else None)

        st.dataframe(df_wi, width="stretch")

        if 'expected_revenue' in df_wi.columns and 'revenue' in df_wi.columns:
            id_col = df_wi.columns[0]
            fig = go.Figure()
            fig.add_bar(x=df_wi[id_col], y=df_wi['revenue'],
                        name='Current Revenue', marker_color='#667eea')
            fig.add_bar(x=df_wi[id_col], y=df_wi['expected_revenue'],
                        name='Projected Revenue', marker_color='#00d4aa')
            fig.update_layout(**DARK_LAYOUT, barmode='group',
                              title="Current vs Projected Revenue")
            st.plotly_chart(fig, width="stretch")

    elif rec_section == "Strategic Recommendations":
        budget_recs = engine.generate_budget_recommendations()
        if not budget_recs:
            st.info("Need roi column for recommendations.")
            return
        for rec in budget_recs:
            color = {'Scale Up':'#00d4aa','Cut Spend':'#ff6b9d','Portfolio':'#ffb347'}.get(rec['category'],'#667eea')
            st.markdown(f"""
            <div style="background:#13152a;border:1px solid #252840;border-left:4px solid {color};
                        border-radius:10px;padding:1rem;margin:.5rem 0">
                <span style="background:{color}22;border:1px solid {color};color:{color};
                             border-radius:12px;padding:.1rem .6rem;font-size:.78rem">{rec['category']}</span>
                <h4 style="color:#f0f2ff;margin:.5rem 0 .2rem">{rec['recommendation']}</h4>
                <p style="color:#8892b0;margin:0;font-size:.88rem">📌 {rec['action']}</p>
                <p style="color:#8892b0;margin:.2rem 0 0;font-size:.82rem">
                    Impact: <strong style="color:{color}">{rec['impact']}</strong></p>
            </div>""", unsafe_allow_html=True)

    elif rec_section == "Export Report":
        rows = {'Metric':[], 'Value':[]}
        if 'roi' in dfp.columns:
            rows['Metric'] += ['Total Campaigns','Avg ROI','Median ROI']
            rows['Value']  += [len(dfp), f"{dfp['roi'].mean():.3f}", f"{dfp['roi'].median():.3f}"]
        if 'revenue' in dfp.columns:
            rows['Metric'].append('Total Revenue')
            rows['Value'].append(f"${dfp['revenue'].sum():,.0f}")
        if 'acquisition_cost' in dfp.columns:
            rows['Metric'].append('Total Ad Spend')
            rows['Value'].append(f"${dfp['acquisition_cost'].sum():,.0f}")
        if 'roas' in dfp.columns:
            rows['Metric'].append('Avg ROAS')
            rows['Value'].append(f"{dfp['roas'].mean():.2f}x")

        rpt = pd.DataFrame(rows)
        st.dataframe(rpt, width="stretch")

        csv = dfp.to_csv(index=False)
        st.download_button("📥 Download Full Processed Dataset", csv,
                           "nykaa_processed.csv", "text/csv",
                           width="stretch")

        rpt_csv = rpt.to_csv(index=False)
        st.download_button("📥 Download Summary Report", rpt_csv,
                           "nykaa_report.csv", "text/csv",
                           width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def welcome_screen():
    st.markdown("""
    <div class="page-header" style="text-align:center">
        <h1>🎯 Nykaa Marketing Intelligence Platform</h1>
        <p>Upload your campaign CSV to unlock powerful analytics & AI-driven recommendations</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("📋 Data Overview",   "Shape, dtypes, missing values, column distributions"),
        ("📊 KPI Dashboard",   "CTR, CPC, ROAS, ROI — interactive metric explorer"),
        ("🔍 EDA",             "Distributions, segment analysis, channel effectiveness, outliers"),
        ("🎯 Performance",     "Campaign ranking, channel mix, duration, break-even"),
        ("📊 A/B Testing",     "Statistical significance across types, languages, duration"),
        ("🤖 Predictive ML",   "Random Forest, XGBoost, CatBoost — predict revenue/ROI"),
        ("💡 Recommendations", "Budget allocation, what-if analysis, strategic insights"),
        ("📥 Export",          "Download processed data & summary reports"),
    ]
    for i, (title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="section-card">
                <h4 style="color:#a9b8ff;margin:0 0 .4rem">{title}</h4>
                <p style="color:#8892b0;font-size:.85rem;margin:0">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📊 Expected CSV columns:** `campaign_id`, `campaign_type`, `target_audience`, `language`, `impressions`, `clicks`, `leads`, `conversions`, `revenue`, `acquisition_cost`, `roi`, `duration`, `engagement_score`, `channel_used` (comma-separated)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if 'df_processed' not in st.session_state:
        st.session_state['df_processed'] = None

    df, dfp = render_sidebar()

    if df is None:
        welcome_screen()
        return

    st.markdown("""
    <div class="page-header">
        <h1>🎯 Nykaa Marketing Intelligence Platform</h1>
        <p>Interactive analytics dashboard — choose any analysis from each tab</p>
    </div>""", unsafe_allow_html=True)

    if dfp is None:
        st.info("👈 Click **'Run Full KPI Analysis'** in the sidebar to enable all tabs.")

        # Still show overview with raw data
        tab_overview(df, None)
        return

    tabs = st.tabs(["📋 Overview", "📊 KPI Dashboard", "🔍 EDA",
                    "🎯 Performance", "📊 A/B Testing", "🤖 Predictive ML", "💡 Recommendations"])

    with tabs[0]: tab_overview(df, dfp)
    with tabs[1]: tab_kpi(dfp)
    with tabs[2]: tab_eda(dfp)
    with tabs[3]: tab_performance(dfp)
    with tabs[4]: tab_hypothesis(dfp)
    with tabs[5]: tab_modeling(dfp)
    with tabs[6]: tab_recommendations(dfp)


if __name__ == "__main__":
    main()