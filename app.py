# app.py
# Purpose: This is the main dashboard file. It creates the actual visual 
# interface you see in the web browser. It builds the KPI cards, the tabs, 
# the interactive Plotly charts, and handles uploading your Excel file.

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from etl import load_and_clean

st.set_page_config(page_title="Meta Ads Analytics", page_icon="📊", layout="wide")
DATA_DIR = Path("data")
DEFAULT_FILE = DATA_DIR / "meta ads 1.xlsx"

@st.cache_data(show_spinner="Loading and cleaning data…")
def _load(file_bytes: bytes, filename: str, target_roas: float, target_cac: float | None) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    tmp = DATA_DIR / ("_uploaded_" + filename)
    tmp.write_bytes(file_bytes)
    return load_and_clean(tmp, target_roas=target_roas, target_cac=target_cac)

def _get_data(target_roas: float, target_cac: float | None) -> pd.DataFrame:
    up = st.sidebar.file_uploader("Upload Meta Ads export (xlsx / csv)", type=["xlsx", "xls", "csv"])
    if up is not None:
        return _load(up.getvalue(), up.name, target_roas, target_cac)
    if DEFAULT_FILE.exists():
        return _load(DEFAULT_FILE.read_bytes(), DEFAULT_FILE.name, target_roas, target_cac)
    st.info("Upload a Meta Ads Manager export in the sidebar to begin.")
    st.stop()

st.sidebar.header("Settings")
target_roas = st.sidebar.number_input("Target ROAS (scale flag threshold)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
target_cac_in = st.sidebar.number_input("Target CAC in USD (optional, 0 to disable)", min_value=0.0, value=0.0, step=1.0)
target_cac = target_cac_in if target_cac_in > 0 else None

df = _get_data(target_roas, target_cac)

st.sidebar.markdown("### Filters")
periods = sorted({str(d) for d in df["reporting_start"].dropna().unique()}) if "reporting_start" in df else []
if periods:
    chosen = st.sidebar.multiselect("Reporting period start", periods, default=periods)
    if chosen: df = df[df["reporting_start"].astype(str).isin(chosen)]

languages = sorted(df["language"].unique())
regions = sorted(df["region"].unique())
funnels = sorted(df["funnel_stage"].unique())

sel_lang = st.sidebar.multiselect("Language", languages, default=languages)
sel_reg = st.sidebar.multiselect("Region", regions, default=regions)
sel_fun = st.sidebar.multiselect("Funnel stage", funnels, default=funnels)
min_spend = st.sidebar.slider("Minimum spend per ad (USD)", 0, 500, 0, step=10)

mask = df["language"].isin(sel_lang) & df["region"].isin(sel_reg) & df["funnel_stage"].isin(sel_fun) & (df["spend"] >= min_spend)
df = df[mask].copy()
if df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.title("📊 Meta Ads Analytics Dashboard")
total_spend = df["spend"].sum()
total_rev = df["revenue"].sum()
total_purch = df["purchases"].sum()
blended_roas = total_rev / total_spend if total_spend else np.nan
blended_cac = total_spend / total_purch if total_purch else np.nan
avg_ctr = (df["clicks"].sum() / df["impressions"].sum() * 100 if df["impressions"].sum() else np.nan)
avg_cpm = (total_spend / df["impressions"].sum() * 1000 if df["impressions"].sum() else np.nan)
avg_freq = (df["impressions"].sum() / df["reach"].sum() if df["reach"].sum() else np.nan)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Spend", f"${total_spend:,.0f}")
c2.metric("Revenue", f"${total_rev:,.0f}")
c3.metric("Blended ROAS", f"{blended_roas:.2f}" if pd.notna(blended_roas) else "—")
c4.metric("Purchases", f"{total_purch:,.0f}")
c5.metric("Blended CAC", f"${blended_cac:.2f}" if pd.notna(blended_cac) else "—")
c6.metric("Avg frequency", f"{avg_freq:.2f}" if pd.notna(avg_freq) else "—")
st.caption(f"{len(df):,} ads • CTR {avg_ctr:.2f}% • CPM ${avg_cpm:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🌍 Language × Region", "🏆 Ad leaderboard", "🚨 Diagnostics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        by_lang = df.groupby("language", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), purchases=("purchases", "sum"))
        by_lang["roas"] = by_lang["revenue"] / by_lang["spend"].replace(0, np.nan)
        by_lang = by_lang.sort_values("spend", ascending=False)
        fig = px.bar(by_lang.melt(id_vars="language", value_vars=["spend", "revenue"], var_name="Metric", value_name="USD"), x="language", y="USD", color="Metric", barmode="group", title="Spend vs Revenue by language")
        fig.update_layout(xaxis_tickangle=-45, legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_region = df.groupby("region", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), purchases=("purchases", "sum"))
        by_region["roas"] = by_region["revenue"] / by_region["spend"].replace(0, np.nan)
        by_region = by_region.sort_values("spend", ascending=False)
        fig = px.bar(by_region, x="region", y="roas", color="roas", color_continuous_scale="RdYlGn", title="ROAS by region")
        fig.add_hline(y=1.0, line_dash="dot", line_color="red", annotation_text="breakeven")
        fig.add_hline(y=target_roas, line_dash="dot", line_color="green", annotation_text="target")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Spend share by ad set")
    top_adsets = df.groupby("adset_name", as_index=False)["spend"].sum().sort_values("spend", ascending=False)
    if len(top_adsets) > 0:
        fig = px.treemap(top_adsets, path=["adset_name"], values="spend", title="Where the money goes")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    matrix = df.pivot_table(index="language", columns="region", values="revenue", aggfunc="sum", fill_value=0)
    spend_matrix = df.pivot_table(index="language", columns="region", values="spend", aggfunc="sum", fill_value=0)
    roas_matrix = (matrix / spend_matrix.replace(0, np.nan)).round(2)
    st.subheader("ROAS heatmap (language × region)")
    if roas_matrix.shape[0] and roas_matrix.shape[1]:
        fig = px.imshow(roas_matrix, aspect="auto", color_continuous_scale="RdYlGn", zmin=0, zmax=3, labels=dict(x="Region", y="Language", color="ROAS"))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Language performance table")
    tbl = df.groupby("language", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), purchases=("purchases", "sum"), impressions=("impressions", "sum"), clicks=("clicks", "sum"), lpvs=("landing_page_views", "sum"))
    tbl["roas"] = (tbl["revenue"] / tbl["spend"].replace(0, np.nan)).round(2)
    tbl["cac"] = (tbl["spend"] / tbl["purchases"].replace(0, np.nan)).round(2)
    tbl["ctr%"] = (tbl["clicks"] / tbl["impressions"].replace(0, np.nan) * 100).round(2)
    tbl["purchase_rate%"] = (tbl["purchases"] / tbl["lpvs"].replace(0, np.nan) * 100).round(2)
    st.dataframe(tbl.sort_values("spend", ascending=False)[["language", "spend", "revenue", "roas", "cac", "purchases", "ctr%", "purchase_rate%"]], use_container_width=True, hide_index=True)

with tab3:
    min_ads_spend = st.number_input("Min spend per ad for the leaderboard (USD)", min_value=0, value=100, step=10)
    ads = df.groupby(["ad_name", "language", "region", "funnel_stage"], as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), purchases=("purchases", "sum"), impressions=("impressions", "sum"), clicks=("clicks", "sum"), lpvs=("landing_page_views", "sum"), frequency=("frequency", "mean"))
    ads["roas"] = (ads["revenue"] / ads["spend"].replace(0, np.nan)).round(2)
    ads["cac"] = (ads["spend"] / ads["purchases"].replace(0, np.nan)).round(2)
    ads["ctr%"] = (ads["clicks"] / ads["impressions"].replace(0, np.nan) * 100).round(2)
    ads["purchase_rate%"] = (ads["purchases"] / ads["lpvs"].replace(0, np.nan) * 100).round(2)
    ads = ads[ads["spend"] >= min_ads_spend]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Top 15 by ROAS")
        st.dataframe(ads.sort_values("roas", ascending=False).head(15)[["ad_name", "language", "region", "spend", "revenue", "roas", "cac", "frequency"]], use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### 🔴 Bottom 15 by ROAS")
        st.dataframe(ads.sort_values("roas", ascending=True).head(15)[["ad_name", "language", "region", "spend", "revenue", "roas", "cac", "frequency"]], use_container_width=True, hide_index=True)
    plot_ads = ads.dropna(subset=["ctr%", "purchase_rate%"]).copy()
    if not plot_ads.empty:
        fig = px.scatter(plot_ads, x="ctr%", y="purchase_rate%", size="spend", color="roas", hover_name="ad_name", color_continuous_scale="RdYlGn", hover_data=["language", "region", "spend", "roas"])
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Auto-flags")
    flags_df = df[df["flags"] != ""].groupby(["ad_name", "language", "region", "flags"], as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), purchases=("purchases", "sum"), roas=("roas", "mean"), cac=("cac", "mean")).sort_values("spend", ascending=False)
    for flag, label in [("cut", "🛑 Cut candidates (ROAS < 1)"), ("scale", f"🚀 Scale candidates (ROAS ≥ {target_roas})"), ("fatigue", "😩 Fatigue risk"), ("high_cac", "💸 High CAC")]:
        sub = flags_df[flags_df["flags"].str.contains(flag)]
        if not sub.empty:
            st.markdown(f"#### {label}")
            st.dataframe(sub[["ad_name", "language", "region", "spend", "revenue", "roas", "cac"]], use_container_width=True, hide_index=True)
    st.subheader("Raw data")
    st.dataframe(df, use_container_width=True, hide_index=True)