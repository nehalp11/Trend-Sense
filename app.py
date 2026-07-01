# TrendSense — Streamlit Dashboard
# Interactive fashion trend explorer

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trend_scorer import compute_trend_score, rank_trends

# Page config
st.set_page_config(
    page_title="TrendSense — Indian Fashion Trend Detector",
    page_icon="🔥",
    layout="wide"
)

# Header
st.title("🔥 TrendSense")
st.markdown("**Emerging Indian Fashion Trend Detection** | Powered by RAR-BERT + CLIP")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Show top N trends", min_value=3, max_value=20, value=10)
min_score = st.sidebar.slider("Minimum trend score", min_value=0.0, max_value=1.0, value=0.2)
st.sidebar.markdown("---")
st.sidebar.markdown("**Source Weights**")
w_google = st.sidebar.slider("Google Trends weight", 0.0, 1.0, 0.40)
w_reddit = st.sidebar.slider("Reddit buzz weight", 0.0, 1.0, 0.35)
w_pinterest = st.sidebar.slider("Pinterest weight", 0.0, 1.0, 0.25)

# Sample data (replace with real pipeline output)
SAMPLE_TRENDS = [
    ("mirror work dupatta", 0.85, 0.72, 0.60),
    ("phulkari jacket", 0.92, 0.88, 0.75),
    ("boho kurta", 0.45, 0.68, 0.55),
    ("kalamkari fusion", 0.20, 0.25, 0.18),
    ("block print co-ord", 0.30, 0.40, 0.35),
    ("indo western blazer", 0.70, 0.65, 0.58),
    ("ethnic joggers", 0.55, 0.60, 0.45),
    ("handloom saree", 0.78, 0.70, 0.65),
    ("tie-dye dupatta", 0.40, 0.50, 0.42),
    ("bandhani dress", 0.65, 0.72, 0.60),
]

# Compute scores
weights = {"google_trends": w_google, "reddit_buzz": w_reddit, "pinterest": w_pinterest}
trends = [
    compute_trend_score(kw, g, r, p, weights)
    for kw, g, r, p in SAMPLE_TRENDS
]
df = rank_trends(trends)
df = df[df["final_score"] >= min_score].head(top_n)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trends Tracked", len(SAMPLE_TRENDS))
col2.metric("Emerging Trends", len(df[df["tier_label"] == "emerging"]))
col3.metric("Viral Trends", len(df[df["tier_label"] == "viral"]))
col4.metric("Avg Weeks to Mainstream", int(df["predicted_mainstream_weeks"].mean()))

st.markdown("---")

# Trend table
st.subheader("📊 Top Emerging Trends")
display_df = df[[
    "keyword", "tier", "final_score",
    "google_score", "reddit_score", "pinterest_score",
    "predicted_mainstream_weeks"
]].rename(columns={
    "keyword": "Fashion Keyword",
    "tier": "Trend Tier",
    "final_score": "Trend Score",
    "google_score": "Google",
    "reddit_score": "Reddit",
    "pinterest_score": "Pinterest",
    "predicted_mainstream_weeks": "Weeks to Mainstream"
})

st.dataframe(display_df, use_container_width=True)

# Bar chart
st.subheader("📈 Trend Score Comparison")
chart_data = df.set_index("keyword")[["google_score", "reddit_score", "pinterest_score"]]
st.bar_chart(chart_data)

# Footer
st.markdown("---")
st.markdown(
    "*TrendSense — B.E. Project | MITE Mangalore | "
    "RAR-BERT + CLIP + DBSCAN | Team of 4*"
)
