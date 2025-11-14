import time
import re
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from gnews import GNews

# -----------------------------
# STREAMLIT SETTINGS
# -----------------------------
st.set_page_config(page_title="Stock News & Sentiment Dashboard", layout="wide")

# -----------------------------
# LAZY VADER INITIALIZATION
# -----------------------------
_vader = None
def get_vader():
    global _vader
    if _vader is None:
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except Exception:
            _vader = None
    return _vader

def analyze_sentiment(text):
    analyzer = get_vader()
    if not analyzer:
        return "Neutral", "🟡", 0.0
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive", "🟢", score
    elif score < -0.2:
        return "Negative", "🔴", score
    else:
        return "Neutral", "🟡", score

# -----------------------------
# UI SETTINGS PANEL
# -----------------------------
st.sidebar.header("⚙️ Settings")
try:
    dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)
except Exception:
    dark_mode = st.sidebar.checkbox("🌗 Dark Mode", value=True)

# -----------------------------
# THEME COLORS
# -----------------------------
if dark_mode:
    bg_gradient = "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    text_color = "#EAEAEA"
    accent_color = "#00E676"
    plot_theme = "plotly_dark"
else:
    bg_gradient = "linear-gradient(135deg, #FFFFFF, #E0E0E0, #F5F5F5)"
    text_color = "#111111"
    accent_color = "#0078FF"
    plot_theme = "plotly_white"

st.markdown(
    f"""
    <style>
      body {{
        background: {bg_gradient};
        color: {text_color};
      }}
      .stApp {{
        background: {bg_gradient} !important;
        color: {text_color} !important;
      }}
      h1, h2, h3, h4 {{ color: {accent_color} !important; }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TIME FILTER
# -----------------------------
st.sidebar.header("📅 Filter Options")
time_period = st.sidebar.selectbox(
    "Select Time Period",
    ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"]
)

today = datetime.today()
if time_period == "Last Week":
    start_date = today - timedelta(days=7)
elif time_period == "Last Month":
    start_date = today - timedelta(days=30)
elif time_period == "Last 3 Months":
    start_date = today - timedelta(days=90)
else:
    start_date = today - timedelta(days=180)

# -----------------------------
# F&O STOCK LIST
# -----------------------------
fo_stocks = [
    "Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank",
    "State Bank of India", "HCL Technologies", "Wipro", "Larsen & Toubro",
    "Tata Motors", "Bajaj Finance", "Axis Bank", "NTPC", "ITC",
    "Adani Enterprises", "Coal India", "Power Grid", "Maruti Suzuki",
    "Tech Mahindra", "Sun Pharma"
]

# -----------------------------
# CACHED NEWS FETCHER
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(stock, start, end, max_results=40):
    try:
        g = GNews(language="en", country="IN", max_results=max_results)
        g.start_date, g.end_date = start, end
        articles = g.get_news(stock) or []

        # Dedupe by normalized title
        seen = set()
        unique = []
        for a in articles:
            title = (a.get("title") or "").strip()
            norm = re.sub(r"\W+", " ", title.lower()).strip()
            key = norm[:150]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_news(stocks, start, end, max_results=40):
    results = []
    with ThreadPoolExecutor(max_workers=4) as exe:
        tasks = {exe.submit(fetch_news, s, start, end, max_results): s for s in stocks}
        for fut in as_completed(tasks):
            stock = tasks[fut]
            try:
                arts = fut.result() or []
                results.append({"Stock": stock, "Articles": arts, "News Count": len(arts)})
            except Exception:
                results.append({"Stock": stock, "Articles": [], "News Count": 0})
    return results

# -----------------------------
# SCORING ENGINE
# (UNCHANGED — your original logic preserved)
# -----------------------------
WEIGHTS = {
    "earnings_guidance": 30, "M&A_JV": 25, "management_change": 20, "buyback_dividend": 20,
    "contract_deal": 25, "block_insider": 25, "policy_regulation": 20,
    "analyst_move": 15, "numeric_mentioned": 10, "trusted_source": 15,
    "speculative_penalty": -15, "low_quality_penalty": -10,
    "max_corroboration_bonus": 20,
}

HIGH_PRIORITY_KEYWORDS = {
    "earnings": ["earnings","quarter","q1","q2","q3","q4","revenue","profit","loss","guidance","outlook","beat","miss","results"],
    "MA": ["acquires","acquisition","merger","demerger","spin-off","joint venture","jv"],
    "management": ["appoint","resign","ceo","cfo","chairman","board","director","promoter"],
    "corp_action": ["buyback","dividend","split","bonus issue","rights issue","share pledge"],
    "contract": ["contract","order","tender","deal","agreement","license","wins order"],
    "regulatory": ["sebi","investigation","fraud","lawsuit","penalty","fine","policy","pli"],
    "analyst": ["upgrade","downgrade","target","recommendation","brokerage"],
    "block": ["block deal","bulk deal","insider","promoter buy","promoter sell"],
}

TRUSTED_SOURCES = {
    "reuters","bloomberg","economic times","economictimes","livemint","mint",
    "business standard","cnbc","financial times","ft","press release","nse","bse"
}

LOW_QUALITY = {"blog","medium","wordpress","forum","reddit","quora"}

SPECULATIVE = ["may","might","could","rumour","rumor","reportedly","possible","speculat"]

NUMERIC_RE = re.compile(r'[%₹$£€]|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', re.IGNORECASE)

def is_trusted(p):
    return p and any(x in p.lower() for x in TRUSTED_SOURCES)

def is_low(p):
    return p and any(x in p.lower() for x in LOW_QUALITY)

def has_numeric(t):
    return bool(NUMERIC_RE.search(t or ""))

def score_article(title, desc, publisher, corroboration_sources=None):
    t = f"{title} {desc}".lower()

    score = 0
    reasons = []

    def match(keys, weight, label):
        nonlocal score
        if any(k in t for k in keys):
            score += weight
            reasons.append(label)

    match(HIGH_PRIORITY_KEYWORDS["earnings"], WEIGHTS["earnings_guidance"], "Earnings")
    match(HIGH_PRIORITY_KEYWORDS["MA"], WEIGHTS["M&A_JV"], "M&A/JV")
    match(HIGH_PRIORITY_KEYWORDS["management"], WEIGHTS["management_change"], "Mgmt")
    match(HIGH_PRIORITY_KEYWORDS["corp_action"], WEIGHTS["buyback_dividend"], "Corp Action")
    match(HIGH_PRIORITY_KEYWORDS["contract"], WEIGHTS["contract_deal"], "Contract")
    match(HIGH_PRIORITY_KEYWORDS["regulatory"], WEIGHTS["policy_regulation"], "Regulatory")
    match(HIGH_PRIORITY_KEYWORDS["analyst"], WEIGHTS["analyst_move"], "Analyst")
    match(HIGH_PRIORITY_KEYWORDS["block"], WEIGHTS["block_insider"], "Block Deal")

    if has_numeric(t):
        score += WEIGHTS["numeric_mentioned"]
        reasons.append("Numeric")

    if is_trusted(publisher):
        score += WEIGHTS["trusted_source"]
        reasons.append("Trusted")

    if is_low(publisher):
        score += WEIGHTS["low_quality_penalty"]
        reasons.append("Low Quality")

    if any(w in t for w in SPECULATIVE):
        score += WEIGHTS["speculative_penalty"]
        reasons.append("Speculative")

    # Corroboration
    if corroboration_sources:
        trusted_count = sum(1 for s in set(corroboration_sources) if is_trusted(s))
        if trusted_count > 1:
            score += min(WEIGHTS["max_corroboration_bonus"], 5 * (trusted_count - 1))
            reasons.append("Corroborated")

    score = max(0, min(100, score))
    return score, reasons

