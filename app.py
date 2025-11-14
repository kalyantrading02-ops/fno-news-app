# app.py
import os
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
# USER-PROVIDED API KEY (embedded as requested)
# -----------------------------
# NOTE: For production prefer: st.secrets or environment variable.
NEWSAPI_KEY = "0d9d845466e14187b52b8717c1eb993f"

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
      .stButton button {{ border-radius: 6px; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💹 Stock Market News & Sentiment Dashboard")

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

# Performance knobs (kept as sidebar controls)
max_results = st.sidebar.slider("Max articles per stock (fetch)", 10, 80, 40, step=5)
workers = st.sidebar.slider("Concurrent fetch workers", 2, 8, 4, step=1)

# Optional manual refresh
if st.sidebar.button("🔁 Manual Refresh (clear caches)"):
    try:
        st.cache_data.clear()
        st.success("Caches cleared")
    except Exception:
        st.warning("Unable to clear caches in this Streamlit version")
    # clear session results cache too
    st.session_state.pop("all_results_cache", None)
    st.session_state.pop("last_applied_filters", None)

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
# UTILITIES: normalizer and numeric regex
# -----------------------------
NUMERIC_RE = re.compile(r'[%₹$£€]|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', re.IGNORECASE)

def _normalize_article(raw: Dict[str, Any], source_name_hint: str = "GNews") -> Dict[str, Any]:
    title = raw.get("title") or raw.get("headline") or ""
    desc = raw.get("description") or raw.get("snippet") or raw.get("summary") or ""
    url = raw.get("url") or raw.get("link") or raw.get("source_url") or raw.get("sourceUrl") or "#"
    pub = ""
    if isinstance(raw.get("publisher"), dict):
        pub = raw["publisher"].get("title") or ""
    elif raw.get("source"):
        pub = raw.get("source")
    elif raw.get("source_name"):
        pub = raw.get("source_name")
    else:
        pub = source_name_hint
    published = raw.get("published") or raw.get("published date") or raw.get("publishedAt") or ""
    return {"title": title, "description": desc, "url": url, "publisher": {"title": pub}, "published": published, "raw": raw}

# -----------------------------
# RESILIENT FETCHERS: GNews primary, NewsAPI fallback
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(stock: str, start: datetime, end: datetime, max_results_local: int = 40) -> List[Dict[str, Any]]:
    """
    Per-stock fetch. Cached to avoid repeated network calls for the same stock/time-window.
    """
    articles_out = []
    # 1) Try GNews
    try:
        g = GNews(language="en", country="IN", max_results=max_results_local)
        try:
            g.start_date, g.end_date = start, end
        except Exception:
            pass
        raw = g.get_news(stock) or []
        for r in raw:
            norm = _normalize_article(r, source_name_hint="GNews")
            if norm["title"] and norm["url"]:
                articles_out.append(norm)
    except Exception:
        articles_out = []

    # 2) Fallback: NewsAPI
    if (not articles_out) and NEWSAPI_KEY:
        try:
            q = f'"{stock}" OR {stock}'
            params = {
                "q": q,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": max_results_local,
                "apiKey": NEWSAPI_KEY
            }
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for a in data.get("articles", [])[:max_results_local]:
                    raw_norm = {
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "url": a.get("url"),
                        "publisher": {"title": (a.get("source") or {}).get("name") or "NewsAPI"},
                        "published": a.get("publishedAt"),
                        "raw": a
                    }
                    articles_out.append(raw_norm)
        except Exception:
            pass

    # Final dedupe by normalized title
    seen = set()
    unique = []
    for a in articles_out:
        t = (a.get("title") or "").strip()
        key = re.sub(r"\W+", " ", t.lower()).strip()[:150]
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    return unique

def fetch_all_news(stocks: List[str], start: datetime, end: datetime, max_results_local: int = 40, max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Concurrently fetch news for a list of stocks.
    This function is intentionally NOT cached (aggregated results may vary by filter).
    Accepts both positional and keyword usages matching calls elsewhere.
    """
    results = []
    worker_count = min(max_workers, max(1, len(stocks)))
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(fetch_news, s, start, end, max_results_local): s for s in stocks}
            for fut in as_completed(futures):
                stock = futures[fut]
                try:
                    arts = fut.result() or []
                    results.append({"Stock": stock, "Articles": arts, "News Count": len(arts)})
                except Exception as e:
                    # log to Streamlit UI for visibility; continue
                    st.warning(f"Error fetching news for {stock}: {str(e)[:200]}")
                    results.append({"Stock": stock, "Articles": [], "News Count": 0})
    except Exception as e:
        st.error(f"fetch_all_news fatal error: {str(e)[:300]}")
        return [{"Stock": s, "Articles": [], "News Count": 0} for s in stocks]
    return results

# -----------------------------
# SCORING ENGINE (UNCHANGED)
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

# -----------------------------
# RBI NEWS FETCHER (adds RBI-specific policy/news)
# -----------------------------
RBI_KEYWORDS = [
    "rbi","reserve bank of india","repo rate","policy rate","monetary policy","mpc","mpc meeting","m-pc","rbi governor",
    "rate cut","rate hike","bank rate","cash reserve ratio","crr","statutory liquidity ratio","slr","inflation target",
    "standing deposit facility"
]

@st.cache_data(ttl=600, show_spinner=False)
def fetch_rbi_news(start: datetime, end: datetime, max_results_local: int = 50) -> List[Dict[str, Any]]:
    out = []
    # Try GNews for RBI terms
    try:
        g = GNews(language="en", country="IN", max_results=max_results_local)
        try:
            g.start_date, g.end_date = start, end
        except Exception:
            pass
        raw = g.get_news("RBI") or g.get_news("Reserve Bank of India") or []
        for r in raw:
            norm = _normalize_article(r, source_name_hint="GNews")
            title_l = (norm["title"] or "").lower()
            desc_l = (norm["description"] or "").lower()
            if any(k in title_l or k in desc_l for k in RBI_KEYWORDS):
                out.append(norm)
    except Exception:
        out = []

    # NewsAPI fallback for RBI
    if NEWSAPI_KEY:
        try:
            params = {
                "q": "RBI OR \"Reserve Bank of India\" OR repo OR mpc OR \"monetary policy\"",
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_results_local,
                "apiKey": NEWSAPI_KEY
            }
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for a in data.get("articles", [])[:max_results_local]:
                    title = a.get("title","") or ""
                    desc = a.get("description","") or ""
                    if any(k in title.lower() or k in desc.lower() for k in RBI_KEYWORDS):
                        raw_norm = {"title": a.get("title"), "description": a.get("description"), "url": a.get("url"),
                                    "publisher": {"title": (a.get("source") or {}).get("name") or "NewsAPI"}, "published": a.get("publishedAt"), "raw": a}
                        out.append(raw_norm)
        except Exception:
            pass

    # dedupe
    final = []
    seen = set()
    for a in out:
        t = (a.get("title") or "").strip()
        key = re.sub(r"\W+", " ", t.lower()).strip()[:150]
        if not key or key in seen:
            continue
        seen.add(key)
        final.append(a)
    return final

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
st.session_state.setdefault("saved_articles", [])
st.session_state.setdefault("manual_events", [])
# optional cached aggregated results (not required but helpful)
# We don't auto-fill it here to keep behavior consistent; left for manual workflows if needed
# st.session_state.setdefault("all_results_cache", None)
# st.session_state.setdefault("last_applied_filters", None)

# -----------------------------
# INITIAL NEWS FETCH (lightweight - first 10 stocks)
# -----------------------------
initial_stocks = fo_stocks[:10]
with st.spinner("Fetching latest financial news (light)..."):
    try:
        raw_news_results = fetch_all_news(initial_stocks, start_date, today, max_results_local=max_results)
    except TypeError:
        # support alternate signatures if any other code called fetch_all_news differently
        raw_news_results = fetch_all_news(initial_stocks, start_date, today, max_results_local)
    except Exception as e:
        st.warning(f"Initial lightweight fetch failed: {str(e)[:200]}")
        raw_news_results = [{"Stock": s, "Articles": [], "News Count": 0} for s in initial_stocks]

# Normalize publishers and build headline_map for corroboration (from the same results)
news_results = []
headline_map = {}
for r in raw_news_results:
    stock = r.get("Stock", "")
    articles = r.get("Articles", []) or []
    filtered = []
    for art in articles:
        pub_field = art.get("publisher")
        pub_title = ""
        if isinstance(pub_field, dict):
            pub_title = (pub_field.get("title") or "").strip()
            art["publisher"] = {"title": pub_title}
        elif isinstance(pub_field, str):
            pub_title = pub_field.strip()
            art["publisher"] = {"title": pub_title}
        else:
            pub_title = (art.get("source") or "").strip()
            art["publisher"] = {"title": pub_title}
        if pub_title:
            filtered.append(art)

        title = art.get("title") or ""
        norm_head = re.sub(r'\W+', " ", title.lower()).strip()
        key = norm_head[:120] if norm_head else f"{stock.lower()}_{(title or '')[:40]}"
        headline_map.setdefault(key, []).append(pub_title or "unknown")

    news_results.append({"Stock": stock, "Articles": filtered, "News Count": len(filtered)})

# ---- Append RBI news to news_results so it appears in News tab and in events (preserve original feature)
try:
    rbi_hits = fetch_rbi_news(start_date, today, max_results_local=max_results) or []
    if rbi_hits:
        news_results.insert(0, {"Stock": "RBI (Policy)", "Articles": rbi_hits, "News Count": len(rbi_hits)})
        for art in rbi_hits:
            title = art.get("title") or ""
            norm_head = re.sub(r'\W+', " ", title.lower()).strip()
            key = norm_head[:120] if norm_head else f"rbi_{(title or '')[:40]}"
            headline_map.setdefault(key, []).append((art.get("publisher") or {}).get("title") or "RBI")
except Exception:
    pass

# -----------------------------
# TABS
# -----------------------------
news_tab, trending_tab, sentiment_tab, events_tab = st.tabs(
    ["📰 News", "🔥 Trending Stocks", "💬 Sentiment", "📅 Upcoming Events"]
)

# -----------------------------
# TAB: NEWS (keeps original UI and behavior)
# -----------------------------
with news_tab:
    st.header("🗞️ Latest Market News for F&O Stocks")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        only_impact = st.checkbox("🔎 Show only market-impacting news (score ≥ threshold)", value=True)
    with c2:
        threshold = st.slider("Minimum score to show", 0, 100, 40)
    with c3:
        show_snippet = st.checkbox("Show snippet", value=True)

    st.markdown("---")

    displayed_total = 0
    filtered_out_total = 0

    # Use the fetched `news_results` (first 10 stocks + RBI). Full list is fetched on-demand in Trending.
    for res in news_results:
        stock = res.get("Stock", "Unknown")
        articles = res.get("Articles", []) or []
        scored_list = []

        for art in articles:
            title = art.get("title") or ""
            desc = art.get("description") or art.get("snippet") or ""
            pub_field = art.get("publisher")
            publisher = ""
            if isinstance(pub_field, dict):
                publisher = pub_field.get("title") or ""
            elif isinstance(pub_field, str):
                publisher = pub_field
            else:
                publisher = art.get("source") or ""

            norm_head = re.sub(r'\W+', " ", (title or "").lower()).strip()
            key = norm_head[:120] if norm_head else f"{stock.lower()}_{(title or '')[:40]}"
            publishers_for_head = headline_map.get(key, [])

            score, reasons = score_article(title, desc, publisher, corroboration_sources=publishers_for_head)
            # RBI boost: if the article mentions RBI keywords, boost the score to increase visibility
            title_l = (title or "").lower()
            desc_l = (desc or "").lower()
            if any(k in title_l or k in desc_l for k in RBI_KEYWORDS):
                score = min(100, score + 30)
                reasons = reasons + ["RBI mention (boosted)"]

            scored_list.append({
                "title": title,
                "desc": desc,
                "publisher": publisher or "Unknown Source",
                "url": art.get("url") or art.get("link") or "#",
                "score": score,
                "reasons": reasons,
                "raw": art
            })

        if only_impact:
            visible = [s for s in scored_list if s["score"] >= threshold]
        else:
            visible = scored_list

        filtered_out_total += (len(scored_list) - len(visible))
        displayed_total += len(visible)

        with st.expander(f"🔹 {stock} ({len(visible)} Articles shown, scanned {len(scored_list)})", expanded=False):
            if visible:
                for idx, art in enumerate(visible[:10]):
                    title = art["title"]
                    url = art["url"]
                    publisher = art["publisher"]
                    pub_raw = art.get("raw", {}) if isinstance(art.get("raw", {}), dict) else {}
                    published_date = pub_raw.get("published date") or pub_raw.get("publishedDate") or pub_raw.get("published") or "N/A"
                    score = art["score"]
                    if score >= 70:
                        priority_label = "High"
                        priority_icon = "🔺"
                    elif score >= threshold:
                        priority_label = "Medium"
                        priority_icon = "🟨"
                    else:
                        priority_label = "Low"
                        priority_icon = "🟩"

                    reasons_txt = " • ".join(art["reasons"]) if art["reasons"] else "Signals detected"
                    sentiment_label, sentiment_emoji, s_score = analyze_sentiment(title + " " + (art.get("desc") or ""))

                    st.markdown(f"**[{title}]({url})** {priority_icon} *{priority_label} ({score})* 🏢 *{publisher}* | 🗓️ *{published_date}*")
                    st.markdown(f"*Reasons:* {reasons_txt} • *Sentiment:* {sentiment_emoji} {sentiment_label}")

                    if show_snippet and art.get("desc"):
                        snippet = art["desc"] if len(art["desc"]) < 220 else art["desc"][:217] + "..."
                        st.markdown(f"> {snippet}")

                    safe_stock = re.sub(r'\W+', '_', stock.lower())
                    save_key = f"save_{safe_stock}_{idx}_{abs(hash(url))}"
                    if st.button("💾 Save / Watch", key=save_key):
                        found = next((x for x in st.session_state["saved_articles"] if x["url"] == url), None)
                        if not found:
                            st.session_state["saved_articles"].append({
                                "title": title, "url": url, "stock": stock, "date": published_date, "score": score
                            })
                            st.success("Saved to Watchlist")
                        else:
                            st.info("Already in Watchlist")
                    st.markdown("---")
            else:
                st.info("No market-impacting news found for this stock in the selected time period.")

    st.markdown(f"**Summary:** Displayed **{displayed_total}** articles • Filtered out **{filtered_out_total}** • Scanned **{sum(len(r.get('Articles', [])) for r in news_results)}**")
    st.markdown("---")

    st.subheader("👀 Watchlist (Saved Articles)")
    if st.session_state["saved_articles"]:
        df_watch = pd.DataFrame(st.session_state["saved_articles"])
        if "date" in df_watch.columns:
            df_watch["date"] = df_watch["date"].astype(str)
        st.dataframe(df_watch[["stock", "title", "score", "date", "url"]], use_container_width=True)
    else:
        st.info("No saved articles yet — click 💾 Save / Watch on any article card.")

# -----------------------------
# TAB: TRENDING (fixed & robust) - includes RBI implicitly via news_results if present
# -----------------------------
with trending_tab:
    st.header(f"🔥 Trending F&O Stocks — News Impact ({time_period})")

    st.markdown(
        "This chart shows the **real effect** of news on each F&O stock for the selected time window. "
        "Bars = summed impact score (higher = more/stronger signals). Line = relative effect vs top stock (top = 100%)."
    )

    # Use cached aggregated results (if previously stored) otherwise do a light fetch for UI responsiveness
    all_results = st.session_state.get("all_results_cache")
    cache_meta = st.session_state.get("last_applied_filters")

    # If no cached aggregated results, do a light fetch (first 10) so the tab renders quickly
    if not all_results:
        with st.spinner("Loading a light snapshot for Trending (expand to fetch full dataset)..."):
            try:
                initial_stocks = fo_stocks[:10]
                try:
                    all_results = fetch_all_news(initial_stocks, start_date, today, max_results_local=10, max_workers=2)
                except TypeError:
                    # try signature fallback
                    try:
                        all_results = fetch_all_news(initial_stocks, start_date, today, 10)
                    except Exception:
                        all_results = fetch_all_news(initial_stocks, start_date, today)
            except Exception:
                all_results = [{"Stock": s, "Articles": [], "News Count": 0} for s in initial_stocks]

    # Build corroboration headline map from fetched data
    headline_map_full = {}
    for res in all_results:
        stock = res.get("Stock", "")
        for art in res.get("Articles", []) or []:
            title = art.get("title") or ""
            norm_head = re.sub(r'\W+', " ", title.lower()).strip()
            key = norm_head[:120] if norm_head else f"{stock.lower()}_{(title or '')[:40]}"
            pub = art.get("publisher")
            pub_name = ""
            if isinstance(pub, dict):
                pub_name = pub.get("title") or ""
            elif isinstance(pub, str):
                pub_name = pub
            else:
                pub_name = art.get("source") or ""
            headline_map_full.setdefault(key, []).append(pub_name or "unknown")

    # Aggregate metrics per stock (sum of article scores, count, avg)
    metrics = []
    for res in all_results:
        stock = res.get("Stock", "")
        articles = res.get("Articles") or []
        sum_score = 0.0
        count = 0
        for art in articles:
            title = art.get("title") or ""
            desc = art.get("description") or art.get("snippet") or ""
            pub_field = art.get("publisher")
            if isinstance(pub_field, dict):
                publisher = pub_field.get("title") or ""
            elif isinstance(pub_field, str):
                publisher = pub_field
            else:
                publisher = art.get("source") or ""

            norm_head = re.sub(r'\W+', " ", (title or "").lower()).strip()
            key = norm_head[:120] if norm_head else f"{stock.lower()}_{(title or '')[:40]}"
            pubs = headline_map_full.get(key, [])
            score, reasons = score_article(title, desc, publisher, corroboration_sources=pubs)

            sum_score += float(score)
            count += 1

        avg_score = (sum_score / count) if count else 0.0
        metrics.append({
            "Stock": stock,
            "SumScore": float(sum_score),
            "Count": int(count),
            "AvgScore": float(round(avg_score, 2)),
        })

    df_metrics = pd.DataFrame(metrics).sort_values("SumScore", ascending=False).reset_index(drop=True)

    if df_metrics.empty or df_metrics["SumScore"].sum() == 0:
        st.info("No impactful news detected for the current dataset. Use 'Apply Filters' (if you added) or broaden the timeperiod.")
        st.dataframe(df_metrics.head(20), use_container_width=True)
    else:
        top_val = df_metrics["SumScore"].max() if df_metrics["SumScore"].max() > 0 else 1
        df_metrics["Percent"] = (df_metrics["SumScore"] / top_val) * 100
        df_metrics["PercentLabel"] = df_metrics["Percent"].round(1).astype(str) + "%"

        # Chart: bar (percent label above bar) + line markers (no duplicate text)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_metrics["Stock"],
            y=df_metrics["SumScore"],
            name="Summed Impact Score",
            text=df_metrics["PercentLabel"],        # percent label only on bars
            textposition="outside",
            customdata=df_metrics[["Count"]].values,
            hovertemplate="<b>%{x}</b><br>Sum Score: %{y:.0f}<br>Articles: %{customdata[0]}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_metrics["Stock"],
            y=df_metrics["Percent"],
            name="Relative Effect (%)",
            yaxis="y2",
            mode="lines+markers",                   # no text on line (avoids duplicate percent text)
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Relative: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            template=plot_theme,
            title=dict(text=f"Trending F&O Stocks — News Impact ({time_period})", x=0.5),
            xaxis=dict(tickangle=-35),
            yaxis=dict(title="Summed Impact Score", rangemode="tozero"),
            yaxis2=dict(title="Relative Effect (%)", overlaying="y", side="right",
                        range=[0, max(110, df_metrics["Percent"].max() * 1.15)]),
            margin=dict(t=80, b=140),
            height=520
        )
        if dark_mode:
            fig.update_traces(selector=dict(type='bar'), textfont=dict(color="#FFFFFF"))
            fig.update_layout(font=dict(color="#EAEAEA"))
        else:
            fig.update_traces(selector=dict(type='bar'), textfont=dict(color="#111111"))
            fig.update_layout(font=dict(color="#111111"))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Market-impacting News Summary")
        df_display = df_metrics[["Stock", "SumScore", "Count", "AvgScore", "Percent"]].copy()
        df_display["SumScore"] = df_display["SumScore"].round(1)
        df_display["Percent"] = df_display["Percent"].round(1)
        st.dataframe(df_display.head(20).rename(columns={
            "SumScore": "Impact Score",
            "Count": "Articles Scanned",
            "AvgScore": "Avg per Article",
            "Percent": "Relative (%)"
        }), use_container_width=True)

# -----------------------------
# TAB: SENTIMENT (restored)
# -----------------------------
with sentiment_tab:
    st.header("💬 Sentiment Analysis")

    # Use cached aggregated results first (fast). If not present, do a light fetch snapshot.
    all_results = st.session_state.get("all_results_cache")
    if not all_results:
        # Try to reuse raw_news_results produced earlier (light fetch). Otherwise do a lightweight fetch of first 10 stocks.
        all_results = st.session_state.get("all_results_cache") or (raw_news_results if 'raw_news_results' in globals() else None)
        if not all_results:
            with st.spinner("Loading a quick sentiment snapshot (press Apply Filters for full data)..."):
                try:
                    initial_stocks = fo_stocks[:10]
                    try:
                        all_results = fetch_all_news(initial_stocks, start_date, today, max_results_local=10, max_workers=2)
                    except TypeError:
                        # signature fallback variants
                        try:
                            all_results = fetch_all_news(initial_stocks, start_date, today, 10)
                        except Exception:
                            all_results = fetch_all_news(initial_stocks, start_date, today)
                except Exception:
                    all_results = []

    # Compute sentiment for headlines (small sample per stock)
    with st.spinner("Analyzing sentiment..."):
        sentiment_rows = []
        for res in all_results:
            stock = res.get("Stock", "")
            for art in (res.get("Articles") or [])[:3]:
                title = art.get("title") or ""
                desc = art.get("description") or art.get("snippet") or ""
                combined = f"{title}. {desc}"
                s_label, emoji, s_score = analyze_sentiment(combined)
                sentiment_rows.append({
                    "Stock": stock,
                    "Headline": title,
                    "Sentiment": s_label,
                    "Emoji": emoji,
                    "Score": s_score
                })

        if sentiment_rows:
            sentiment_df = pd.DataFrame(sentiment_rows).sort_values(by=["Stock", "Score"], ascending=[True, False])
            st.dataframe(sentiment_df, use_container_width=True)
            csv_bytes = sentiment_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sentiment Data", csv_bytes, "sentiment_data.csv", "text/csv")
        else:
            st.warning("No sentiment data available for the current dataset. Press 'Apply Filters' (if added) to fetch full data for the selected timeframe.")

# -----------------------------
# PART C — Events tab, manual events, footer
# -----------------------------
from dateutil.parser import parse as dtparse

EVENT_WINDOW_DAYS = 90
EVENT_KEYWORDS = {
    "earnings": ["result", "results", "earnings", "q1", "q2", "q3", "q4", "quarterly results", "financial results", "results on", "declare", "declare on"],
    "board": ["board meeting", "board to meet", "board will meet", "board meeting on"],
    "dividend": ["ex-date", "ex date", "record date", "dividend", "dividend on", "dividend record"],
    "agm": ["agm", "annual general meeting", "egm", "extra ordinary general meeting"],
    "buyback": ["buyback", "buy-back", "tender offer", "acceptance date", "buyback record"],
    "ipo_listing": ["ipo", "listing", "to list", "list on"],
    "other": ["merger", "acquisition", "rights issue", "split", "bonus issue", "scheme of arrangement"],
}

DATE_PATTERNS = [
    r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b(?:[\s,]+\d{4})?',
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,)?\s*\d{0,4}\b',
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
    r'\b(next week|next month|tomorrow|today|this week|this month)\b'
]

def try_parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = dtparse(s, fuzzy=True)
        return dt
    except Exception:
        fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d %b", "%d %B"]
        for f in fmts:
            try:
                dt = datetime.strptime(s, f)
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.today().year)
                return dt
            except Exception:
                continue
    return None

def text_for_search(art):
    parts = []
    if art.get("title"):
        parts.append(art.get("title"))
    if art.get("description"):
        parts.append(art.get("description"))
    if art.get("snippet"):
        parts.append(art.get("snippet"))
    return " ".join(parts or [""]).lower()

# Extract events from news_results (company/corporate events)
events = []
for res in news_results:
    stock = res.get("Stock", "Unknown")
    for art in res.get("Articles", []) or []:
        txt = text_for_search(art)
        if not txt.strip():
            continue

        matched_types = []
        for etype, kws in EVENT_KEYWORDS.items():
            for kw in kws:
                if kw in txt:
                    matched_types.append(etype)
                    break

        if not matched_types:
            continue

        found_dates = []
        for patt in DATE_PATTERNS:
            for m in re.finditer(patt, txt, flags=re.IGNORECASE):
                cand = m.group(0)
                parsed = try_parse_date(cand)
                if parsed:
                    found_dates.append(parsed)
                else:
                    rel = cand.lower()
                    now = datetime.now()
                    if "tomorrow" in rel:
                        found_dates.append(now + timedelta(days=1))
                    elif "today" in rel:
                        found_dates.append(now)
                    elif "next week" in rel:
                        found_dates.append(now + timedelta(days=7))
                    elif "next month" in rel:
                        found_dates.append(now + timedelta(days=30))

        if not found_dates:
            m = re.search(r'on ([A-Za-z0-9 ,\-thstndrd]{3,40})', txt)
            if m:
                cand = m.group(1)
                parsed = try_parse_date(cand)
                if parsed:
                    found_dates.append(parsed)

        for dt in found_dates:
            if not isinstance(dt, datetime):
                continue
            if dt.date() < datetime.now().date():
                continue
            if (dt - datetime.now()).days > EVENT_WINDOW_DAYS:
                continue

            etype_label = matched_types[0] if matched_types else "update"
            desc = art.get("title") or art.get("description") or ""
            pub = art.get("publisher")
            source = ""
            if isinstance(pub, dict):
                source = pub.get("title") or ""
            elif isinstance(pub, str):
                source = pub
            else:
                source = art.get("source") or ""
            url = art.get("url") or art.get("link") or "#"
            priority = "Normal"
            try:
                if is_trusted(source):
                    priority = "High"
            except Exception:
                priority = "Normal"

            events.append({
                "stock": stock,
                "type": etype_label,
                "desc": desc,
                "date": dt,
                "source": source,
                "url": url,
                "priority": priority
            })

# Deduplicate events by (stock, type, date)
unique = {}
for e in events:
    key = (e["stock"], e["type"], e["date"].date())
    if key not in unique:
        unique[key] = e
    else:
        existing = unique[key]
        if e["source"] and e["source"] not in existing.get("source", ""):
            existing["source"] += f"; {e['source']}"

events = sorted(unique.values(), key=lambda x: (x["date"], x.get("priority", "") == "High"))

# Include manual events from session state
manual = st.session_state.get("manual_events", []) or []
for me in manual:
    events.append({
        "stock": me.get("stock", "Manual"),
        "type": me.get("type", "manual"),
        "desc": me.get("desc", ""),
        "date": me.get("date"),
        "source": "Manual",
        "url": "#",
        "priority": me.get("priority", "Normal")
    })

events = sorted(events, key=lambda x: (x["date"] if isinstance(x["date"], datetime) else datetime.max))

# Events tab UI
with events_tab:
    st.subheader(f"📅 Upcoming Market-Moving Events (next {EVENT_WINDOW_DAYS} days) — extracted from news")

    # Use cached aggregated results first; fallback to light snapshot
    all_results = st.session_state.get("all_results_cache")
    if not all_results:
        with st.spinner("Loading events snapshot (use Apply Filters to refresh full data)..."):
            try:
                initial_stocks = fo_stocks[:10]
                try:
                    all_results = fetch_all_news(initial_stocks, start_date, today, max_results_local=10, max_workers=2)
                except TypeError:
                    all_results = fetch_all_news(initial_stocks, start_date, today, 10)
            except Exception:
                all_results = []

    # Build events from all_results (same extraction logic as before)
    events = []
    for res in all_results:
        stock = res.get("Stock", "Unknown")
        for art in res.get("Articles", []) or []:
            txt = text_for_search(art)
            if not txt.strip():
                continue

            matched_types = []
            for etype, kws in EVENT_KEYWORDS.items():
                for kw in kws:
                    if kw in txt:
                        matched_types.append(etype)
                        break

            if not matched_types:
                continue

            found_dates = []
            for patt in DATE_PATTERNS:
                for m in re.finditer(patt, txt, flags=re.IGNORECASE):
                    cand = m.group(0)
                    parsed = try_parse_date(cand)
                    if parsed:
                        found_dates.append(parsed)
                    else:
                        rel = cand.lower()
                        now = datetime.now()
                        if "tomorrow" in rel:
                            found_dates.append(now + timedelta(days=1))
                        elif "today" in rel:
                            found_dates.append(now)
                        elif "next week" in rel:
                            found_dates.append(now + timedelta(days=7))
                        elif "next month" in rel:
                            found_dates.append(now + timedelta(days=30))

            if not found_dates:
                m = re.search(r'on ([A-Za-z0-9 ,\-thstndrd]{3,40})', txt)
                if m:
                    cand = m.group(1)
                    parsed = try_parse_date(cand)
                    if parsed:
                        found_dates.append(parsed)

            for dt in found_dates:
                if not isinstance(dt, datetime):
                    continue
                if dt.date() < datetime.now().date():
                    continue
                if (dt - datetime.now()).days > EVENT_WINDOW_DAYS:
                    continue

                etype_label = matched_types[0] if matched_types else "update"
                desc = art.get("title") or art.get("description") or ""
                pub = art.get("publisher")
                source = ""
                if isinstance(pub, dict):
                    source = pub.get("title") or ""
                elif isinstance(pub, str):
                    source = pub
                else:
                    source = art.get("source") or ""
                url = art.get("url") or art.get("link") or "#"
                priority = "Normal"
                try:
                    if is_trusted(source):
                        priority = "High"
                except Exception:
                    priority = "Normal"

                events.append({
                    "stock": stock,
                    "type": etype_label,
                    "desc": desc,
                    "date": dt,
                    "source": source,
                    "url": url,
                    "priority": priority
                })

    # Deduplicate and combine with manual events from session state (same logic)
    unique = {}
    for e in events:
        key = (e["stock"], e["type"], e["date"].date())
        if key not in unique:
            unique[key] = e
        else:
            existing = unique[key]
            if e["source"] and e["source"] not in existing.get("source", ""):
                existing["source"] += f"; {e['source']}"

    events = sorted(unique.values(), key=lambda x: (x["date"], x.get("priority", "") == "High"))

    manual = st.session_state.get("manual_events", []) or []
    for me in manual:
        events.append({
            "stock": me.get("stock", "Manual"),
            "type": me.get("type", "manual"),
            "desc": me.get("desc", ""),
            "date": me.get("date"),
            "source": "Manual",
            "url": "#",
            "priority": me.get("priority", "Normal")
        })

    events = sorted(events, key=lambda x: (x["date"] if isinstance(x["date"], datetime) else datetime.max))

    # UI rendering
    if events:
        rows = []
        for e in events:
            rows.append({
                "Stock": e["stock"],
                "Event": e["type"].title(),
                "When": e["date"].strftime("%Y-%m-%d %H:%M") if isinstance(e["date"], datetime) else str(e["date"]),
                "Priority": e.get("priority", "Normal"),
                "Source": e.get("source", ""),
                "Link": e.get("url", "#")
            })
        df_events = pd.DataFrame(rows)
        st.dataframe(df_events, use_container_width=True)
        st.download_button("📥 Download Extracted Events (CSV)", df_events.to_csv(index=False).encode("utf-8"), "extracted_events.csv", "text/csv")

        for e in events[:10]:
            date_str = e["date"].strftime("%Y-%m-%d") if isinstance(e["date"], datetime) else str(e["date"])
            st.markdown(f"- **{e['stock']}** — *{e['type'].title()}* on **{date_str}** — *{e['priority']}* — [{e['source']}]({e['url']})")
    else:
        st.info("No upcoming company updates found from recent news in cache. Press 'Apply Filters' (if implemented) to fetch full data if needed.")

# FOOTER
st.markdown("---")
st.caption(f"📊 Data Source: Google News + NewsAPI (fallback) | Mode: {'Dark' if dark_mode else 'Light'} | Auto-refresh: session-cached | Built with Streamlit & Plotly")
