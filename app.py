# app.py (optimized merge)
import time
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
from gnews import GNews
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests  # NEW: used for Finnhub calendar fetch
from typing import List, Dict, Any, Optional

# -----------------------------
# INITIAL SETUP
# -----------------------------
# Guarded NLTK download - will be quiet if already present
try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass

st.set_page_config(page_title="Stock News & Sentiment Dashboard", layout="wide")

# Create analyzer if possible; if it fails, handle later with fallback
analyzer = None
try:
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    analyzer = None

# --------- HELPERS & OPTIMIZATIONS ----------
@st.cache_resource
def get_gnews_instance(language="en", country="IN", max_results=20):
    try:
        return GNews(language=language, country=country, max_results=max_results)
    except Exception:
        return None

def normalize_key(text: str):
    return re.sub(r"\W+", " ", (text or "").lower()).strip()

def simple_sentiment(text: str):
    if not text:
        return 0.0
    pos = {"good","great","positive","up","gain","profit","beat","beats","win","surge","strong"}
    neg = {"bad","down","loss","decline","miss","fall","worse","fraud","drop","weak"}
    words = set(re.findall(r"\w+", (text or "").lower()))
    score = len(words & pos) - len(words & neg)
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / (len(words)**0.5)))

def compute_sentiment(text: str):
    # Use NLTK VADER if available; otherwise simple fallback
    if analyzer:
        try:
            return float(analyzer.polarity_scores(text).get("compound", 0.0))
        except Exception:
            return simple_sentiment(text)
    else:
        return simple_sentiment(text)

# -----------------------------
# FINNHUB: Upcoming Events Fetcher (NEW FEATURE)
# -----------------------------
# This is non-intrusive: used only in the Upcoming Events tab if user provides a key.
def _iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def fetch_finnhub_economic_calendar(api_key: str, start: datetime, end: datetime, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch economic calendar from Finnhub between start and end (inclusive).
    Returns normalized list with keys: date (datetime or None), title, country, impact, raw.
    """
    events = []
    if not api_key:
        return events
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {
        "from": _iso_date(start),
        "to": _iso_date(end),
        "token": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception as e:
        # return empty and show error to user at UI time
        return [{"error": str(e)}]

    # Finnhub returns a dict that may contain 'economic'
    raw_events = []
    if isinstance(data, dict):
        if "economic" in data and isinstance(data["economic"], list):
            raw_events = data["economic"]
        else:
            # some responses might be list-like
            for v in data.values():
                if isinstance(v, list):
                    raw_events = v
                    break
    elif isinstance(data, list):
        raw_events = data

    for ev in raw_events:
        # attempt to read multiple possible keys
        date_raw = ev.get("date") or ev.get("eventDate") or ev.get("time") or ev.get("datetime")
        event_date = None
        try:
            if date_raw:
                event_date = pd.to_datetime(date_raw).to_pydatetime()
        except Exception:
            event_date = None
        title = ev.get("title") or ev.get("event") or ev.get("name") or ev.get("description") or ""
        country_ev = (ev.get("country") or ev.get("countryCode") or "").upper()
        impact = ev.get("impact") or ev.get("importance") or ""
        events.append({
            "date": event_date,
            "title": title,
            "country": country_ev,
            "impact": impact,
            "raw": ev
        })
    if country:
        country_up = country.strip().upper()
        events = [e for e in events if not e.get("country") or e.get("country").startswith(country_up)]
    events.sort(key=lambda x: (x["date"] or datetime.max))
    return events

# -----------------------------
# APPLY THEMES (CSS) (unchanged)
# -----------------------------
# Sidebar toggle compatibility across Streamlit versions
st.sidebar.header("⚙️ Settings")
try:
    dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True, help="Switch instantly between Dark & Light Mode")
except Exception:
    dark_mode = st.sidebar.checkbox("🌗 Dark Mode", value=True, help="Switch instantly between Dark & Light Mode")

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
body {{ background: {bg_gradient}; color: {text_color}; }}
.stApp {{ background: {bg_gradient} !important; color: {text_color} !important; }}
h1, h2, h3, h4, h5 {{ color: {accent_color} !important; }}
.stButton button {{ background-color: {accent_color} !important; color: black !important; border-radius: 6px; }}
.stDataFrame {{ border-radius: 10px; background-color: rgba(255,255,255,0.02); }}
.news-card {{ border-radius: 10px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); transition: transform .12s ease, box-shadow .12s ease; background: rgba(255,255,255,0.02); margin-bottom: 12px; }}
.news-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 30px rgba(0,0,0,0.16); }}
.headline {{ font-weight:600; font-size:16px; margin-bottom:6px; }}
.meta {{ font-size:12px; color: #9aa0a6; margin-bottom:8px; }}
.snip {{ font-size:13px; color: #dfe6ea; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; margin-right:6px; }}
.badge-source {{ background: rgba(255,255,255,0.04); color:inherit; }}
.badge-pos {{ background: rgba(0,200,83,0.12); color:#00C853; }}
.badge-neu {{ background: rgba(255,193,7,0.08); color:#FFC107; }}
.badge-neg {{ background: rgba(239,83,80,0.08); color:#EF5350; }}
.priority-high {{ background: rgba(239,83,80,0.12); color:#EF5350; padding:6px 10px; border-radius:8px; font-weight:600; }}
.priority-med {{ background: rgba(255,193,7,0.08); color:#FFC107; padding:6px 10px; border-radius:8px; font-weight:600; }}
.priority-low {{ background: rgba(128,128,128,0.06); color:#9aa0a6; padding:6px 10px; border-radius:8px; font-weight:600; }}
.reason-chip {{ display:inline-block; margin:3px 4px; padding:4px 8px; border-radius:999px; font-size:12px; background: rgba(255,255,255,0.03); }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# APP TITLE (unchanged)
# -----------------------------
st.title("💹 Stock Market News & Sentiment Dashboard")

# -----------------------------
# AUTO REFRESH EVERY 10 MIN (robust) (unchanged)
# -----------------------------
refresh_interval = 600  # 10 minutes
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
else:
    if time.time() - st.session_state["last_refresh"] > refresh_interval:
        st.session_state["last_refresh"] = time.time()
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                st.warning("Auto-refresh is unavailable in this Streamlit version. Please refresh manually.")
                st.stop()

# -----------------------------
# SIDEBAR FILTERS (unchanged)
# -----------------------------
st.sidebar.header("📅 Filter Options")
time_period = st.sidebar.selectbox("Select Time Period", ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"])

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
# F&O STOCK LIST (unchanged)
# -----------------------------
fo_stocks = [
    "Reliance Industries",
    "TCS",
    "Infosys",
    "HDFC Bank",
    "ICICI Bank",
    "State Bank of India",
    "HCL Technologies",
    "Wipro",
    "Larsen & Toubro",
    "Tata Motors",
    "Bajaj Finance",
    "Axis Bank",
    "NTPC",
    "ITC",
    "Adani Enterprises",
    "Coal India",
    "Power Grid",
    "Maruti Suzuki",
    "Tech Mahindra",
    "Sun Pharma",
]

# -----------------------------
# OPTIMIZED FETCH / CACHE LAYER
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(stock, start, end, max_results=50):
    """
    Optimized fetch_news: reuses GNews instance, deduplicates titles, and returns normalized article list.
    """
    try:
        gnews = get_gnews_instance(max_results=max_results)
        if not gnews:
            # fallback: try creating a local instance
            gnews = GNews(language="en", country="IN", max_results=max_results)
    except Exception:
        try:
            gnews = GNews(language="en", country="IN", max_results=max_results)
        except Exception:
            return []
    try:
        # some GNews versions accept start_date/end_date attributes; ignore failures
        try:
            gnews.start_date, gnews.end_date = start, end
        except Exception:
            pass
        raw = gnews.get_news(stock) or []
    except Exception:
        raw = []
    seen = set()
    out = []
    for item in raw:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        key = normalize_key(title)[:200]
        if key in seen:
            continue
        seen.add(key)
        publisher = item.get("publisher") or item.get("source") or {}
        pub_name = publisher.get("title") if isinstance(publisher, dict) else (publisher or "")
        published = item.get("published", "") or item.get("publishedAt", "") or item.get("time", "")
        out.append({
            "title": title,
            "published": published,
            "description": item.get("description") or item.get("content") or "",
            "url": item.get("url") or item.get("link") or "",
            "publisher_name": pub_name or "",
            "raw": item
        })
        if len(out) >= max_results:
            break
    return out


@st.cache_data(ttl=300)
def fetch_news_for_stock(stock: str, start_date: str, end_date: str, max_results=20):
    """
    Compatibility wrapper for other parts of the app that used a different signature.
    """
    return fetch_news(stock, start_date, end_date, max_results=max_results)

def fake_articles_for(stock, n=4):
    """Fallback sample data if GNews not available."""
    now = datetime.utcnow()
    samples = []
    for i in range(n):
        t = (now - timedelta(hours=3 * i)).isoformat()
        samples.append({
            "title": f"{stock} — Sample headline #{i+1}",
            "published": t,
            "description": f"Sample description for {stock} article {i+1}.",
            "url": "",
            "publisher_name": "SampleNews",
        })
    return samples

def fetch_all_news(stocks, start, end, max_results=20, workers=6):
    """
    Fetches news for a list of stocks (parallel with limited workers),
    dedupes, computes sentiment & score once per article, and returns a list of
    {Stock, Articles, News Count} along with a headline_map.
    """
    t0 = time.time()
    results = []
    if not stocks:
        return [], {}
    # Parallel fetch (each fetch_news is cached individually)
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(stocks)))) as ex:
        futures = {ex.submit(fetch_news_for_stock, s, start, end, max_results): s for s in stocks}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                arts = fut.result()
                if not arts:
                    arts = fake_articles_for(s, n=min(4, max_results))
            except Exception:
                arts = fake_articles_for(s, n=min(4, max_results))
            results.append({"Stock": s, "Articles": arts, "News Count": len(arts)})

    # Compute sentiment and score once per article and build headline_map
    headline_map = {}
    for res in results:
        for art in res.get("Articles", []) or []:
            title = art.get("title") or ""
            desc = art.get("description") or art.get("snippet") or ""
            combined = f"{title}. {desc}"
            art["_sentiment"] = float(compute_sentiment(combined))
            # Score: lightweight rule-based recency+keyword scoring (0-100 scale)
            sc = 50.0
            try:
                # recency heuristic
                pub = art.get("published") or ""
                days = 999.0
                if pub:
                    try:
                        pub_dt = pd.to_datetime(pub)
                        days = max(0.0, (datetime.utcnow() - pub_dt.to_pydatetime()).total_seconds() / (3600 * 24))
                    except Exception:
                        days = 7.0
                recency = max(0.0, 1.0 - (days / 14.0))
                sc = 50.0 + (recency * 40.0)
            except Exception:
                sc = 50.0
            title_l = title.lower() if title else ""
            if any(k in title_l for k in ["upgrade", "beats", "record", "acquires", "appoints", "surge"]):
                sc += 20
            if any(k in title_l for k in ["decline", "loss", "cuts", "lawsuit", "fraud", "recall", "drop", "weak"]):
                sc -= 20
            art["_score"] = float(max(0.0, min(100.0, sc)))
            key = normalize_key(title)[:120] or normalize_key(desc)[:120]
            headline_map.setdefault(key, []).append(art.get("publisher_name") or "unknown")

    elapsed = time.time() - t0
    meta = {"fetched_seconds": elapsed, "stocks": len(stocks), "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
    return {"results": results, "headline_map": headline_map, "meta": meta}

# -----------------------------
# UI LAYOUT & TABS
# -----------------------------
st.title("Stock News & Sentiment — Optimized")

# Controls
col1, col2, col3 = st.columns([3,2,2])
with col1:
    stocks_raw = st.text_input("Stocks / keywords (comma separated)", value="TCS, INFY, WIPRO, HCLTECH")
    stocks = [s.strip() for s in stocks_raw.split(",") if s.strip()][:20]
with col2:
    max_results_ui = st.number_input("Max articles per keyword", min_value=3, max_value=50, value=20, step=1)
with col3:
    workers_ui = st.number_input("Max worker threads", min_value=1, max_value=20, value=6, step=1)

today = datetime.utcnow().date()
start_date_ui = st.date_input("Start date", value=(today - timedelta(days=7)))
end_date_ui = st.date_input("End date", value=today)

if not stocks:
    st.warning("Please enter at least one keyword.")
    st.stop()

fetch_now = st.button("Fetch Now (bust cache)")

# Manual cache bust mechanism if user clicks button
if fetch_now:
    try:
        st.experimental_memo.clear()
    except Exception:
        try:
            st.cache_data.clear()
        except Exception:
            pass
    st.experimental_rerun()

# Fetch data (cached internally)
with st.spinner("Fetching and processing news (cached where possible)..."):
    data_package = fetch_all_news(stocks, start_date_ui.isoformat(), end_date_ui.isoformat(), max_results=max_results_ui, workers=workers_ui)

news_results = data_package.get("results", [])
headline_map = data_package.get("headline_map", {})
meta = data_package.get("meta", {})

st.markdown(f"**Fetched:** {meta.get('stocks',0)} keywords in {meta.get('fetched_seconds',0):.2f}s  —  {meta.get('ts')}")
st.write("VADER available:" if analyzer else "VADER not available — using fallback sentiment.")

# Top-level summary
total_articles = sum([r.get("News Count", 0) for r in news_results])
colA, colB, colC = st.columns(3)
colA.metric("Keywords", len(stocks))
colB.metric("Total Articles (approx)", total_articles)
colC.metric("Last fetch (UTC)", meta.get("ts"))

# Tabs
news_tab, trending_tab, sentiment_tab, events_tab = st.tabs(["News", "Trending", "Sentiment", "Upcoming Events"])

def render_article_card(article, stock=None):
    title = article.get("title") or ""
    desc = article.get("description") or ""
    pub = article.get("publisher_name") or article.get("publisher") or ""
    published = article.get("published") or ""
    url = article.get("url") or ""
    sent = article.get("_sentiment", 0.0)
    sc = article.get("_score", 0.0)
    with st.expander(f"{title} — {pub} [{published[:10]}]"):
        st.write(desc)
        st.write(f"**Sentiment:** {sent:.3f} — **Score:** {sc:.1f}")
        if url:
            st.markdown(f"[Source link]({url})")
        if stock:
            st.caption(f"Keyword: {stock}")

# TAB 1 — NEWS
with news_tab:
    st.header("📰 Latest News (grouped by keyword)")
    displayed_total = 0
    filtered_out_total = 0
    for res in sorted(news_results, key=lambda x: x.get("Stock","")):
        stock = res.get("Stock")
        articles = res.get("Articles") or []
        if not articles:
            st.info(f"No articles found for {stock}.")
            continue
        st.write(f"### {stock} — {len(articles)} items")
        # show top by score
        for art in sorted(articles, key=lambda a: a.get("_score", 0.0), reverse=True)[:10]:
            render_article_card(art, stock=stock)
            displayed_total += 1
    st.markdown(f"**Summary:** Displayed **{displayed_total}** articles • Filtered out **{filtered_out_total}** • Scanned **{sum(len(r.get('Articles', [])) for r in news_results)}**")
    st.markdown("---")
    st.subheader("👀 Watchlist (Saved Articles)")
    if "saved_articles" not in st.session_state:
        st.session_state["saved_articles"] = []
    if st.session_state["saved_articles"]:
        df_watch = pd.DataFrame(st.session_state["saved_articles"])
        if "date" in df_watch.columns:
            df_watch["date"] = df_watch["date"].astype(str)
        st.dataframe(df_watch[["stock", "title", "score", "date", "url"]], use_container_width=True)
    else:
        st.info("No saved articles yet — click 💾 Save / Watch on any article card.")

# TAB 2 — TRENDING (market-impacting news only)
with trending_tab:
    st.header(f"🔥 Trending F&O Stocks by Market-Impacting News — {time_period}")

    # Choose threshold for "market-impacting" — change this number if you want stricter/looser filtering
    impact_threshold = 70  # score on 0-100

    with st.spinner("Filtering for market-impacting items..."):
        counts = []
        for res in news_results:
            stock_name = res.get("Stock", "")
            articles = res.get("Articles") or []
            impactful_count = 0
            for art in articles:
                title = art.get("title") or ""
                desc = art.get("description") or art.get("snippet") or ""
                norm_head = re.sub(r'\W+', " ", (title or "").lower()).strip()
                key = norm_head[:120] if norm_head else f"{stock_name.lower()}_{(title or '')[:40]}"
                publishers_for_head = headline_map.get(key, [])
                score = art.get("_score", 0.0)
                if score >= impact_threshold:
                    impactful_count += 1
            counts.append({"Stock": stock_name, "News Count": int(impactful_count)})

        df_counts = pd.DataFrame(counts).sort_values("News Count", ascending=False).reset_index(drop=True)

    if df_counts.empty:
        st.info("No data available — try changing the time period or increasing max_results in fetcher.")
    else:
        if df_counts["News Count"].sum() == 0:
            df_counts["Label"] = df_counts["News Count"].astype(str)
            y_field = "News Count"
            hover_template_extra = "%{y}"
            yaxis_title = "Market-impacting News Mentions (count)"
        else:
            top_value = df_counts["News Count"].max() if df_counts["News Count"].max() > 0 else 1
            df_counts["Percent"] = (df_counts["News Count"] / top_value) * 100
            df_counts["Label"] = df_counts["Percent"].round(1).astype(str) + "%"
            y_field = "Percent"
            hover_template_extra = "%{y:.1f}%"
            yaxis_title = "Relative Popularity (%) (top = 100%)"

        palette = ["#0078FF", "#00C853", "#EF5350", "#9C27B0", "#FF9800", "#00BCD4", "#8BC34A", "#9E9E9E"]
        colors = [palette[i % len(palette)] for i in range(len(df_counts))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_counts["Stock"],
            y=df_counts[y_field],
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.4)', width=1.25)),
            text=df_counts["Label"],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: ' + hover_template_extra + '<extra></extra>',
        ))
        fig.update_layout(title_text="Trending: Market-impacting Mentions", showlegend=False, template=plot_theme, yaxis_title=yaxis_title)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Raw counts")
        st.dataframe(df_counts, use_container_width=True)

        top_nonzero = df_counts[df_counts["News Count"] > 0].head(3)
        if not top_nonzero.empty:
            st.success(f"🚀 Top Trending (market-impacting): {', '.join(top_nonzero['Stock'].tolist())}")
            st.caption(f"Showing articles with score ≥ {impact_threshold}. Adjust `impact_threshold` in the code to tune sensitivity.")
        else:
            st.info("No market-impacting news found in the selected timeframe (all counts are 0).")

# TAB 3 — SENTIMENT
with sentiment_tab:
    st.header("💬 Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment_data = []
        for res in news_results:
            stock = res.get("Stock", "Unknown")
            for art in res.get("Articles", [])[:5]:
                title = art.get("title") or ""
                desc = art.get("description") or art.get("snippet") or ""
                combined = f"{title}. {desc}"
                sent_score = art.get("_sentiment", compute_sentiment(combined))
                label = "Positive" if sent_score > 0.2 else ("Negative" if sent_score < -0.2 else "Neutral")
                emoji = "😊" if sent_score > 0.2 else ("😟" if sent_score < -0.2 else "😐")
                sentiment_data.append({"Stock": stock, "Headline": title, "Sentiment": label, "Emoji": emoji, "Score": sent_score})
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data).sort_values(by=["Stock", "Score"], ascending=[True, False])
            st.dataframe(sentiment_df, use_container_width=True)
            csv_bytes = sentiment_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sentiment Data", csv_bytes, "sentiment_data.csv", "text/csv")
        else:
            st.warning("No sentiment data found for the selected timeframe.")

# TAB 4 — UPCOMING EVENTS (company events extracted from headlines)
EVENT_KEYWORDS = {
    "results": ["results", "earnings", "q1", "q2", "q3", "q4", "quarterly results", "financial results", "results on", "declare", "declare on"],
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
        from dateutil.parser import parse as dtparse
        return dtparse(s, fuzzy=True)
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
            # try date patterns
            for pat in DATE_PATTERNS:
                m = re.search(pat, txt)
                if m:
                    matched_types.append("other")
                    break
        if matched_types:
            # parse a candidate date if present
            date_candidate = None
            for pat in DATE_PATTERNS:
                m = re.search(pat, txt)
                if m:
                    date_candidate = try_parse_date(m.group(0))
                    break
            events.append({
                "stock": stock,
                "type": matched_types[0],
                "date": date_candidate,
                "source": art.get("publisher_name") or "",
                "url": art.get("url") or "",
                "headline": art.get("title") or ""
            })

EVENT_WINDOW_DAYS = 30
with events_tab:
    st.subheader(f"📅 Upcoming Market-Moving Events (next {EVENT_WINDOW_DAYS} days) — {len(events)} found (from news)")

    st.markdown("### Events extracted from news headlines (company / corporate events)")
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
        st.download_button(
            "📥 Download Extracted Events (CSV)",
            df_events.to_csv(index=False).encode("utf-8"),
            "extracted_events.csv",
            "text/csv"
        )
        for e in events[:10]:
            date_str = e["date"].strftime("%Y-%m-%d") if isinstance(e["date"], datetime) else str(e["date"])
            st.markdown(f"- **{e['stock']}** — *{e['type'].title()}* on **{date_str}** — [{e['source']}]({e['url']})")
    else:
        st.info("No upcoming company updates found from recent news. Add manually if needed.")

    with st.expander("➕ Add manual event"):
        m_stock = st.text_input("Stock name / company")
        m_type = st.selectbox(
            "Event type",
            ["Earnings/Results", "Board Meeting", "Ex-dividend / Record Date", "AGM/EGM", "Buyback", "IPO/Listing", "Other"]
        )
        m_date = st.date_input("Event date", value=datetime.now().date() + timedelta(days=7))
        m_desc = st.text_area("Short description (optional)")
        m_priority = st.selectbox("Priority", ["Normal", "High"])
        if st.button("Add event to watchlist"):
            st.session_state.setdefault("manual_events", [])
            st.session_state["manual_events"].append({
                "stock": m_stock,
                "type": m_type,
                "date": datetime.combine(m_date, datetime.min.time()),
                "desc": m_desc,
                "priority": m_priority
            })
            st.success("Manual event added (session only). It will appear in Upcoming Events on next refresh.")

# -----------------------------
# PREMIUM FOOTER (replace old simple footer)
# -----------------------------
footer_html = f"""
<style>
/* Premium footer */
.premium-footer {{
  display: flex;
  gap: 18px;
  align-items: center;
  justify-content: space-between;
  padding: 18px 20px;
  margin-top: 18px;
  border-radius: 12px;
  backdrop-filter: blur(6px) saturate(120%);
  -webkit-backdrop-filter: blur(6px) saturate(120%);
  box-shadow: 0 8px 30px rgba(2,6,23,0.45);
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  color: {text_color};
  border: 1px solid rgba(255,255,255,0.03);
}}
.footer-left, .footer-center, .footer-right {{
  display:flex;
  align-items:center;
  gap:12px;
}}
.footer-left .logo {{
  width:44px;
  height:44px;
  border-radius:10px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  font-weight:700;
  font-size:18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 6px 18px rgba(0,0,0,0.25) inset;
}}
.footer-title {{ font-weight:700; font-size:14px; }}
.footer-sub {{ font-size:12px; color: rgba(255,255,255,0.6); }}

.footer-center a {{
  font-size:13px;
  color: {text_color};
  text-decoration: none;
  padding:6px 10px;
  border-radius:8px;
  transition: all .12s ease;
  background: rgba(255,255,255,0.01);
  border: 1px solid rgba(255,255,255,0.02);
}}
.footer-center a:hover {{
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.28);
}}

.footer-right .meta-chip {{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  font-size:12px;
  color: rgba(255,255,255,0.85);
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.02);
  margin-left:8px;
}}

.small-muted {{ font-size:12px; color: rgba(255,255,255,0.55); margin-right:8px; }}
</style>

<div class="premium-footer">
  <div class="footer-left">
    <div class="logo" aria-hidden="true">
      <!-- small inline svg / symbol -->
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="2" width="20" height="20" rx="4" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.2"/>
        <path d="M6 12 L10 8 L14 12 L18 8" stroke="{accent_color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        <circle cx="8" cy="16" r="1.2" fill="{accent_color}" />
        <circle cx="16" cy="16" r="1.2" fill="{accent_color}" />
      </svg>
    </div>
    <div>
      <div class="footer-title">Stock News & Sentiment</div>
      <div class="footer-sub">Real-time headlines • Lightweight sentiment</div>
    </div>
  </div>

  <div class="footer-center">
    <a href="#" target="_blank">How it works</a>
    <a href="#" target="_blank">Data sources</a>
    <a href="#" target="_blank">Report issue</a>
  </div>

  <div class="footer-right">
    <div class="small-muted">Data:</div>
    <div class="meta-chip">Google News</div>
    <div class="small-muted">Mode:</div>
    <div class="meta-chip">{ 'Dark' if dark_mode else 'Light' }</div>
    <div class="small-muted">Last:</div>
    <div class="meta-chip">{ meta.get("ts", "") }</div>
    <div class="small-muted">v</div>
    <div class="meta-chip">1.8</div>
  </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
