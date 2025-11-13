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
import requests
from typing import List, Dict, Any, Optional

import streamlit as st
import time

# ---- Silent Timed Refresh System (10 minutes) ----
# Initialize session state
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()

# Calculate time elapsed
current_time = time.time()
elapsed = current_time - st.session_state.last_refresh_time

# If 10 minutes passed → clear cache and refresh
if elapsed > 600:  # 600 seconds = 10 minutes
    st.cache_data.clear()
    st.session_state.last_refresh_time = current_time
    st.toast("Data refreshed automatically (10 minutes elapsed).")

# Display last updated info in sidebar
mins_ago = int(elapsed // 60)
st.sidebar.info(f"⏱️ Last updated: {mins_ago} minute(s) ago")

# ---- Your main app code below ----
st.title("📊 Stock Market News & Sentiment Dashboard (Optimized)")

# Example placeholder (keep your existing logic below)
st.write("✅ Cached data loads instantly, auto-refreshes every 10 minutes silently.")

# -----------------------------
# INITIAL SETUP
# -----------------------------
# Download VADER lexicon once (quiet)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except Exception:
    nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Stock News & Sentiment Dashboard", layout="wide")
analyzer = SentimentIntensityAnalyzer()

# -----------------------------
# Utility: ISO date
# -----------------------------

def _iso_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

# -----------------------------
# Finnhub fetcher (kept, but not called by default)
# -----------------------------

def fetch_finnhub_economic_calendar(api_key: str, start: datetime, end: datetime, country: Optional[str] = None) -> List[Dict[str, Any]]:
    events = []
    if not api_key:
        return events
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {"from": _iso_date(start), "to": _iso_date(end), "token": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception as e:
        return [{"error": str(e)}]

    raw_events = []
    if isinstance(data, dict):
        if "economic" in data and isinstance(data["economic"], list):
            raw_events = data["economic"]
        else:
            for v in data.values():
                if isinstance(v, list):
                    raw_events = v
                    break
    elif isinstance(data, list):
        raw_events = data

    for ev in raw_events:
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
        events.append({"date": event_date, "title": title, "country": country_ev, "impact": impact, "raw": ev})
    if country:
        country_up = country.strip().upper()
        events = [e for e in events if not e.get("country") or e.get("country").startswith(country_up)]
    events.sort(key=lambda x: (x["date"] or datetime.max))
    return events

# -----------------------------
# SIDEBAR — SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")
try:
    dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True, help="Switch instantly between Dark & Light Mode")
except Exception:
    dark_mode = st.sidebar.checkbox("🌗 Dark Mode", value=True, help="Switch instantly between Dark & Light Mode")

st.sidebar.markdown("---")
# Allow user to paste or upload a 200-stock list
st.sidebar.subheader("Stocks list")
use_full_list = st.sidebar.checkbox("Use full F&O list (paste/upload below)", value=False)
stock_text = st.sidebar.text_area("Paste newline-separated stock/company names (when using full list)", height=120)
uploaded = st.sidebar.file_uploader("Or upload CSV with one column of stock names", type=["csv", "txt"])

# Caching preference
st.sidebar.subheader("Performance")
use_cache = st.sidebar.checkbox("Enable caching for news (faster reloads)", value=True)

# Cleaner spacing / faster tabs preference
st.sidebar.subheader("UI")
ui_compact = st.sidebar.checkbox("Use slightly cleaner spacing & faster tabs", value=True)

# -----------------------------
# APPLY THEMES (CSS)
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

spacing_css = "padding:6px 0;" if ui_compact else "padding:10px 0;"

st.markdown(
    f"""
<style>
body {{ background: {bg_gradient}; color: {text_color}; }}
.stApp {{ background: {bg_gradient} !important; color: {text_color} !important; }}
h1, h2, h3, h4, h5 {{ color: {accent_color} !important; }}
.stButton button {{ background-color: {accent_color} !important; color: black !important; border-radius: 6px; }}
.stDataFrame {{ border-radius: 10px; background-color: rgba(255,255,255,0.02); }}
.news-card {{ border-radius: 10px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); transition: transform .12s ease, box-shadow .12s ease; background: rgba(255,255,255,0.02); margin-bottom: 12px; }}
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
.section-compact {{ {spacing_css} }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# APP TITLE
# -----------------------------
st.title("💹 Stock Market News & Sentiment Dashboard")

# -----------------------------
# AUTO REFRESH (kept but safer)
# -----------------------------
refresh_interval = 600  # 10 minutes
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
else:
    if time.time() - st.session_state["last_refresh"] > refresh_interval:
        st.session_state["last_refresh"] = time.time()
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                st.warning("Auto-refresh is unavailable in this Streamlit version. Please refresh manually.")
                st.stop()

# -----------------------------
# Sidebar filters
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
# F&O STOCK LIST (default small sample)
# User can paste/upload full 200 list via sidebar above
# -----------------------------
DEFAULT_FO_STOCKS = [
    "Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank", "State Bank of India",
    "HCL Technologies", "Wipro", "Larsen & Toubro", "Tata Motors", "Bajaj Finance", "Axis Bank",
    "NTPC", "ITC", "Adani Enterprises", "Coal India", "Power Grid", "Maruti Suzuki", "Tech Mahindra", "Sun Pharma",
]

# Build final stock list
fo_stocks = DEFAULT_FO_STOCKS.copy()
# Add from uploaded file
if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded, header=None)
        vals = df_up.iloc[:, 0].astype(str).str.strip().tolist()
        fo_stocks = [v for v in vals if v]
    except Exception:
        try:
            txt = uploaded.getvalue().decode("utf-8")
            vals = [l.strip() for l in txt.splitlines() if l.strip()]
            fo_stocks = vals
        except Exception:
            st.sidebar.error("Uploaded file could not be parsed. Please paste the list instead.")

# Add from pasted text if requested
if use_full_list and stock_text.strip():
    pasted = [l.strip() for l in stock_text.splitlines() if l.strip()]
    if pasted:
        fo_stocks = pasted

# If user enabled full list but provided none, show guidance
if use_full_list and not fo_stocks:
    st.sidebar.warning("Full list selected but no stocks provided — please paste names or upload CSV.")

# -----------------------------
# FETCHERS & CACHING
# - Use st.cache_data for news fetcher; controlled by sidebar toggle
# -----------------------------

# Provide wrapper to optionally use cache
if use_cache:
    cache_decorator = st.cache_data
else:
    # no-op decorator fallback
    def cache_decorator(ttl=None, show_spinner=True):
        def inner(f):
            return f
        return inner

@cache_decorator(ttl=300, show_spinner=False)
def fetch_news(stock, start, end, max_results=30):
    """
    Fetch news for a stock with deduplication and robust error handling.
    Cached by (stock, start, end, max_results) when caching is enabled.
    """
    try:
        gnews = GNews(language="en", country="IN", max_results=max_results)
        try:
            gnews.start_date, gnews.end_date = start, end
        except Exception:
            pass

        raw = gnews.get_news(stock) or []
        if not raw:
            return []

        seen = set()
        unique_articles = []
        for art in raw:
            title = (art.get("title") or "").strip()
            norm = re.sub(r'\W+', " ", title.lower()).strip()
            key = norm[:200] if norm else json.dumps(art, sort_keys=True)[:120]
            if key in seen:
                continue
            seen.add(key)
            unique_articles.append(art)

        return unique_articles
    except Exception:
        return []


@cache_decorator(ttl=600, show_spinner=False)
def fetch_all_news(stocks: List[str], start: datetime, end: datetime, max_results_per_stock: int = 30):
    """
    Fetch all news concurrently for a list of stocks. Returns list of dicts {Stock, Articles, News Count}.
    Uses a controlled ThreadPoolExecutor to avoid extreme resource use when fetching 200 stocks.
    """
    results = []
    # cap workers sensibly depending on number of stocks
    workers = min(15, max(4, len(stocks) // 5))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_news, s, start, end, max_results_per_stock): s for s in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                articles = future.result() or []
                # Only keep articles with publisher/title to avoid waste
                filtered = []
                for art in articles:
                    pub = art.get("publisher")
                    pub_title = ""
                    if isinstance(pub, dict):
                        pub_title = (pub.get("title") or "").strip()
                    elif isinstance(pub, str):
                        pub_title = pub.strip()
                    else:
                        pub_title = (art.get("source") or "").strip()
                    if pub_title:
                        if not isinstance(pub, dict):
                            art["publisher"] = {"title": pub_title}
                        else:
                            art["publisher"]["title"] = pub_title
                        filtered.append(art)
                results.append({"Stock": stock, "Articles": filtered, "News Count": len(filtered)})
            except Exception:
                results.append({"Stock": stock, "Articles": [], "News Count": 0})
    return results

# -----------------------------
# Sentiment helper
# -----------------------------

def analyze_sentiment(text):
    if not text:
        text = ""
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive", "🟢", score
    elif score < -0.2:
        return "Negative", "🔴", score
    else:
        return "Neutral", "🟡", score

# -----------------------------
# Scoring engine (kept unchanged)
# -----------------------------
WEIGHTS = {
    "earnings_guidance": 30,
    "M&A_JV": 25,
    "management_change": 20,
    "buyback_dividend": 20,
    "contract_deal": 25,
    "block_insider": 25,
    "policy_regulation": 20,
    "analyst_move": 15,
    "numeric_mentioned": 10,
    "trusted_source": 15,
    "speculative_penalty": -15,
    "low_quality_penalty": -10,
    "max_corroboration_bonus": 20,
}

HIGH_PRIORITY_KEYWORDS = {
    "earnings": ["earnings", "quarter", "q1", "q2", "q3", "q4", "revenue", "profit", "loss", "guidance", "outlook", "beat", "miss", "results"],
    "MA": ["acquires", "acquisition", "merger", "demerger", "spin-off", "spin off", "joint venture", "jv"],
    "management": ["appoint", "resign", "ceo", "cfo", "chairman", "board", "director", "promoter", "coo", "md"],
    "corp_action": ["buyback", "dividend", "split", "bonus issue", "bonus", "rights issue", "rights", "share pledge", "pledge"],
    "contract": ["contract", "order", "tender", "deal", "agreement", "licence", "license", "wins order"],
    "regulatory": ["sebi", "investigation", "fraud", "lawsuit", "penalty", "fine", "regulation", "ban", "policy", "pli", "subsidy", "tariff"],
    "analyst": ["upgrade", "downgrade", "target", "recommendation", "brokerage", "analyst"],
    "block": ["block deal", "bulk deal", "blocktrade", "block-trade", "insider", "promoter buy", "promoter selling", "promoter sell"],
}

TRUSTED_SOURCES = {"reuters", "bloomberg", "economic times", "economictimes", "livemint", "mint", "business standard", "business-standard", "cnbc", "ft", "financial times", "press release", "nse", "bse"}
LOW_QUALITY_SOURCES = {"blog", "medium", "wordpress", "forum", "reddit", "quora"}
SPECULATIVE_WORDS = ["may", "might", "could", "rumour", "rumor", "reportedly", "alleged", "possible", "speculat"]
NUMERIC_PATTERN = r'[%₹$£€]|(?:\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b\s*(?:crore|lakh|billion|bn|mn|m|₹|rs\.|rs|rupee|ton|tons|mw|MW|GW))'
numeric_re = re.compile(NUMERIC_PATTERN, re.IGNORECASE)


def norm_text(s):
    return (s or "").strip().lower()


def contains_any(text, keywords):
    t = norm_text(text)
    return any(k in t for k in keywords)


def is_trusted(publisher):
    if not publisher:
        return False
    p = norm_text(publisher)
    return any(ts in p for ts in TRUSTED_SOURCES)


def is_low_quality(publisher):
    if not publisher:
        return False
    p = norm_text(publisher)
    return any(lq in p for lq in LOW_QUALITY_SOURCES)


def has_numeric(text):
    return bool(numeric_re.search(text or ""))


def score_article(title, desc, publisher, corroboration_sources=None):
    raw = 0
    reasons = []
    txt = f"{title} {desc}".lower()

    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["earnings"]):
        raw += WEIGHTS["earnings_guidance"]
        reasons.append("Earnings/Guidance")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["MA"]):
        raw += WEIGHTS["M&A_JV"]
        reasons.append("M&A/JV")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["management"]):
        raw += WEIGHTS["management_change"]
        reasons.append("Management/Govt")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["corp_action"]):
        raw += WEIGHTS["buyback_dividend"]
        reasons.append("Corporate Action")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["contract"]):
        raw += WEIGHTS["contract_deal"]
        reasons.append("Contract/Order")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["regulatory"]):
        raw += WEIGHTS["policy_regulation"]
        reasons.append("Regulatory/Policy")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["analyst"]):
        raw += WEIGHTS["analyst_move"]
        reasons.append("Broker/Analyst Move")
    if contains_any(txt, HIGH_PRIORITY_KEYWORDS["block"]):
        raw += WEIGHTS["block_insider"]
        reasons.append("Block/Insider Deal")

    if has_numeric(txt):
        raw += WEIGHTS["numeric_mentioned"]
        reasons.append("Numeric Mention")

    if is_trusted(publisher):
        raw += WEIGHTS["trusted_source"]
        reasons.append("Trusted Source")

    if is_low_quality(publisher):
        raw += WEIGHTS["low_quality_penalty"]
        reasons.append("Low-quality Source (penalized)")

    if contains_any(txt, SPECULATIVE_WORDS):
        raw += WEIGHTS["speculative_penalty"]
        reasons.append("Speculative Language (penalized)")

    corroboration_bonus = 0
    if corroboration_sources:
        trusted_count = sum(1 for s in set(corroboration_sources) if s and is_trusted(s))
        if trusted_count > 1:
            corroboration_bonus = min(WEIGHTS["max_corroboration_bonus"], 5 * (trusted_count - 1))
            if corroboration_bonus:
                reasons.append("Corroboration")

    score = int(max(0, min(100, raw + corroboration_bonus)))
    return score, reasons

# -----------------------------
# Session defaults (kept)
# -----------------------------
st.session_state.setdefault("saved_articles", [])
st.session_state.setdefault("manual_events", [])

# -----------------------------
# Utility: extract headline map from results for corroboration
# -----------------------------

def build_headline_map(news_results: List[Dict[str, Any]]):
    headline_map = {}
    for res in news_results:
        stock = res.get("Stock", "Unknown")
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
            headline_map.setdefault(key, []).append(pub_name or "unknown")
    return headline_map

# -----------------------------
# EVENT extraction utilities (kept)
# -----------------------------
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


def extract_events_from_news(news_results):
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
                m = re.search(r'on ([A-Za-z0-9 ,\-thstndrd]{3,30})', txt)
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
                else:
                    source = pub or art.get("source") or ""
                url = art.get("url") or art.get("link") or "#"
                priority = "Normal"
                try:
                    if is_trusted(source):
                        priority = "High"
                except Exception:
                    priority = "Normal"
                events.append({"stock": stock, "type": etype_label, "desc": desc, "date": dt, "source": source, "url": url, "priority": priority})

    # dedupe events by (stock, type, date)
    unique = {}
    for e in events:
        key = (e["stock"], e["type"], e["date"].date())
        if key not in unique:
            unique[key] = e
        else:
            existing = unique[key]
            if e["source"] and e["source"] not in existing.get("source", ""):
                existing["source"] += f"; {e['source']}"
    events = sorted(unique.values(), key=lambda x: (x["date"], x["priority"] == "High"))

    # include manual events from session
    manual = st.session_state.get("manual_events", [])
    for me in manual:
        events.append({"stock": me.get("stock", "Manual"), "type": me.get("type", "manual"), "desc": me.get("desc", ""), "date": me.get("date"), "source": "Manual", "url": "#", "priority": me.get("priority", "Normal")})
    events = sorted(events, key=lambda x: (x["date"] if isinstance(x["date"], datetime) else datetime.max))
    return events

# -----------------------------
# MAIN TABS (lazy load heavy fetches inside tabs)
# -----------------------------
news_tab, trending_tab, sentiment_tab, events_tab = st.tabs(["📰 News", "🔥 Trending Stocks", "💬 Sentiment", "📅 Upcoming Events"]) 

# -----------------------------
# TAB 1 — NEWS (lazy fetch)
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

    # Fetch news (cached) only when this tab runs
    with st.spinner("Fetching latest financial news (this may take a moment for large lists)..."):
        raw_news_results = fetch_all_news(fo_stocks, start_date, today, max_results_per_stock=25)

    # Build headline map for corroboration
    news_results = raw_news_results
    headline_map = build_headline_map(news_results)

    displayed_total = 0
    filtered_out_total = 0

    for res in news_results:
        stock = res.get("Stock", "Unknown")
        articles = res.get("Articles", []) or []
        scored_list = []
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
            url = art.get("url") or art.get("link") or "#"
            norm_head = re.sub(r'\W+', " ", title.lower()).strip()
            key = norm_head[:120] if norm_head else f"{stock.lower()}_{(title or '')[:40]}"
            publishers_for_head = headline_map.get(key, [])
            score, reasons = score_article(title, desc, publisher, corroboration_sources=publishers_for_head)
            scored_list.append({"title": title, "desc": desc, "publisher": publisher or "Unknown Source", "url": url, "score": score, "reasons": reasons, "raw": art})

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
                    pub_raw = art.get("raw", {})
                    published_date = pub_raw.get("published date") if isinstance(pub_raw, dict) else "N/A"
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

                    st.markdown(f"**[{title}]({url})**  {priority_icon} *{priority_label} ({score})*  🏢 *{publisher}* | 🗓️ *{published_date or 'N/A'}*")
                    st.markdown(f"*Reasons:* `{reasons_txt}`  •  *Sentiment:* {sentiment_emoji} {sentiment_label}")
                    if show_snippet and art.get("desc"):
                        snippet = art["desc"] if len(art["desc"]) < 220 else art["desc"][:217] + "..."
                        st.markdown(f"> {snippet}")

                    safe_stock = re.sub(r'\W+', '_', stock.lower())
                    save_key = f"save_{safe_stock}_{idx}_{abs(hash(url))}"

                    if st.button("💾 Save / Watch", key=save_key):
                        found = next((x for x in st.session_state["saved_articles"] if x["url"] == url), None)
                        if not found:
                            st.session_state["saved_articles"].append({"title": title, "url": url, "stock": stock, "date": published_date, "score": score})
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
# TAB 2 — TRENDING (lazy fetch)
# -----------------------------
with trending_tab:
    st.header(f"🔥 Trending F&O Stocks by Market-Impacting News — {time_period}")
    impact_threshold = 40

    with st.spinner("Computing trending stocks..."):
        all_results = fetch_all_news(fo_stocks, start_date, today, max_results_per_stock=20)
        headline_map = build_headline_map(all_results)

        counts = []
        for res in all_results:
            stock_name = res.get("Stock", "")
            articles = res.get("Articles") or []
            impactful_count = 0
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
                key = norm_head[:120] if norm_head else f"{stock_name.lower()}_{(title or '')[:40]}"
                publishers_for_head = headline_map.get(key, [])
                score, reasons = score_article(title, desc, publisher, corroboration_sources=publishers_for_head)
                if score >= impact_threshold:
                    impactful_count += 1
            counts.append({"Stock": stock_name, "News Count": int(impactful_count)})

        df_counts = pd.DataFrame(counts).sort_values("News Count", ascending=False).reset_index(drop=True)

    if df_counts.empty:
        st.info("No data available — try changing the time period or paste/upload a full list of stocks.")
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

        # build colors (single-color palette for clarity and faster rendering)
        palette = [accent_color]
        colors = [palette[0] for _ in range(len(df_counts))]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_counts["Stock"],
            y=df_counts[y_field],
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.12)', width=0.6)),
            text=df_counts["Label"],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Value: ' + hover_template_extra + '<extra></extra>',
        ))

        fig.update_layout(
            template=plot_theme,
            title=dict(text=f"Trending F&O Stocks (market-impacting news only) — {time_period}", x=0.5, xanchor='center', font=dict(size=16)),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=70, l=60, r=40, b=120),
            height=480,
        )

        fig.update_xaxes(tickangle=-35, tickfont=dict(size=10), showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.08)', tickfont=dict(size=11), title_text=yaxis_title, rangemode="tozero")

        fig.update_traces(textfont=dict(size=11, color="#ffffff" if dark_mode else "#111111"), cliponaxis=False)

        if dark_mode:
            fig.update_layout(font=dict(color="#EAEAEA"))
        else:
            fig.update_layout(font=dict(color="#111111"))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Market-impacting News Summary")

    df_display = df_counts[["Stock", "News Count"]].copy()
    if "Percent" in df_counts.columns:
        df_display["Percent"] = df_counts["Percent"].round(1)

    st.dataframe(df_display, use_container_width=True)

    top_nonzero = df_counts[df_counts["News Count"] > 0].head(3)
    if not top_nonzero.empty:
        st.success(f"🚀 Top Trending (market-impacting): {', '.join(top_nonzero['Stock'].tolist())}")
        st.caption(f"Showing articles with score ≥ {impact_threshold}. Adjust `impact_threshold` in the code to tune sensitivity.")
    else:
        st.info("No market-impacting news found in the selected timeframe (all counts are 0).")

# -----------------------------
# TAB 3 — SENTIMENT (lazy fetch)
# -----------------------------
with sentiment_tab:
    st.header("💬 Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment_data = []
        all_results = fetch_all_news(fo_stocks[:50], start_date, today, max_results_per_stock=10)
        for res in all_results:
            stock = res.get("Stock", "Unknown")
            for art in res.get("Articles", [])[:3]:
                title = art.get("title") or ""
                desc = art.get("description") or art.get("snippet") or ""
                combined = f"{title}. {desc}"
                s_label, emoji, s_score = analyze_sentiment(combined)
                sentiment_data.append({"Stock": stock, "Headline": title, "Sentiment": s_label, "Emoji": emoji, "Score": s_score})
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data).sort_values(by=["Stock", "Score"], ascending=[True, False])
            st.dataframe(sentiment_df, use_container_width=True)
            csv_bytes = sentiment_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sentiment Data", csv_bytes, "sentiment_data.csv", "text/csv")
        else:
            st.warning("No sentiment data found for the selected timeframe.")

# -----------------------------
# TAB 4 — UPCOMING EVENTS
# -----------------------------
with events_tab:
    # Extract events from news lazily (faster)
    st.subheader(f"📅 Upcoming Market-Moving Events (next {EVENT_WINDOW_DAYS} days)")
    with st.spinner("Extracting events from news headlines..."):
        # reuse smaller fetch to reduce cost
        results_for_events = fetch_all_news(fo_stocks[:80], start_date, today, max_results_per_stock=15)
        events = extract_events_from_news(results_for_events)

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
        st.download_button("📥 Download Extracted Events (CSV)", df_events.to_csv(index=False).encode("utf-8"), "extracted_events.csv", "text/csv")
        for e in events[:10]:
            date_str = e["date"].strftime("%Y-%m-%d") if isinstance(e["date"], datetime) else str(e["date"])
            st.markdown(f"- **{e['stock']}** — *{e['type'].title()}* on **{date_str}** — *{e['priority']}* — [{e['source']}]({e['url']})")
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
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(f"📊 Data Source: Google News | Mode: {'Dark' if dark_mode else 'Light'} | Auto-refresh every 10 min | Built with ❤️ using Streamlit & Plotly")

# End of file
