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

# -----------------------------
# SESSION STATE SETUP
# -----------------------------
st.session_state.setdefault("saved_articles", [])
st.session_state.setdefault("manual_events", [])

# -----------------------------
# INITIAL NEWS FETCH (lightweight - first 10 stocks)
# -----------------------------
# We fetch a smaller set on initial load to keep startup fast.
initial_stocks = fo_stocks[:10]
with st.spinner("Fetching latest financial news (light)..."):
    raw_news_results = fetch_all_news(initial_stocks, start_date, today, max_results=20)

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

# -----------------------------
# TABS
# -----------------------------
news_tab, trending_tab, sentiment_tab, events_tab = st.tabs(
    ["📰 News", "🔥 Trending Stocks", "💬 Sentiment", "📅 Upcoming Events"]
)

# -----------------------------
# TAB: NEWS
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

    # Use the fetched `news_results` (first 10 stocks). Full list is fetched on-demand in Trending.
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
# TAB: TRENDING (fixed & robust)
# -----------------------------
with trending_tab:
    st.header(f"🔥 Trending F&O Stocks by Market-Impacting News — {time_period}")

    impact_threshold = st.slider("Impact threshold (score)", 0, 100, 40)
    st.caption("Lower the threshold to capture more items. Increase Max Articles per stock in sidebar to fetch more.")

    # Fetch full list (this is heavier; done on-demand in Trending)
    with st.spinner("Fetching news for trending calculation..."):
        all_results = fetch_all_news(fo_stocks, start_date, today, max_results=40)

    # Rebuild headline_map from all_results for consistency
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

    counts = []
    debug_samples = []
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
            publishers_for_head = headline_map_full.get(key, [])

            score, reasons = score_article(title, desc, publisher, corroboration_sources=publishers_for_head)
            if score >= impact_threshold:
                impactful_count += 1

            if len(debug_samples) < 6 and title:
                debug_samples.append({"stock": stock_name, "title": title, "score": score, "publisher": publisher})

        counts.append({"Stock": stock_name, "News Count": int(impactful_count)})

    df_counts = pd.DataFrame(counts).sort_values("News Count", ascending=False).reset_index(drop=True)

    if df_counts.empty:
        st.info("No data available — try increasing Max Articles or widening time period.")
    else:
        # Simple bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_counts["Stock"],
            y=df_counts["News Count"],
            text=df_counts["News Count"],
            textposition='outside'
        ))
        fig.update_layout(template=plot_theme, title_text=f"Trending F&O Stocks (market-impacting news only) — {time_period}", xaxis_tickangle=-35, height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Market-impacting News Summary")
        st.dataframe(df_counts, use_container_width=True)

        if df_counts["News Count"].sum() == 0:
            st.warning("All counts are zero. Possible causes: low max_results, strict impact threshold, or no articles returned by the fetcher.")
            st.markdown("**Debug sample headlines (score):**")
            for d in debug_samples:
                st.markdown(f"- **{d['stock']}** — {d['title'][:120]} — *score {d['score']}* — _{d['publisher']}_")

# -----------------------------
# TAB: SENTIMENT
# -----------------------------
with sentiment_tab:
    st.header("💬 Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment_data = []
        # We use first 10 stocks' results (fast)
        for res in raw_news_results:
            stock = res.get("Stock", "Unknown")
            for art in (res.get("Articles") or [])[:3]:
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
# PART C — Events tab, manual events, footer
# -----------------------------

# ---- Helpers for date parsing & event extraction ----
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
    # Try dateutil fuzzy parse
    try:
        dt = dtparse(s, fuzzy=True)
        return dt
    except Exception:
        # fallback formats
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

# ---- Extract events from news_results (company/corporate events) ----
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
            # Skip past dates and far future
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

# ---- Events tab UI ----
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
        st.download_button("📥 Download Extracted Events (CSV)", df_events.to_csv(index=False).encode("utf-8"), "extracted_events.csv", "text/csv")

        for e in events[:10]:
            date_str = e["date"].strftime("%Y-%m-%d") if isinstance(e["date"], datetime) else str(e["date"])
            st.markdown(f"- **{e['stock']}** — *{e['type'].title()}* on **{date_str}** — *{e['priority']}* — [{e['source']}]({e['url']})")
    else:
        st.info("No upcoming company updates found from recent news. Add manually if needed.")

    # ---- Manual Add Section ----
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

# ---- FOOTER ----
st.markdown("---")
st.caption(f"📊 Data Source: Google News | Mode: {'Dark' if dark_mode else 'Light'} | Auto-refresh: session-cached | Built with Streamlit & Plotly")
