import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote
import time

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="F&O Stocks News Tracker", layout="wide", page_icon="📊")

# ----------------- HEADER -----------------
st.title("📊 F&O Stocks Google News Tracker")

# ----------------- STOCK LIST -----------------
fo_stocks = [
    "Reliance Industries", "HDFC Bank", "ICICI Bank", "Infosys", "TCS", "State Bank of India",
    "Hindustan Unilever", "Bharti Airtel", "ITC", "Larsen & Toubro", "Axis Bank", "Kotak Mahindra Bank",
    "Wipro", "HCL Technologies", "Adani Enterprises", "Adani Ports", "Tata Motors", "Bajaj Finance",
    "Bajaj Finserv", "Maruti Suzuki", "NTPC", "Power Grid Corporation", "Sun Pharma",
    "Dr Reddy's Laboratories", "Divi's Laboratories", "Nestle India", "UltraTech Cement",
    "Grasim Industries", "JSW Steel", "Tata Steel"
]

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Settings")

period = st.sidebar.selectbox(
    "🗓️ Choose News Period",
    ("Last 1 Week", "Last 1 Month", "Last 3 Months", "Last 6 Months")
)

days_map = {"Last 1 Week": 7, "Last 1 Month": 30, "Last 3 Months": 90, "Last 6 Months": 180}
days = days_map[period]
since_date = datetime.now() - timedelta(days=days)

st.sidebar.write(f"📅 Showing news since: {since_date.strftime('%d %b %Y')}")

# Search bar
search_query = st.sidebar.text_input("🔍 Search specific stock (optional)", "").strip()

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #111; color: #fff; }
        .stDataFrame, .stTextInput, .stSelectbox { color: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------- FETCH NEWS FUNCTION -----------------
def fetch_news(query):
    safe_query = quote(f"{query} stock india")
    url = f"https://news.google.com/rss/search?q={safe_query}"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published >= since_date:
            source = entry.get("source", {}).get("title", "Unknown Source")
            articles.append({
                "Stock": query,
                "Title": entry.title,
                "Source": source,
                "Link": entry.link,
                "Published": published.strftime("%Y-%m-%d %H:%M")
            })
    return articles

# ----------------- AUTO REFRESH -----------------
refresh_rate = st.sidebar.slider("⏱️ Auto-refresh every (minutes)", 0, 60, 0)
if refresh_rate > 0:
    st.sidebar.info(f"App will auto-refresh every {refresh_rate} minute(s).")
    st_autorefresh = st.experimental_rerun  # auto rerun alias
    time.sleep(refresh_rate * 60)
    st.experimental_rerun()

# ----------------- FETCH BUTTON -----------------
if st.button("🔍 Fetch Latest News"):
    st.info("Fetching news from Google... please wait ⏳")
    all_news = []
    progress = st.progress(0)
    for i, stock in enumerate(fo_stocks):
        if search_query and search_query.lower() not in stock.lower():
            progress.progress((i + 1) / len(fo_stocks))
            continue
        news = fetch_news(stock)
        all_news.extend(news)
        progress.progress((i + 1) / len(fo_stocks))

    if all_news:
        df = pd.DataFrame(all_news)
        df = df.sort_values(by="Published", ascending=False)

        # Filter by stock name inside the app
        filter_stock = st.selectbox("📂 Filter by Stock", ["All"] + sorted(df["Stock"].unique().tolist()))
        if filter_stock != "All":
            df = df[df["Stock"] == filter_stock]

        # Display logo with name (simple emoji-based)
        df["Logo"] = df["Stock"].apply(lambda x: "📈" if "Bank" in x else "🏢")
        df = df[["Logo", "Stock", "Title", "Source", "Published", "Link"]]

        st.success(f"✅ Found {len(df)} news articles!")

        # Display styled table
        st.dataframe(df, use_container_width=True)

        # Download as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download News as CSV",
            data=csv,
            file_name="fo_stock_news.csv",
            mime="text/csv"
        )

    else:
        st.warning("No news found for selected period or stock.")
else:
    st.info("Click '🔍 Fetch Latest News' to begin.")
