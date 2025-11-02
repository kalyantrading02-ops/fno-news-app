import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="F&O Stocks Google News", layout="wide")

st.title("📊 F&O Stocks News Tracker (Google News RSS)")

# --- Stock list (F&O)
fo_stocks = [
    "Reliance Industries", "HDFC Bank", "ICICI Bank", "Infosys", "TCS", "State Bank of India",
    "Hindustan Unilever", "Bharti Airtel", "ITC", "Larsen & Toubro", "Axis Bank", "Kotak Mahindra Bank",
    "Wipro", "HCL Technologies", "Adani Enterprises", "Adani Ports", "Tata Motors", "Bajaj Finance",
    "Bajaj Finserv", "Maruti Suzuki", "NTPC", "Power Grid Corporation", "Sun Pharma", "Dr Reddy's Laboratories",
    "Divi's Laboratories", "Nestle India", "UltraTech Cement", "Grasim Industries", "JSW Steel", "Tata Steel"
]

# --- Time period selection
st.sidebar.header("Select News Period")
period = st.sidebar.selectbox(
    "Choose Time Frame",
    ("Last 1 Week", "Last 1 Month", "Last 3 Months", "Last 6 Months")
)

days_map = {
    "Last 1 Week": 7,
    "Last 1 Month": 30,
    "Last 3 Months": 90,
    "Last 6 Months": 180
}
days = days_map[period]
since_date = datetime.now() - timedelta(days=days)

st.sidebar.write(f"📅 Showing news since: {since_date.strftime('%d %b %Y')}")

# --- News fetching
def fetch_news(query):
    url = f"https://news.google.com/rss/search?q={query}+stock+india"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published >= since_date:
            articles.append({
                "Stock": query,
                "Title": entry.title,
                "Link": entry.link,
                "Published": published.strftime("%Y-%m-%d %H:%M")
            })
    return articles

# --- Display
if st.button("🔍 Fetch Latest News"):
    all_news = []
    progress = st.progress(0)
    for i, stock in enumerate(fo_stocks):
        news = fetch_news(stock)
        all_news.extend(news)
        progress.progress((i + 1) / len(fo_stocks))
    if all_news:
        df = pd.DataFrame(all_news)
        df = df.sort_values(by="Published", ascending=False)
        st.success(f"Fetched {len(df)} news articles!")
        st.dataframe(df)
    else:
        st.warning("No news found for the selected period.")
else:
    st.info("Click 'Fetch Latest News' to begin.")
