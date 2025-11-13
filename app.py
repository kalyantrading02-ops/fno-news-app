import streamlit as st
import pandas as pd
import re
from gnews import GNews
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px

# ---------------- Page Setup ----------------
st.set_page_config(page_title="F&O Stocks News Dashboard", layout="wide")
st.title("📈 F&O Stocks News Dashboard")

# ---------------- F&O Stock List ----------------
@st.cache_data
def get_fo_stocks():
    return [
        "Reliance Industries", "TCS", "Infosys", "HDFC Bank", "ICICI Bank",
        "Axis Bank", "State Bank of India", "Hindustan Unilever", "ITC",
        "Bharti Airtel", "Kotak Mahindra Bank", "Larsen & Toubro", "Bajaj Finance",
        "Maruti Suzuki", "Wipro", "NTPC", "Power Grid Corporation", "UltraTech Cement",
        "Adani Enterprises", "Adani Ports", "HCL Technologies", "Sun Pharma",
        "Titan Company", "Nestle India", "Tech Mahindra", "Tata Steel",
        "JSW Steel", "Mahindra & Mahindra", "ONGC", "Grasim Industries",
        "SBI Life Insurance", "Bajaj Finserv", "HDFC Life Insurance",
        "Divi's Laboratories", "Eicher Motors", "Coal India", "Cipla",
        "Hero MotoCorp", "Britannia Industries", "Dr Reddy's Laboratories",
        "Hindalco Industries", "Tata Motors", "IndusInd Bank", "BPCL", "Asian Paints",
        "UPL", "Apollo Hospitals", "ICICI Lombard", "Shree Cement",
        "DLF", "Zee Entertainment", "GAIL", "Ambuja Cements", "Vedanta",
        "SRF", "Siemens", "Torrent Power", "Havells India", "Tata Power",
        "Pidilite Industries", "Bank of Baroda", "Muthoot Finance", "Colgate Palmolive",
        "Chola Finance", "Bosch", "Page Industries", "ACC", "TVS Motor",
        "Bharat Forge", "Canara Bank", "Indigo", "Bharat Electronics", "L&T Finance"
    ]

fo_stocks = get_fo_stocks()

# ---------------- Fetch News ----------------
@st.cache_data(ttl=1800)  # 30 minutes cache
def fetch_stock_news(stock_name):
    """Fetch latest news for a given stock with safe handling."""
    try:
        google_news = GNews(language='en', country='IN', max_results=10)
        news_list = google_news.get_news(stock_name)

        if not news_list:
            return pd.DataFrame(columns=["title", "published date", "link", "Sentiment"])

        df = pd.DataFrame(news_list)
        for col in ["title", "published date", "link"]:
            if col not in df.columns:
                df[col] = None

        df["Sentiment"] = df["title"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        return df[["title", "published date", "link", "Sentiment"]]

    except Exception as e:
        st.warning(f"⚠️ Error fetching news for {stock_name}: {e}")
        return pd.DataFrame(columns=["title", "published date", "link", "Sentiment"])

# ---------------- Parallel Fetch ----------------
@st.cache_data(ttl=1800)
def fetch_all_news(stocks):
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for stock, df in zip(stocks, executor.map(fetch_stock_news, stocks)):
            results[stock] = df
    return results

# ---------------- Sidebar ----------------
st.sidebar.header("🔍 Filters")
selected_stocks = st.sidebar.multiselect("Select Stocks", fo_stocks, default=["Reliance Industries", "TCS", "Infosys"])
time_period = st.sidebar.selectbox("Select Time Period", ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months"])

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📰 Stock News", "📊 Trending Stocks", "😊 Sentiment", "📅 Upcoming Events"])

# ---------------- Tab 1: Stock News ----------------
with tab1:
    st.subheader(f"🗞️ Latest News for Selected Stocks ({time_period})")
    all_news = fetch_all_news(selected_stocks)

    for stock in selected_stocks:
        st.markdown(f"### 🏢 {stock}")
        df = all_news.get(stock)

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                sentiment = "😊 Positive" if row["Sentiment"] > 0 else "😐 Neutral" if row["Sentiment"] == 0 else "😞 Negative"
                link_text = f"[Read more]({row['link']})" if row['link'] else ""
                st.markdown(f"- **{row['title']}** ({sentiment})  \n🗓️ {row['published date']}  \n{link_text}")
        else:
            st.info("No news found for this stock.")

# ---------------- Tab 2: Trending Stocks ----------------
with tab2:
    st.subheader("🔥 Most Mentioned Stocks (Trending)")

    all_news = fetch_all_news(fo_stocks)
    mentions = {}

    for stock, df in all_news.items():
        if df is not None and not df.empty:
            count = sum(
                re.search(rf"\b{re.escape(stock.split()[0])}\b", str(title), re.IGNORECASE)
                for title in df["title"]
            )
            mentions[stock] = count

    if mentions:
        df_mentions = pd.DataFrame(mentions.items(), columns=["Stock", "Mentions"]).sort_values(by="Mentions", ascending=False).head(20)
        fig = px.bar(df_mentions, x="Stock", y="Mentions", text="Mentions", title="Top 20 Trending Stocks")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trending data available.")

# ---------------- Tab 3: Sentiment ----------------
with tab3:
    st.subheader("😊 Sentiment Overview")

    sentiment_summary = []
    all_news = fetch_all_news(selected_stocks)

    for stock, df in all_news.items():
        if df is not None and not df.empty:
            avg_sentiment = round(df["Sentiment"].mean(), 3)
            sentiment_summary.append({"Stock": stock, "Sentiment": avg_sentiment})

    if sentiment_summary:
        df_sent = pd.DataFrame(sentiment_summary)
        fig = px.bar(df_sent, x="Stock", y="Sentiment", color="Sentiment", title="Average Sentiment per Stock")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available.")

# ---------------- Tab 4: Upcoming Events ----------------
with tab4:
    st.subheader("📅 Upcoming Market Events")
    st.markdown("""
    **Upcoming Key Market Events (India):**
    - 📊 Quarterly Results: November–December
    - 🏦 RBI Policy Meeting: Early December
    - 💰 F&O Expiry: Last Thursday of every month
    - 📈 GDP Data Release: End of November
    - 🗳️ State Election Results: December
    """)
