# TrendSense — Google Trends Data Collector
# Collects search volume signals for Indian fashion keywords

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime

# Indian fashion keywords to track
FASHION_KEYWORDS = [
    "kurta style 2026",
    "saree draping trend",
    "indo western outfit",
    "ethnic wear men",
    "palazzo pants india",
    "mirror work dupatta",
    "phulkari trend",
    "kalamkari fashion",
    "block print kurta",
    "boho indian fashion"
]

def fetch_google_trends(keywords: list, timeframe: str = "today 3-m", geo: str = "IN") -> pd.DataFrame:
    """
    Fetch Google Trends data for given keywords in India.

    Args:
        keywords: list of fashion keywords to track
        timeframe: time range (default: last 3 months)
        geo: geography (default: India)

    Returns:
        DataFrame with trend scores for each keyword over time
    """
    pytrends = TrendReq(hl='en-IN', tz=330)

    all_data = []

    # Process in batches of 5 (Google Trends API limit)
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i+5]
        pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
        data = pytrends.interest_over_time()

        if not data.empty:
            data = data.drop(columns=["isPartial"], errors="ignore")
            all_data.append(data)

    if all_data:
        return pd.concat(all_data, axis=1)
    return pd.DataFrame()


def compute_trend_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute week-over-week momentum for each keyword.
    Higher momentum = faster growing trend.
    """
    momentum = df.pct_change(periods=7).fillna(0)
    momentum.columns = [f"{col}_momentum" for col in momentum.columns]
    return momentum


def get_trending_keywords(df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Return top N trending keywords based on recent score.
    """
    recent = df.tail(7).mean().sort_values(ascending=False)
    return recent.head(top_n).index.tolist()


if __name__ == "__main__":
    print("Fetching Google Trends data for Indian fashion keywords...")
    df = fetch_google_trends(FASHION_KEYWORDS)

    if not df.empty:
        df.to_csv("google_trends_data.csv")
        print(f"Saved {len(df)} rows of trend data.")
        print("\nTop trending keywords right now:")
        print(get_trending_keywords(df))
    else:
        print("No data fetched. Check internet connection or keyword list.")
