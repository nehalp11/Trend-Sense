# TrendSense — Reddit Data Collector
# Collects fashion discovery posts from Indian fashion subreddits

import praw
import pandas as pd
from datetime import datetime, timezone

# Target subreddits for Indian fashion trend discovery
TARGET_SUBREDDITS = [
    "IndianFashionAddicts",
    "IndiaShopping",
    "ethnic_wear",
    "streetwear",
    "femalefashionadvice"
]

# Keywords that indicate a DISCOVERY post (present tense)
DISCOVERY_KEYWORDS = [
    "just found", "anyone else wearing", "this is trending",
    "blowing up", "everywhere now", "so popular right now",
    "new trend", "obsessed with", "everyone is wearing",
    "spotted", "just saw", "going viral"
]

# Keywords that indicate a HISTORICAL post (filter these out)
HISTORICAL_KEYWORDS = [
    "used to", "remember when", "back in", "throwback",
    "was popular", "was trending", "old trend", "nostalgia",
    "classic", "vintage", "retro"
]


def init_reddit(client_id: str, client_secret: str, user_agent: str) -> praw.Reddit:
    """Initialize Reddit API connection."""
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )


def scrape_subreddit(reddit: praw.Reddit, subreddit_name: str, limit: int = 100) -> pd.DataFrame:
    """
    Scrape posts from a subreddit and classify as discovery or historical.

    Args:
        reddit: authenticated Reddit instance
        subreddit_name: name of subreddit to scrape
        limit: number of posts to fetch

    Returns:
        DataFrame with post data and classification
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.new(limit=limit):
        text = (post.title + " " + post.selftext).lower()

        is_discovery = any(kw in text for kw in DISCOVERY_KEYWORDS)
        is_historical = any(kw in text for kw in HISTORICAL_KEYWORDS)

        posts.append({
            "id": post.id,
            "title": post.title,
            "text": post.selftext[:500],
            "score": post.score,
            "comments": post.num_comments,
            "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            "subreddit": subreddit_name,
            "is_discovery": is_discovery,
            "is_historical": is_historical,
            "url": post.url
        })

    return pd.DataFrame(posts)


def compute_buzz_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Reddit buzz score for each post.
    Discovery posts weighted higher than historical ones.
    """
    df["buzz_score"] = (
        df["score"] * 0.5 +
        df["comments"] * 0.3 +
        df["is_discovery"].astype(int) * 50 -
        df["is_historical"].astype(int) * 30
    )
    return df


if __name__ == "__main__":
    print("Reddit scraper initialized.")
    print("Note: Add your Reddit API credentials to use this module.")
    print(f"Target subreddits: {TARGET_SUBREDDITS}")
    print(f"Discovery keywords: {len(DISCOVERY_KEYWORDS)}")
    print(f"Historical keywords (to filter): {len(HISTORICAL_KEYWORDS)}")
