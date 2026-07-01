# TrendSense — Weighted Trend Score Calculator
# Combines signals from Google Trends, Reddit, and Pinterest

import pandas as pd
import numpy as np

# Default weights for trend score formula
# Tuned for Indian fashion context
DEFAULT_WEIGHTS = {
    "google_trends": 0.40,   # Search momentum — strong intent signal
    "reddit_buzz":   0.35,   # Community discovery — early adopter signal
    "pinterest":     0.25    # Visual spread — mainstream adoption signal
}


def normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_trend_score(
    keyword: str,
    google_score: float,
    reddit_score: float,
    pinterest_score: float,
    weights: dict = DEFAULT_WEIGHTS
) -> dict:
    """
    Compute the final weighted trend score for a fashion keyword.

    Formula:
        TrendScore = (w1 × GoogleTrends_score)
                   + (w2 × Reddit_buzz_score)
                   + (w3 × Pinterest_visual_score)

    Args:
        keyword: fashion trend keyword
        google_score: normalized Google Trends score (0-1)
        reddit_score: normalized Reddit buzz score (0-1)
        pinterest_score: normalized Pinterest spread score (0-1)
        weights: dict of weights for each source

    Returns:
        dict with keyword, individual scores, final score, and tier
    """
    final_score = (
        weights["google_trends"] * google_score +
        weights["reddit_buzz"] * reddit_score +
        weights["pinterest"] * pinterest_score
    )

    # Classify trend tier
    if final_score >= 0.75:
        tier = "🔥 Viral"
        tier_label = "viral"
    elif final_score >= 0.50:
        tier = "📈 Emerging"
        tier_label = "emerging"
    elif final_score >= 0.25:
        tier = "🌱 Early Signal"
        tier_label = "early"
    else:
        tier = "👀 Watching"
        tier_label = "watching"

    return {
        "keyword": keyword,
        "google_score": round(google_score, 3),
        "reddit_score": round(reddit_score, 3),
        "pinterest_score": round(pinterest_score, 3),
        "final_score": round(final_score, 3),
        "tier": tier,
        "tier_label": tier_label,
        "predicted_mainstream_weeks": estimate_mainstream_weeks(final_score)
    }


def estimate_mainstream_weeks(score: float) -> int:
    """
    Estimate weeks until the trend goes mainstream
    based on current trend score.
    """
    if score >= 0.75:
        return 2
    elif score >= 0.50:
        return 4
    elif score >= 0.25:
        return 8
    else:
        return 12


def rank_trends(trends: list) -> pd.DataFrame:
    """
    Rank a list of trend score dicts by final score.

    Returns:
        Sorted DataFrame of trends
    """
    df = pd.DataFrame(trends)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


if __name__ == "__main__":
    # Demo with sample data
    sample_trends = [
        compute_trend_score("mirror work dupatta", 0.85, 0.72, 0.60),
        compute_trend_score("boho kurta", 0.45, 0.68, 0.55),
        compute_trend_score("block print co-ord", 0.30, 0.40, 0.35),
        compute_trend_score("phulkari jacket", 0.92, 0.88, 0.75),
        compute_trend_score("kalamkari fusion", 0.20, 0.25, 0.18),
    ]

    ranked = rank_trends(sample_trends)
    print("=== TrendSense — Top Emerging Indian Fashion Trends ===\n")
    for _, row in ranked.iterrows():
        print(f"{row.name}. {row['keyword']}")
        print(f"   Score: {row['final_score']} | Tier: {row['tier']}")
        print(f"   Goes mainstream in ~{row['predicted_mainstream_weeks']} weeks\n")
