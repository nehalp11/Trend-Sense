# TrendSense — DBSCAN Trend Clustering
# Groups similar fashion micro-trends together

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_trends(keywords: list, scores: list, eps: float = 0.5, min_samples: int = 2) -> pd.DataFrame:
    """
    Cluster fashion trend keywords using DBSCAN.

    DBSCAN chosen over K-Means because:
    - Doesn't require specifying number of clusters
    - Handles noise/outlier trends naturally
    - Can find clusters of arbitrary shape

    Args:
        keywords: list of trend keywords
        scores: list of trend scores
        eps: max distance between points in same cluster
        min_samples: minimum points to form a cluster

    Returns:
        DataFrame with keywords, scores, and cluster labels
    """
    # Vectorize keywords using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    keyword_vectors = vectorizer.fit_transform(keywords).toarray()

    # Add trend scores as a feature
    score_feature = np.array(scores).reshape(-1, 1)
    features = np.hstack([keyword_vectors, score_feature])

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(features_scaled)

    # Build result DataFrame
    df = pd.DataFrame({
        "keyword": keywords,
        "trend_score": scores,
        "cluster": labels
    })

    # Label clusters
    df["cluster_label"] = df["cluster"].apply(
        lambda x: f"Cluster {x}" if x >= 0 else "Outlier (Unique Trend)"
    )

    return df.sort_values(["cluster", "trend_score"], ascending=[True, False])


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each cluster — top keyword and average score.
    """
    summary = df[df["cluster"] >= 0].groupby("cluster").agg(
        top_keyword=("keyword", "first"),
        avg_score=("trend_score", "mean"),
        trend_count=("keyword", "count")
    ).reset_index()

    summary["avg_score"] = summary["avg_score"].round(3)
    return summary.sort_values("avg_score", ascending=False)


if __name__ == "__main__":
    # Demo clustering
    sample_keywords = [
        "mirror work dupatta", "mirror embroidery dupatta", "mirror work kurti",
        "phulkari jacket", "phulkari kurta", "phulkari dupatta",
        "boho kurta", "boho ethnic wear",
        "unique trend xyz"  # This will be an outlier
    ]
    sample_scores = [0.85, 0.82, 0.78, 0.90, 0.88, 0.75, 0.60, 0.55, 0.10]

    df = cluster_trends(sample_keywords, sample_scores)
    print("=== TrendSense Cluster Results ===\n")
    print(df[["keyword", "trend_score", "cluster_label"]].to_string(index=False))

    print("\n=== Cluster Summary ===\n")
    print(get_cluster_summary(df).to_string(index=False))
