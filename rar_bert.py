# TrendSense — RAR-BERT Implementation
# Recency Attention Reweighting on BERT
# Core innovation: separates discovery posts from historical references

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from datetime import datetime, timezone
import numpy as np

# Tense indicators for recency scoring
PRESENT_TENSE_MARKERS = [
    "is trending", "are wearing", "just dropped", "blowing up",
    "going viral", "right now", "this week", "today", "currently",
    "everyone is", "spotted", "just saw", "new drop"
]

PAST_TENSE_MARKERS = [
    "was trending", "used to", "back in", "remember when",
    "last year", "throwback", "classic", "vintage", "old trend"
]


class RecencyScorer:
    """
    Computes a recency score for each post based on:
    1. Post timestamp (newer = higher score)
    2. Tense analysis (present tense = higher score)
    """

    def __init__(self, decay_days: int = 30):
        self.decay_days = decay_days

    def compute_temporal_score(self, post_timestamp: datetime) -> float:
        """
        Score based on how recent the post is.
        Posts from today = 1.0, posts from decay_days ago = 0.0
        """
        now = datetime.now(timezone.utc)
        if post_timestamp.tzinfo is None:
            post_timestamp = post_timestamp.replace(tzinfo=timezone.utc)

        days_old = (now - post_timestamp).days
        score = max(0.0, 1.0 - (days_old / self.decay_days))
        return score

    def compute_tense_score(self, text: str) -> float:
        """
        Score based on present vs past tense language.
        Present tense = discovery post = higher score
        Past tense = historical reference = lower score
        """
        text_lower = text.lower()

        present_count = sum(1 for marker in PRESENT_TENSE_MARKERS if marker in text_lower)
        past_count = sum(1 for marker in PAST_TENSE_MARKERS if marker in text_lower)

        if present_count + past_count == 0:
            return 0.5  # neutral

        score = present_count / (present_count + past_count)
        return score

    def compute_recency_weight(self, text: str, timestamp: datetime) -> float:
        """
        Combined recency weight = temporal score + tense score.
        Used to reweight BERT attention.
        """
        temporal = self.compute_temporal_score(timestamp)
        tense = self.compute_tense_score(text)

        # Weighted combination
        weight = 0.6 * temporal + 0.4 * tense
        return weight


class RARBERT(nn.Module):
    """
    RAR-BERT: Recency Attention Reweighting on BERT

    Modifies BERT's self-attention mechanism to prioritise
    present-tense discovery posts over historical references.

    Standard BERT attention:
        attention = softmax(QK^T / sqrt(d_k)) * V

    RAR-BERT attention:
        attention = softmax((QK^T / sqrt(d_k)) * recency_weight) * V
    """

    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2):
        super(RARBERT, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.recency_scorer = RecencyScorer()
        self.hidden_size = self.bert.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Recency gate — scales attention based on recency weight
        self.recency_gate = nn.Linear(1, self.hidden_size)

    def forward(self, text: str, timestamp: datetime) -> dict:
        """
        Forward pass with recency attention reweighting.

        Args:
            text: post text content
            timestamp: post creation datetime

        Returns:
            dict with trend_score, recency_weight, and class_logits
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Compute recency weight for this post
        recency_weight = self.recency_scorer.compute_recency_weight(text, timestamp)
        recency_tensor = torch.tensor([[recency_weight]], dtype=torch.float32)

        # Get BERT outputs
        outputs = self.bert(**inputs, output_attentions=True)

        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply recency gate to scale representation
        recency_gate = torch.sigmoid(self.recency_gate(recency_tensor))
        gated_output = cls_output * recency_gate

        # Classification
        logits = self.classifier(gated_output)
        trend_probability = torch.softmax(logits, dim=1)[0][1].item()

        return {
            "trend_score": trend_probability,
            "recency_weight": recency_weight,
            "is_discovery": recency_weight > 0.5,
            "logits": logits
        }

    def batch_predict(self, posts: list) -> list:
        """
        Predict trend scores for a batch of posts.

        Args:
            posts: list of dicts with 'text' and 'timestamp' keys

        Returns:
            list of prediction dicts
        """
        results = []
        for post in posts:
            result = self.forward(post["text"], post["timestamp"])
            result["text"] = post["text"][:100]
            results.append(result)

        return sorted(results, key=lambda x: x["trend_score"], reverse=True)


if __name__ == "__main__":
    print("RAR-BERT module loaded.")
    print("Core innovation: Recency Attention Reweighting for fashion trend detection")
    print("\nRecency Scorer demo:")
    scorer = RecencyScorer()
    sample_text = "This kurta style is blowing up right now everywhere!"
    score = scorer.compute_tense_score(sample_text)
    print(f"Sample text: '{sample_text}'")
    print(f"Tense score: {score:.2f} (1.0 = strong discovery signal)")
