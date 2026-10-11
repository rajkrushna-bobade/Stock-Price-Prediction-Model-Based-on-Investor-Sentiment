"""
sentiment_analysis.py
----------------------
Computes investor sentiment scores from news/tweet headlines using:
  1. VADER  - fast, rule-based, good for short social-media style text
  2. BERT   - transformer-based (distilbert-sst2), context aware

Final sentiment score per headline is a weighted average of the two,
then aggregated to a daily sentiment score used as a model feature.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False


class SentimentAnalyzer:
    def __init__(self, vader_weight: float = 0.4, bert_weight: float = 0.6, use_bert: bool = True):
        self.vader = SentimentIntensityAnalyzer()
        self.vader_weight = vader_weight
        self.bert_weight = bert_weight
        self.use_bert = use_bert and _BERT_AVAILABLE

        if self.use_bert:
            # Lightweight sentiment model fine-tuned on SST-2
            self.bert_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        else:
            self.bert_pipeline = None

    def _vader_score(self, text: str) -> float:
        """Returns compound score in range [-1, 1]."""
        return self.vader.polarity_scores(text)["compound"]

    def _bert_score(self, text: str) -> float:
        """Returns signed score in range [-1, 1] (POSITIVE -> +prob, NEGATIVE -> -prob)."""
        if not self.use_bert:
            return 0.0
        result = self.bert_pipeline(text[:512])[0]
        score = result["score"]
        return score if result["label"] == "POSITIVE" else -score

    def score_text(self, text: str) -> float:
        """Weighted combination of VADER + BERT for a single piece of text."""
        v = self._vader_score(text)
        b = self._bert_score(text) if self.use_bert else v
        weight_sum = self.vader_weight + (self.bert_weight if self.use_bert else 0)
        combined = (v * self.vader_weight + b * (self.bert_weight if self.use_bert else 0)) / weight_sum
        return combined

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "headline", date_col: str = "date") -> pd.DataFrame:
        """
        Scores every row and aggregates to mean daily sentiment.
        Returns a DataFrame indexed by date with column 'sentiment_score'.
        """
        df = df.copy()
        df["sentiment_score"] = df[text_col].apply(self.score_text)
        daily = df.groupby(date_col)["sentiment_score"].mean().reset_index()
        daily[date_col] = pd.to_datetime(daily[date_col])
        daily = daily.set_index(date_col)
        return daily


if __name__ == "__main__":
    # quick manual test
    sa = SentimentAnalyzer(use_bert=False)  # set True if transformers/torch installed
    df = pd.read_csv("data/sample_news.csv")
    daily_sentiment = sa.score_dataframe(df)
    print(daily_sentiment)
