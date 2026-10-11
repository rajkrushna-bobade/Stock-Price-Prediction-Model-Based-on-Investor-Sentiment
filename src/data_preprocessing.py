"""
data_preprocessing.py
----------------------
- Downloads historical OHLCV stock data via yfinance
- Merges it with daily investor sentiment scores
- Scales features and builds sliding-window sequences for LSTM training
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "sentiment_score"]


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Downloads OHLCV data for a ticker between start and end dates."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def merge_with_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins daily sentiment onto price data by date index.
    Missing sentiment days are filled with neutral (0.0).
    """
    merged = price_df.join(sentiment_df, how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    return merged


def scale_features(df: pd.DataFrame, feature_cols=FEATURE_COLS):
    """Scales feature columns to [0, 1]. Returns scaled array + fitted scaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler


def create_sequences(scaled_data: np.ndarray, window: int = 60, target_col_idx: int = 3):
    """
    Builds sliding-window sequences for supervised time-series learning.
    target_col_idx = 3 -> 'Close' price (index within FEATURE_COLS).

    Returns X of shape (samples, window, n_features) and y of shape (samples,).
    """
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i - window:i])
        y.append(scaled_data[i, target_col_idx])
    return np.array(X), np.array(y)


def train_test_split_sequences(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    """Chronological split (no shuffling) — important for time-series data."""
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
