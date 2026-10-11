"""
predict.py
-----------
Loads a trained model, runs predictions on recent data, and exports
a predictions.json file consumed by the app/index.html dashboard.
"""

import argparse
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from data_preprocessing import (
    fetch_stock_data, merge_with_sentiment, scale_features,
    create_sequences, FEATURE_COLS,
)
from sentiment_analysis import SentimentAnalyzer


def main(args):
    model = load_model(f"models/{args.ticker}_lstm_sentiment.keras")

    price_df = fetch_stock_data(args.ticker, args.start, args.end)
    sa = SentimentAnalyzer(use_bert=args.use_bert)
    news_df = pd.read_csv(args.news_csv)
    sentiment_df = sa.score_dataframe(news_df)

    merged = merge_with_sentiment(price_df, sentiment_df)
    scaled, scaler = scale_features(merged, FEATURE_COLS)
    X, y = create_sequences(scaled, window=args.window, target_col_idx=FEATURE_COLS.index("Close"))

    y_pred_scaled = model.predict(X).flatten()

    # inverse-transform just the Close column
    close_idx = FEATURE_COLS.index("Close")
    dummy_true = np.zeros((len(y), len(FEATURE_COLS)))
    dummy_true[:, close_idx] = y
    dummy_pred = np.zeros((len(y_pred_scaled), len(FEATURE_COLS)))
    dummy_pred[:, close_idx] = y_pred_scaled

    y_true_actual = scaler.inverse_transform(dummy_true)[:, close_idx]
    y_pred_actual = scaler.inverse_transform(dummy_pred)[:, close_idx]

    dates = merged.index[args.window:].strftime("%Y-%m-%d").tolist()

    output = {
        "ticker": args.ticker,
        "dates": dates,
        "actual": y_true_actual.tolist(),
        "predicted": y_pred_actual.tolist(),
        "sentiment": merged["sentiment_score"].iloc[args.window:].tolist(),
    }

    with open("app/predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved app/predictions.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--news_csv", type=str, default="data/sample_news.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--use_bert", type=bool, default=False)
    args = parser.parse_args()
    main(args)
