"""
train.py
---------
End-to-end training pipeline:
  1. Fetch stock price data
  2. Compute investor sentiment from news data
  3. Merge, scale, and build sequences
  4. Train (optionally tune) the LSTM model
  5. Evaluate against a price-only baseline
  6. Save the trained model
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import (
    fetch_stock_data, merge_with_sentiment, scale_features,
    create_sequences, train_test_split_sequences, FEATURE_COLS,
)
from sentiment_analysis import SentimentAnalyzer
from lstm_model import build_lstm_model
from hyperparameter_tuning import run_hyperparameter_search


def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def main(args):
    print(f"Fetching price data for {args.ticker}...")
    price_df = fetch_stock_data(args.ticker, args.start, args.end)

    print("Scoring investor sentiment from news data...")
    sa = SentimentAnalyzer(use_bert=args.use_bert)
    news_df = pd.read_csv(args.news_csv)
    sentiment_df = sa.score_dataframe(news_df)

    print("Merging price + sentiment data...")
    merged = merge_with_sentiment(price_df, sentiment_df)

    print("Scaling features and building sequences...")
    scaled, scaler = scale_features(merged, FEATURE_COLS)
    X, y = create_sequences(scaled, window=args.window, target_col_idx=FEATURE_COLS.index("Close"))
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, test_ratio=0.2)

    input_shape = (X_train.shape[1], X_train.shape[2])

    if args.tune:
        print("Running hyperparameter search (this can take a while)...")
        model, best_hp = run_hyperparameter_search(X_train, y_train, X_test, y_test, input_shape)
    else:
        print("Building default LSTM model...")
        model = build_lstm_model(input_shape)

    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    print("Evaluating...")
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    dir_acc = directional_accuracy(y_test, y_pred)

    print(f"RMSE (scaled): {rmse:.4f}")
    print(f"MAE  (scaled): {mae:.4f}")
    print(f"Directional Accuracy: {dir_acc * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.ticker}_lstm_sentiment.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment-augmented LSTM stock predictor")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--news_csv", type=str, default="data/sample_news.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tune", type=bool, default=False)
    parser.add_argument("--use_bert", type=bool, default=False)
    args = parser.parse_args()
    main(args)
