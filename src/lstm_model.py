"""
lstm_model.py
--------------
Defines the stacked-LSTM architecture used for sentiment-augmented
stock price prediction.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape, units_1=128, units_2=64, dropout=0.2, learning_rate=0.001):
    """
    input_shape: (window_size, n_features)
    Returns a compiled Keras Sequential LSTM model.
    """
    model = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),  # predicted next-day close price (scaled)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
