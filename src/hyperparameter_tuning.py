"""
hyperparameter_tuning.py
--------------------------
Uses keras-tuner (Hyperband) to search LSTM hyperparameters:
units per layer, dropout rate, learning rate, and batch size.
"""

import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_hypermodel(input_shape):
    def model_builder(hp):
        units_1 = hp.Int("units_1", min_value=32, max_value=256, step=32)
        units_2 = hp.Int("units_2", min_value=16, max_value=128, step=16)
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
        lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model = Sequential([
            LSTM(units_1, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units_2, return_sequences=False),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
        return model

    return model_builder


def run_hyperparameter_search(X_train, y_train, X_val, y_val, input_shape,
                                max_epochs=30, directory="tuner_dir", project_name="stock_lstm_tuning"):
    """
    Runs Hyperband search and returns the best model + best hyperparameters.
    """
    tuner = kt.Hyperband(
        build_hypermodel(input_shape),
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory=directory,
        project_name=project_name,
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, verbose=1)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    print("Best hyperparameters found:")
    for param in ["units_1", "units_2", "dropout", "learning_rate"]:
        print(f"  {param}: {best_hp.get(param)}")

    return best_model, best_hp
