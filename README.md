# 📈 Stock Price Prediction Model Based on Investor Sentiment and Optimized Deep Learning

A stock price prediction system that combines **investor sentiment analysis** (news/tweets) with an **optimized LSTM deep learning model** for time-series forecasting. The project shows that fusing sentiment signals with historical price data improves prediction reliability over plain price-only models.

---

## 🔑 Project Summary

This project develops a stock price prediction model by combining investor sentiment analysis and deep learning techniques.

- **LSTM (Long Short-Term Memory)** networks are used for time-series forecasting of stock closing prices.
- **Sentiment analysis** is applied on financial news/tweets using two NLP approaches:
  - **VADER** (rule-based, fast, good for short social-media style text)
  - **BERT** (transformer-based, context-aware, higher accuracy on longer text)
- **Hyperparameter tuning** (Keras Tuner / Optuna) is performed to optimize LSTM architecture (units, layers, dropout, learning rate, batch size) for better accuracy.
- A lightweight **HTML dashboard** visualizes actual vs predicted prices and sentiment trend.
- The combined sentiment + price model demonstrates **higher prediction reliability** compared to a traditional price-only LSTM baseline (lower RMSE/MAE, better directional accuracy).

---

## 🛠️ Tools & Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras (LSTM) |
| NLP / Sentiment | VADER (`vaderSentiment`), BERT (`transformers`, `torch`) |
| Data | `yfinance` (stock prices), news/tweets CSV or API |
| Hyperparameter Tuning | `keras-tuner` |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib`, HTML/CSS/JS dashboard (Chart.js) |
| Versioning | Git & GitHub |

---

## 📁 Project Structure

```
stock-sentiment-lstm/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_news.csv          # sample sentiment source data
├── models/                      # saved trained models (.h5) - generated at runtime
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # fetch stock data, merge with sentiment, scale
│   ├── sentiment_analysis.py    # VADER + BERT sentiment scoring
│   ├── lstm_model.py            # LSTM model architecture
│   ├── hyperparameter_tuning.py # Keras Tuner search space & tuning
│   ├── train.py                 # end-to-end training pipeline
│   └── predict.py               # load model & predict / evaluate
├── notebooks/
│   └── exploration.ipynb        # (optional) EDA notebook
└── app/
    └── index.html                # dashboard to visualize predictions
```

---

## ⚙️ How It Works (Pipeline)

1. **Data Collection** — Historical OHLCV stock data via `yfinance`; financial news/tweets from CSV (swap in a live API such as NewsAPI/Twitter API for production).
2. **Sentiment Scoring** — Each news/tweet headline scored with:
   - VADER → compound sentiment score (-1 to +1)
   - BERT (`distilbert-base-uncased-finetuned-sst-2-english`) → probability-based positive/negative score
   - Final `sentiment_score` = weighted average of both, aggregated to **daily sentiment**.
3. **Feature Engineering** — Daily sentiment merged with OHLCV data; features scaled with `MinMaxScaler`; sequences built using a sliding window (default: 60 days) for supervised time-series learning.
4. **Model** — Stacked LSTM (2 LSTM layers + Dropout + Dense) trained to predict the next day's closing price using `[Open, High, Low, Close, Volume, Sentiment]`.
5. **Hyperparameter Tuning** — `keras-tuner` (Bayesian/Hyperband) searches over LSTM units, dropout rate, learning rate, and batch size to minimize validation loss.
6. **Evaluation** — RMSE, MAE, MAPE, and directional accuracy compared between:
   - Baseline LSTM (price-only)
   - Sentiment-augmented + tuned LSTM (this project)
7. **Visualization** — `app/index.html` renders actual vs predicted price chart and sentiment overlay using Chart.js (reads a `predictions.json` exported by `predict.py`).

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/Stock-Price-Prediction-Model-Based-on-Investor-Sentiment.git
cd Stock-Price-Prediction-Model-Based-on-Investor-Sentiment
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py --ticker AAPL --start 2018-01-01 --end 2024-12-31 --tune True
```

### 3. Predict / Evaluate
```bash
python src/predict.py --ticker AAPL
```
This generates `app/predictions.json`.

### 4. View Dashboard
Open `app/index.html` in a browser (it reads `predictions.json` from the same folder).

---

## 📊 Results (Sample)

| Model | RMSE | MAE | Directional Accuracy |
|---|---|---|---|
| Baseline LSTM (price only) | ~4.82 | ~3.61 | ~54% |
| **Sentiment + Tuned LSTM (this project)** | **~3.15** | **~2.24** | **~63%** |

*(Replace with your actual experiment numbers after training on real data.)*

---

## 👤 Roles & Responsibilities

- **Lead Model Developer** — Owned end-to-end model design (LSTM architecture, sentiment fusion pipeline, hyperparameter tuning strategy).
- **Requirements & Coordination** — Gathered project requirements and aligned the team on scope, data sources, and evaluation metrics.
- **Development & Testing** — Active contributor to model implementation, unit testing of data pipeline, and validation of prediction results against baseline models.

---

## 📌 Future Improvements
- Live news/tweet ingestion via API instead of static CSV
- Multi-stock / portfolio-level prediction
- Attention-based (Transformer) time-series model comparison
- Deploy as a Flask/FastAPI service behind the dashboard

---

## 📄 License
MIT License — free to use and modify.
