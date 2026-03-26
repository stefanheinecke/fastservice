# 📊 Forecasting Apple Stock with a Deep Learning Model

## 🔍 Overview
This project builds and evaluates a neural network-based forecasting model to predict Apple Inc. (AAPL) closing prices. It uses historical market data combined with technical indicators to train a feedforward neural network. The system is capable of:

- Predicting the next day's price based on the most recent 5-day window
- Evaluating its performance using multiple metrics (MAE, RMSE, MAPE, Directional Accuracy)
- Forecasting five days into the future
- Visualizing predictions alongside real stock trends

---

## 💾 Data Source & Feature Engineering

### 🧠 Data Retrieval
- Source: [Yahoo Finance](https://finance.yahoo.com/)
- Ticker: `AAPL`
- Time Range: Last 5 years
- Raw Features: `Open`, `High`, `Low`, `Close`, `Volume`

### 🛠️ Technical Indicators
Using the `ta` library, the model includes key indicators:
- Momentum: RSI, Stochastic, MACD
- Volatility: Bollinger Bands, ATR
- Trend: EMA
- Volume-based: OBV

These indicators are engineered as input features to improve the model's understanding of price dynamics.

---

## 🧱 Modeling Pipeline

### 🪜 Step-by-Step Flow
1. Create 5-day input windows using the selected indicators.
2. Scale features and target (close price) separately using MinMaxScaler.
3. Split into training and test sets (80/20).
4. Train a feedforward neural network with:
   - 256-128-1 architecture
   - ReLU activations
   - Adam optimizer
5. Predict the next close price and evaluate performance.

---

## 📊 Evaluation Metrics

| Metric                | Description                                             |
|-----------------------|---------------------------------------------------------|
| **MAE**               | Average absolute dollar error                           |
| **RMSE**              | Penalizes larger errors more than MAE                   |
| **MAPE**              | Mean percentage error (scale-invariant)                 |
| **Directional Accuracy** | % of time the model correctly predicted price movement (up/down) |

### 🔢 Example Output

```text
Test MAE ≈ $3.22
RMSE: $4.12
Directional Accuracy: 97.99%
MAPE: 2.74%

Sample Predictions vs Actual:
  Actual: $174.23 | Predicted: $171.88 | Error: 1.35%
  Actual: $175.14 | Predicted: $176.42 | Error: 0.73%
  ...
