<<<<<<< HEAD
# Stock Market Prediction Baseline

This repository contains a baseline LSTM model for stock market trend prediction (binary classification of daily price movements).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-baseline.git
   cd stock-prediction-baseline
=======
# Stock Market Price Movement Prediction

A project to predict the direction of stock price movement using historical data and technical indicators.

## Objective

To develop and compare multiple machine learning and deep learning models for binary classification:  
Will the stock price increase by more than 0.2% on the next day?

## Dataset

- Source: [Yahoo Finance](https://finance.yahoo.com/)
- Period: `2010-01-01` — `2023-12-31`
- Tickers: `AAPL`, `GOOG`, `TSLA`, `MSFT`, `NFLX`, `NVDA`, `META`, `AMZN`
- Features:
  - RSI, MACD, Moving Averages
  - Daily return, Volatility, Bollinger Bands
  - Momentum, OBV
  - Lag features: `Close_t-1`, `Close_t-2`, etc.

## Models & Results

| Model           | Accuracy | ROC AUC | F1 Score |
|------------------|----------|---------|----------|
| LSTM             | ~0.52    | ~0.52   | ~0.43    |
| GRU              | ~0.52    | ~0.52   | ~0.43    |
| **MLP**          | 0.505    | 0.527   | **0.549** |
| LightGBM         | **0.526**| **0.531**| 0.501    |
| Random Forest    | 0.512    | 0.530   | 0.497    |

> **MLP** achieved the best F1 score; **LightGBM** performed best overall in terms of balance.
> **LightGBM** offered the best tradeoff between accuracy and AUC

## Installation
```bash
pip install -r requirements.txt
python main.py --all
>>>>>>> 65686c6e75c25962cd1a17ec904714d39770b9ae
