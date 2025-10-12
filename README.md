# Intraday Stock Price Predictor

This project implements an advanced deep learning model for **intraday stock price forecasting**, designed to predict the price trend over the next **30 five-minute intervals**.

***

## Key Features

* **Model:** **Deep Bidirectional LSTM** network for enhanced sequence pattern recognition.
* **Data:** **Multivariate (OHLCV)** input for comprehensive analysis.
* **Stability:** Uses **RobustScaler** to mitigate the impact of financial outliers.
* **Performance:** Implements an **optimized multi-step rolling forecast** for fast inference.

***

## How to Run

1.  Run the entire **`mlproject.ipynb`** notebook on a machine with **GPU access** (recommended for training).
2.  The final cell launches an inbuilt UI for custom stock predictions.
