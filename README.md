# Optimized Bidirectional LSTM for Intraday Stock Price Prediction

This file implements an advanced **Bidirectional Long Short-Term Memory (Bi-LSTM)** network for **intraday stock price forecasting**, focusing on predicting the next 30 five-minute closing prices for a given stock.

***

## 1. Core Model and Architecture

The model is a deep, multi-layered Bi-LSTM designed to process sequences both forward and backward in time, enhancing its ability to capture temporal patterns. It uses a **multi-step output** predicting 5 intervals simultaneously and features L2 regularization and Dropout to prevent overfitting.

| Layer Type | Units/Details | Output Shape | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | `(90, 5)` | N/A | **90 past 5-minute intervals** of OHLCV data. |
| **Bidirectional LSTM (1)**| 150 units, L2 Reg. | `(None, 90, 300)` | Initial feature extraction. |
| **Bidirectional LSTM (2)**| 100 units, L2 Reg. | `(None, 90, 200)` | Deeper temporal pattern learning. |
| **LSTM (3)** | **75 units** (Increased Capacity), L2 Reg. | `(None, 75)` | Condenses time series into a single feature vector. |
| **Output (Dense)** | **5 units** | `(None, 5)` | Predicts the **next 5 closing prices** simultaneously (prediction horizon). |

**Total Trainable Parameters:** **591,615**.

### Training Optimization

* **Optimizer:** **AdamW** is used with a base learning rate of `0.0005`.
* **Scaling:** **`RobustScaler`** is applied to all 5 features (OHLCV) to minimize the impact of outliers.
* **Callbacks:** Includes **`EarlyStopping`** (patience 15) and **`ReduceLROnPlateau`** (patience 7, factor 0.5).
* **Best Validation Loss:** The best achieved scaled Mean Squared Error (MSE) loss during training was **$0.0431$**.

***

## 2. Data and Sequence Configuration

The system uses high-frequency intraday data and is configured for multi-step forecasting:

| Configuration | Value | Description |
| :--- | :--- | :--- |
| **Stock Ticker** | `GOOG` | Stock data retrieved via `yfinance`. |
| **Data Interval** | `5m` | The frequency of the time series data. |
| **Input Features** | **5** (Open, High, Low, Close, Volume) | Features used for training. |
| **Sequence Length**| **90** (`time_steps`) | Number of past 5-minute bars used as input for each prediction. |
| **Prediction Horizon**| **5** | Number of future steps predicted in one model call. |
| **Total Forecast** | **30 intervals** | Final forecast duration, achieved via the rolling window.
| **Target Variable Index** | **3** | The correct index for the **Close Price** in the OHLCV dataframe. |

***

## 3. Evaluation Metrics and Model Comparison

The final model's performance is evaluated using the inverse-transformed price for the **1-Step Ahead** prediction on the test set, and compared against a Standard LSTM benchmark:

| Metric | Bidirectional LSTM (BiLSTM) | Standard LSTM |
| :--- | :--- | :--- |
| **R² Score** | **$0.8945$** | $0.8879$ |
| **RMSE (USD)** | **$0.8606$** | $0.8872$ |
| **MAE (USD)** | **$0.5954$** | $0.6018$ |

The **Bidirectional LSTM** model demonstrated superior performance across all metrics, indicating its effectiveness in modeling the temporal dependencies of the intraday stock data.
