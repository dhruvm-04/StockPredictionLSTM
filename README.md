# Optimized Bidirectional LSTM for Intraday Stock Price Prediction

This file implements an advanced **Bidirectional Long Short-Term Memory (Bi-LSTM)** network for **intraday stock price forecasting**, focusing on predicting the next 30 five-minute closing prices for a given stock.

---

## 1. Core Model and Architecture

The model is designed as a deep, multi-layered Bi-LSTM to enhance temporal context capture by processing data both forward and backward through time.

| Layer Type | Units/Details | Output | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | `(90, 5)` | N/A | **90 past 5-minute intervals** of OHLCV data. |
| **Bidirectional LSTM (1)**| 150 units, `return_sequences=True`, L2 Reg. | `(None, 90, 300)` | Initial feature extraction. |
| **Bidirectional LSTM (2)**| 100 units, `return_sequences=True`, L2 Reg. | `(None, 90, 200)` | Deeper temporal pattern learning. |
| **LSTM (3)** | 50 units, `return_sequences=False`, L2 Reg. | `(None, 50)` | Condenses time series into a single feature vector. |
| **Dense** | 10 units, `relu` | `(None, 10)` | Intermediate non-linear projection. |
| **Output (Dense)** | **5 units** | `(None, 5)` | Predicts the **next 5 closing prices** simultaneously (multi-step output). |
| **Regularization** | `Dropout(0.3)` | N/A | Applied after each recurrent layer to prevent overfitting. |

**Total Trainable Parameters:** 558,765

### Training Optimization
* **Optimizer:** **AdamW** is used with a base learning rate of `0.0005`.
* **Callbacks:** Includes **`EarlyStopping`** (patience 15) and **`ReduceLROnPlateau`** (patience 7, factor 0.5) to manage convergence and prevent overfitting.
* **Scaling:** **`RobustScaler`** is used for all 5 features (OHLCV) to minimize the impact of price/volume outliers.

---

## 2. Data and Sequence Configuration

The system uses high-frequency intraday data and structures it for multi-step forecasting.

| Configuration | Value | Description |
| :--- | :--- | :--- |
| **Stock Ticker** | `GOOG` (Default) | Stock data retrieved via `yfinance`. |
| **Data Interval** | `5m` | The frequency of the time series data. |
| **Input Features** | **5** (Open, High, Low, Close, Volume) | Used for training the sequences. |
| **Sequence Length**| **90** (`time_steps`) | Number of past 5-minute bars used as input for each prediction. |
| **Prediction Horizon**| **5** | Number of future steps predicted in one model call. |
| **Total Forecast** | **30 intervals** | Final forecast duration, achieved via the rolling window. |

---

## 3. Forecasting Mechanism: Rolling Window

The model leverages its 5-step-ahead capability to generate a longer-term forecast (30 intervals) efficiently, replacing single-step predictions with blocks of 5.

1.  **Block Prediction:** The model predicts the next 5 future **Close Prices** in a single call.
2.  **Sequence Update:** The oldest 5 steps are dropped from the 90-step input sequence.
3.  **Feature Filling:** The 5 newly predicted Close Prices are used as proxy values for **Open, High, and Low** in the `new_entries` block. The **Volume** is carried forward from the last known value.
4.  **Recurse:** The new 90-step sequence is fed back into the model to predict the next block. This repeats until 30 intervals are predicted.

---

## 4. Evaluation Metrics (1-Step Ahead)

Performance on the test set, focusing on the nearest 5-minute prediction:

| Metric | Result | Interpretation (High Performance) |
| :--- | :--- | :--- |
| **R² Score** | `0.9023` | Model explains over 90% of the variance in the true price movements. |
| **RMSE** | `0.7626` | Average prediction error is approximately $0.76 USD$ in price scale. |
| **MAE** | `0.5067` | Average absolute error is approximately $0.51 USD$ in price scale. |

## 5. Usage Instructions

1. Download the ```.ipynb``` file or access the link here: (https://www.kaggle.com/code/dhruvmaheshwari2004/mlproject)
2. On Kaggle, access ```Settings > Accelerator``` and ensure acceletator is enabled as ```GPU T4 x2```
3. Run all cells
4. Use the ```Gradio UI``` built into the notebook or use the link that is generated above the output of the last cell to access the UI on a different tab **(you will have to keep the notebook session running after executing for the link to work)**
