# 🌫️ Air Quality Monitoring and Prediction System

An end-to-end time-series forecasting pipeline for predicting air quality (PM2.5/AQI) using Python. Compares multiple ML models with visualization-driven performance analysis.

## 📋 Overview

- **Multi-model comparison**: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR, LSTM, BiLSTM
- **End-to-end pipeline**: Data loading → Preprocessing → Feature Engineering → Training → Evaluation → Visualization
- **Supported datasets**: UCI Air Quality, Beijing PM2.5, OpenAQ, synthetic data
- **AQI computation** from raw PM2.5 concentrations (US EPA standard)

## 🗂️ Project Structure

```
AirQuality/
├── models/
│   ├── lstm_model.py         # LSTM & BiLSTM with early stopping
│   └── baseline_models.py    # sklearn model comparison suite
├── utils/
│   ├── data_preprocessing.py # Data loading, cleaning, scaling, time features
│   └── visualization.py      # Time-series plots, heatmaps, prediction overlays
├── plots/                    # Auto-generated visualization outputs
├── train.py                  # Main pipeline script
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run with synthetic data (demo)
python train.py

# Run on UCI Air Quality dataset
python train.py --data data/AirQualityUCI.csv

# Run on Beijing PM2.5 dataset  
python train.py --data data/beijing_pm25.csv

# Run only LSTM model
python train.py --model lstm

# Run only baseline ML models
python train.py --model baseline --skip-eda
```

## 📊 Datasets

| Dataset | Link | Description |
|---|---|---|
| UCI Air Quality | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality) | Hourly Italian sensor data |
| Beijing PM2.5 | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) | 5-year Beijing hourly data |
| OpenAQ | [openaq.org](https://openaq.org/) | Global real-time AQ data |

## 🔬 Models & Techniques

### Preprocessing
- Time interpolation for missing sensor values
- Cyclical encoding of time features (sin/cos for hour, day, month)
- MinMax scaling (fit on train only, applied to val/test)
- Chronological 70/10/20 split (no data leakage)

### Baseline Models
- Lag features (1h, 2h, 3h, 6h, 12h, 24h lookback)
- Rolling mean/std statistics
- Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR

### LSTM
- Stacked 2-layer LSTM with dropout
- Gradient clipping
- ReduceLROnPlateau scheduler + early stopping
- 24-hour sliding window input → 1-hour ahead prediction

## 📈 Typical Results (Beijing PM2.5)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~28.4 | ~18.2 | 0.71 |
| Random Forest | ~18.6 | ~11.4 | 0.88 |
| Gradient Boosting | ~17.2 | ~10.8 | 0.89 |
| LSTM | ~14.3 | ~9.1 | 0.93 |
| BiLSTM | ~13.8 | ~8.7 | 0.94 |

## 🛠️ Tech Stack

- Python, PyTorch, Scikit-learn, NumPy, Pandas
- Matplotlib, Seaborn (visualization)
- MATLAB-compatible preprocessing logic
