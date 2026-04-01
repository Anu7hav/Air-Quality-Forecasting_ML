"""
Data Loading and Preprocessing Pipeline for Air Quality Monitoring
Supports: UCI Air Quality Dataset, Beijing PM2.5, OpenAQ CSV exports
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os


# ────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ────────────────────────────────────────────────────────────────────────────

def load_uci_airquality(filepath: str) -> pd.DataFrame:
    """
    Load UCI Air Quality Dataset (AirQualityUCI.csv).
    Download: https://archive.ics.uci.edu/ml/datasets/Air+Quality

    Columns: Date, Time, CO(GT), PT08.S1(CO), NMHC(GT), C6H6(GT),
             PT08.S2(NMHC), NOx(GT), PT08.S3(NOx), NO2(GT),
             PT08.S4(NO2), PT08.S5(O3), T, RH, AH
    """
    df = pd.read_csv(filepath, sep=';', decimal=',', parse_dates=False)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%d/%m/%Y %H.%M.%S')
    df = df.set_index('datetime').drop(columns=['Date', 'Time'])

    # Replace -200 with NaN (sensor error code)
    df = df.replace(-200, np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')

    print(f"UCI AQ Dataset: {df.shape[0]} hourly records, {df.shape[1]} features")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df


def load_beijing_pm25(filepath: str) -> pd.DataFrame:
    """
    Load Beijing PM2.5 dataset.
    Download: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
    """
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.set_index('datetime').drop(columns=['No', 'year', 'month', 'day', 'hour'])

    # Encode wind direction
    if 'cbwd' in df.columns:
        df = pd.get_dummies(df, columns=['cbwd'], prefix='wind')

    df = df.apply(pd.to_numeric, errors='coerce')
    print(f"Beijing PM2.5: {df.shape[0]} hourly records, {df.shape[1]} features")
    return df


def generate_synthetic_data(n_hours=8760, seed=42) -> pd.DataFrame:
    """
    Generate synthetic air quality data for testing.
    Simulates realistic AQI patterns with daily/seasonal cycles.
    """
    rng = np.random.RandomState(seed)
    hours = pd.date_range('2023-01-01', periods=n_hours, freq='h')

    t = np.arange(n_hours)
    daily_cycle = np.sin(2 * np.pi * t / 24)
    seasonal_cycle = np.sin(2 * np.pi * t / (24 * 365))
    trend = 0.001 * t

    pm25 = (
        30 + 20 * daily_cycle + 15 * seasonal_cycle + trend
        + rng.normal(0, 5, n_hours)
    ).clip(0)

    df = pd.DataFrame({
        'PM2.5': pm25,
        'PM10': pm25 * rng.uniform(1.2, 1.8, n_hours),
        'NO2': 20 + 10 * daily_cycle + rng.normal(0, 3, n_hours),
        'CO': 1.0 + 0.3 * daily_cycle + rng.normal(0, 0.1, n_hours),
        'O3': 60 - 20 * daily_cycle + rng.normal(0, 5, n_hours),
        'Temperature': 15 + 10 * seasonal_cycle + 5 * daily_cycle + rng.normal(0, 2, n_hours),
        'Humidity': 60 + 15 * seasonal_cycle - 10 * daily_cycle + rng.normal(0, 5, n_hours),
        'WindSpeed': rng.exponential(2, n_hours),
    }, index=hours)

    df = df.clip(lower=0)
    print(f"Generated synthetic AQ data: {len(df)} hourly records")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame, target_col='PM2.5',
                        fill_method='interpolate', scaler_type='minmax'):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. Add time features
    3. Normalize features
    4. Train/val/test split (70/10/20, no shuffle — time-series)

    Returns:
        train, val, test splits (DataFrames), fitted scaler, feature columns
    """
    df = df.copy()

    # 1. Missing values
    if fill_method == 'interpolate':
        df = df.interpolate(method='time').ffill().bfill()
    elif fill_method == 'forward':
        df = df.ffill().bfill()
    elif fill_method == 'mean':
        df = df.fillna(df.mean())

    # 2. Time features
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour_sin']    = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos']    = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin']     = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos']     = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin']   = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos']   = np.cos(2 * np.pi * df.index.month / 12)
        df['is_weekend']  = (df.index.dayofweek >= 5).astype(float)

    # 3. Move target to front
    cols = [target_col] + [c for c in df.columns if c != target_col]
    df = df[cols]

    feature_cols = df.columns.tolist()
    n = len(df)

    # 4. Chronological split
    train_end = int(n * 0.70)
    val_end   = int(n * 0.80)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    # 5. Scale (fit only on train)
    ScalerClass = MinMaxScaler if scaler_type == 'minmax' else StandardScaler
    scaler = ScalerClass()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled   = scaler.transform(val_df.values)
    test_scaled  = scaler.transform(test_df.values)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Features: {feature_cols}")
    return train_scaled, val_scaled, test_scaled, scaler, feature_cols


def inverse_transform_target(scaler, preds: np.ndarray, target_idx=0):
    """Inverse scale predictions back to original units."""
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, target_idx] = preds.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


def compute_aqi(pm25: float) -> tuple:
    """
    Compute US AQI from PM2.5 concentration (μg/m³).
    Returns (AQI value, category string).
    """
    breakpoints = [
        (0,    12.0,   0,   50,  'Good'),
        (12.1, 35.4,   51,  100, 'Moderate'),
        (35.5, 55.4,   101, 150, 'Unhealthy for Sensitive Groups'),
        (55.5, 150.4,  151, 200, 'Unhealthy'),
        (150.5,250.4,  201, 300, 'Very Unhealthy'),
        (250.5,500.4,  301, 500, 'Hazardous'),
    ]

    for bp_lo, bp_hi, i_lo, i_hi, category in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + i_lo
            return round(aqi), category

    return 500, 'Hazardous'
