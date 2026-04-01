"""
Air Quality Monitoring and Prediction System — Main Script
End-to-end pipeline: data loading → preprocessing → model training → evaluation → visualization

Usage:
    python train.py                          # Synthetic data, all models
    python train.py --data path/to/data.csv  # Real dataset
    python train.py --model lstm             # Only LSTM
    python train.py --model baseline         # Only sklearn baselines
"""

import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils.data_preprocessing import (
    generate_synthetic_data, load_uci_airquality, load_beijing_pm25,
    preprocess_pipeline, inverse_transform_target
)
from utils.visualization import (
    plot_time_series, plot_correlation_heatmap,
    plot_daily_pattern, plot_model_comparison,
    plot_predictions_vs_actual, plot_training_curves
)
from models.lstm_model import AirQualityDataset, LSTMForecaster, BiLSTMForecaster, AirQualityTrainer
from models.baseline_models import create_lag_features, ModelComparison


TARGET = 'PM2.5'
SEQ_LEN = 24   # 24-hour lookback
PRED_LEN = 1   # 1-hour ahead


def run_eda(df):
    print("\n--- Exploratory Data Analysis ---")
    print(df.describe().round(2))
    plot_time_series(df, TARGET)
    plot_correlation_heatmap(df)
    plot_daily_pattern(df, TARGET)


def run_baseline(df):
    print("\n--- Baseline ML Models ---")
    df_feat = create_lag_features(df, TARGET)
    feature_cols = [c for c in df_feat.columns if c != TARGET]

    n = len(df_feat)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.8)

    X_train = df_feat.iloc[:train_end][feature_cols].values
    y_train = df_feat.iloc[:train_end][TARGET].values
    X_val   = df_feat.iloc[train_end:val_end][feature_cols].values
    y_val   = df_feat.iloc[train_end:val_end][TARGET].values
    X_test  = df_feat.iloc[val_end:][feature_cols].values
    y_test  = df_feat.iloc[val_end:][TARGET].values

    comp = ModelComparison()
    summary = comp.fit_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
    print("\n" + summary.to_string())

    results = {name: {'RMSE': summary.loc[name, 'Test RMSE'], 'MAE': summary.loc[name, 'Test MAE']}
               for name in summary.index}
    plot_model_comparison(results, metric='RMSE')
    return comp, results


def run_lstm(train_scaled, val_scaled, test_scaled, scaler):
    print("\n--- LSTM Forecasting ---")
    train_ds = AirQualityDataset(train_scaled, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    val_ds   = AirQualityDataset(val_scaled,   seq_len=SEQ_LEN, pred_len=PRED_LEN)
    test_ds  = AirQualityDataset(test_scaled,  seq_len=SEQ_LEN, pred_len=PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=256)
    test_loader  = DataLoader(test_ds,  batch_size=256)

    n_features = train_scaled.shape[1]
    model = LSTMForecaster(input_size=n_features, hidden_size=128, num_layers=2, pred_len=PRED_LEN)
    trainer = AirQualityTrainer(model, lr=1e-3)
    trainer.fit(train_loader, val_loader, n_epochs=50, patience=10)

    metrics, preds, targets = trainer.evaluate(test_loader)
    print(f"\nLSTM Test — RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | MAPE: {metrics['MAPE']:.2f}%")

    plot_training_curves(trainer.train_losses, trainer.val_losses)
    plot_predictions_vs_actual(targets.flatten(), preds.flatten(), model_name='LSTM')
    return trainer, metrics


def main(args):
    print("Air Quality Monitoring and Prediction System")

    # Load data
    if args.data:
        if 'beijing' in args.data.lower():
            df = load_beijing_pm25(args.data)
        else:
            df = load_uci_airquality(args.data)
    else:
        print("No dataset specified — using synthetic data.")
        df = generate_synthetic_data(n_hours=8760)

    # EDA
    if not args.skip_eda:
        run_eda(df)

    # Preprocess
    train_s, val_s, test_s, scaler, feat_cols = preprocess_pipeline(df, target_col=TARGET)

    all_results = {}

    # Baselines
    if args.model in ('all', 'baseline'):
        _, baseline_results = run_baseline(df)
        all_results.update(baseline_results)

    # LSTM
    if args.model in ('all', 'lstm'):
        _, lstm_metrics = run_lstm(train_s, val_s, test_s, scaler)
        all_results['LSTM'] = lstm_metrics

    # Final comparison
    if len(all_results) > 1:
        print("\n--- Final Model Comparison ---")
        plot_model_comparison(all_results, metric='RMSE')

    print("\nDone! Plots saved to plots/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--model', default='all', choices=['all', 'lstm', 'baseline'])
    parser.add_argument('--skip-eda', action='store_true')
    args = parser.parse_args()
    main(args)
