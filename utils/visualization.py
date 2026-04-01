"""
Visualization utilities for Air Quality Analysis and Model Evaluation.
All plots saved to plots/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
os.makedirs('plots', exist_ok=True)


def plot_time_series(df: pd.DataFrame, col: str, title=None, save=True):
    """Plot a single air quality time-series variable."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[col], color=COLORS[0], linewidth=0.8, alpha=0.9)
    ax.fill_between(df.index, df[col], alpha=0.1, color=COLORS[0])
    ax.set_title(title or f'{col} Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{col}_timeseries.png', dpi=150, bbox_inches='tight')
        print(f"Saved: plots/{col}_timeseries.png")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save=True):
    """Correlation matrix of all air quality features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, ax=ax,
                annot_kws={'size': 8}, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig('plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_daily_pattern(df: pd.DataFrame, col: str, save=True):
    """Hourly box plot showing daily AQ patterns."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    df = df.copy()
    df['hour'] = df.index.hour

    fig, ax = plt.subplots(figsize=(12, 5))
    hourly = [df[df['hour'] == h][col].dropna().values for h in range(24)]
    bp = ax.boxplot(hourly, positions=range(24), patch_artist=True, notch=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.6)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(col)
    ax.set_title(f'Hourly {col} Distribution (Daily Pattern)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(24))
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{col}_daily_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: dict, metric='RMSE', save=True):
    """
    Bar chart comparing model performance.
    Args:
        results: Dict of {model_name: {metric: value}}
    """
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, values, color=COLORS[:len(models)], edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison — {metric}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/model_comparison_{metric}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, model_name='LSTM',
                               n_samples=500, save=True):
    """Overlay plot: predicted vs actual AQI values."""
    y_true = np.array(y_true)[:n_samples]
    y_pred = np.array(y_pred)[:n_samples]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time-series overlay
    axes[0].plot(y_true, label='Actual', color='#333', linewidth=1.2, alpha=0.9)
    axes[0].plot(y_pred, label=f'Predicted ({model_name})', color=COLORS[0],
                 linewidth=1.0, alpha=0.85, linestyle='--')
    axes[0].legend()
    axes[0].set_title(f'{model_name} — Predicted vs Actual', fontsize=13)
    axes[0].set_ylabel('PM2.5')

    # Scatter
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=15, color=COLORS[0])
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].legend()
    axes[1].set_title('Scatter: Predicted vs Actual')

    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{model_name}_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(train_losses, val_losses, model_name='LSTM', save=True):
    """Loss curve over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train RMSE', color=COLORS[0])
    ax.plot(val_losses, label='Val RMSE', color=COLORS[1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title(f'{model_name} Training Curves', fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'plots/{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
