"""
Baseline ML Models for Air Quality Prediction
Compares: Linear Regression, Random Forest, XGBoost, SVR, and LSTM
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def create_lag_features(df: pd.DataFrame, target_col: str,
                        lags=(1, 2, 3, 6, 12, 24),
                        rolling_windows=(3, 6, 12, 24)):
    """
    Create lag features and rolling statistics for tabular ML models.

    Args:
        df: DataFrame with time-series columns
        target_col: Column to predict (e.g., 'PM2.5')
        lags: Lag hours to include as features
        rolling_windows: Window sizes for rolling mean/std
    Returns:
        DataFrame with added lag and rolling features
    """
    df = df.copy()

    # Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        df[f'{target_col}_rollmean_{window}h'] = (
            df[target_col].shift(1).rolling(window).mean()
        )
        df[f'{target_col}_rollstd_{window}h'] = (
            df[target_col].shift(1).rolling(window).std()
        )

    # Time-based features
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    return df.dropna()


class ModelComparison:
    """
    Train and compare multiple regression models for AQI prediction.
    Provides unified interface and performance summary.
    """

    MODELS = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    }

    def __init__(self, models=None):
        self.models = models or self.MODELS
        self.scaler = StandardScaler()
        self.results = {}
        self.fitted_models = {}

    def fit_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train and evaluate all models, return comparison table."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)
        X_test_s  = self.scaler.transform(X_test)

        rows = []
        for name, model in self.models.items():
            print(f"Training {name}...", end=' ')
            model.fit(X_train_s, y_train)
            self.fitted_models[name] = model

            val_pred  = model.predict(X_val_s)
            test_pred = model.predict(X_test_s)

            row = {
                'Model': name,
                'Val RMSE':  np.sqrt(mean_squared_error(y_val, val_pred)),
                'Val MAE':   mean_absolute_error(y_val, val_pred),
                'Val R²':    r2_score(y_val, val_pred),
                'Test RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
                'Test MAE':  mean_absolute_error(y_test, test_pred),
                'Test R²':   r2_score(y_test, test_pred),
            }
            self.results[name] = {'val_pred': val_pred, 'test_pred': test_pred}
            rows.append(row)
            print(f"✓ | Val RMSE: {row['Val RMSE']:.4f} | Test RMSE: {row['Test RMSE']:.4f}")

        self.summary = pd.DataFrame(rows).set_index('Model').round(4)
        return self.summary

    def best_model(self, metric='Test RMSE'):
        return self.summary[metric].idxmin()

    def predict(self, model_name: str, X: np.ndarray):
        X_scaled = self.scaler.transform(X)
        return self.fitted_models[model_name].predict(X_scaled)

    def feature_importance(self, model_name: str, feature_names: list):
        """Return feature importances for tree-based models."""
        model = self.fitted_models[model_name]
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            return imp.sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            imp = pd.Series(np.abs(model.coef_), index=feature_names)
            return imp.sort_values(ascending=False)
        else:
            raise ValueError(f"{model_name} does not support feature importance.")
