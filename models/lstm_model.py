"""
LSTM-based Air Quality Forecasting
Predicts PM2.5 / AQI values using multi-variate time-series sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AirQualityDataset(Dataset):
    """Sliding window dataset for time-series forecasting."""

    def __init__(self, data: np.ndarray, seq_len=24, pred_len=1, target_col=0):
        """
        Args:
            data: (T, n_features) normalized array
            seq_len: Input sequence length (hours)
            pred_len: Prediction horizon (hours ahead)
            target_col: Column index of target variable (e.g., PM2.5)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.X, self.y = [], []

        for i in range(len(data) - seq_len - pred_len + 1):
            self.X.append(data[i: i + seq_len])
            self.y.append(data[i + seq_len: i + seq_len + pred_len, target_col])

        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM for multi-variate air quality forecasting.
    Architecture: LSTM → Dropout → LSTM → FC → Output
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, pred_len=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # Last time step
        return self.fc(out)                  # (batch, pred_len)


class BiLSTMForecaster(nn.Module):
    """Bidirectional LSTM for air quality forecasting."""

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.2, pred_len=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class AirQualityTrainer:
    """Training loop with learning rate scheduling and early stopping."""

    def __init__(self, model, lr=1e-3, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.train_losses, self.val_losses = [], []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return np.sqrt(total_loss / len(loader.dataset))

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_targets = [], []
        for X, y in loader:
            X = X.to(self.device)
            pred = self.model(X).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((targets - preds) / (np.abs(targets) + 1e-8))) * 100
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}, preds, targets

    def fit(self, train_loader, val_loader, n_epochs=50, patience=10):
        best_val = float('inf')
        patience_ctr = 0
        best_state = None

        for epoch in range(1, n_epochs + 1):
            train_rmse = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            val_rmse = val_metrics['RMSE']

            self.train_losses.append(train_rmse)
            self.val_losses.append(val_rmse)
            self.scheduler.step(val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                torch.save(best_state, 'best_lstm.pt')
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 5 == 0:
                print(f"Epoch {epoch:>3} | Train RMSE: {train_rmse:.4f} | "
                      f"Val RMSE: {val_rmse:.4f} | Val MAE: {val_metrics['MAE']:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)
        return self
