"""
Train LSTM congestion forecaster.

Run EITHER:
  python ml/src/train_lstm.py          ← pulls data from Django DB
  python ml/src/train_lstm.py --csv    ← uses ml/data/processed/traffic_data.csv

Outputs:
  ml/saved_models/lstm_best.pt
  ml/saved_models/lstm_best.onnx
  ml/saved_models/scaler.pkl
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# ── allow running from project root ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'seq_len':     12,
    'hidden_size': 128,
    'num_layers':  2,
    'dropout':     0.3,
    'lr':          1e-3,
    'batch_size':  64,
    'epochs':      100,
    'patience':    10,
}


def load_from_db() -> pd.DataFrame:
    """Pull traffic readings from Django DB via Django ORM."""
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'traffic_project.settings')
    django.setup()
    from apps.traffic.models import TrafficReading
    qs = TrafficReading.objects.all().values(
        'timestamp', 'vehicle_count', 'avg_speed',
        'congestion_index', 'rainfall_mm'
    ).order_by('timestamp')
    df = pd.DataFrame(list(qs))
    logger.info(f"Loaded {len(df)} rows from database")
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def generate_synthetic(n=5000) -> pd.DataFrame:
    """Generate synthetic data for pipeline testing."""
    import math, random
    from datetime import datetime, timedelta
    rows  = []
    start = datetime(2024, 1, 1)
    for i in range(n):
        ts   = start + timedelta(minutes=5 * i)
        h, w = ts.hour, ts.weekday()
        base = (20 + 15 * math.sin(math.pi * (h - 10) / 12)
                if w >= 5 else
                max(60 * math.exp(-0.5 * ((h - 8) / 1.5) ** 2),
                    70 * math.exp(-0.5 * ((h - 17) / 1.5) ** 2)) + 10)
        idx  = max(0.0, min(100.0, base + random.gauss(0, 5)))
        rows.append({
            'timestamp':        ts.isoformat(),
            'vehicle_count':    max(0, int(idx * 1.2 + random.gauss(0, 6))),
            'avg_speed':        round(max(5.0, 60 - idx * 0.55 + random.gauss(0, 2)), 1),
            'congestion_index': round(idx, 2),
            'rainfall_mm':      0.0,
        })
    return pd.DataFrame(rows)


def train(df: pd.DataFrame):
    from ml.src.features import (engineer_features, get_feature_cols,
                                  make_sequences, chronological_split, fit_scaler)
    from ml.src.lstm_model import TrafficLSTM, EarlyStopping

    os.makedirs(SAVE_DIR, exist_ok=True)

    df        = engineer_features(df)
    feat_cols = get_feature_cols()
    train_df, val_df, test_df = chronological_split(df)
    logger.info(f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    scaler = fit_scaler(train_df, feat_cols,
                        save_path=os.path.join(SAVE_DIR, 'scaler.pkl'))
    for sdf in [train_df, val_df, test_df]:
        sdf[feat_cols] = scaler.transform(sdf[feat_cols])

    X_train, y_train = make_sequences(train_df, CONFIG['seq_len'], feat_cols)
    X_val,   y_val   = make_sequences(val_df,   CONFIG['seq_len'], feat_cols)
    X_test,  y_test  = make_sequences(test_df,  CONFIG['seq_len'], feat_cols)

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=CONFIG['batch_size'], shuffle=True)

    model      = TrafficLSTM(X_train.shape[2], CONFIG['hidden_size'],
                              CONFIG['num_layers'], dropout=CONFIG['dropout']).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion  = nn.MSELoss()
    stopper    = EarlyStopping(CONFIG['patience'])
    best_val   = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(
                model(torch.tensor(X_val).to(DEVICE)),
                torch.tensor(y_val).to(DEVICE)
            ).item()

        scheduler.step(v_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d} | train={t_loss/len(train_dl):.4f} val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'lstm_best.pt'))

        if stopper.step(v_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'lstm_best.pt')))
    model.eval()
    with torch.no_grad():
        preds   = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
        actuals = y_test
    mae  = np.mean(np.abs(preds - actuals))
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    logger.info(f"✅ Test MAE={mae:.3f}  MAPE={mape:.2f}%")

    # Export ONNX
    dummy     = torch.randn(1, CONFIG['seq_len'], X_train.shape[2]).to(DEVICE)
    onnx_path = os.path.join(SAVE_DIR, 'lstm_best.onnx')
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                      opset_version=14)
    logger.info(f"✅ ONNX saved → {onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      help='Path to CSV file (optional)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (for testing)')
    args = parser.parse_args()

    logger.info(f"🚀 Training LSTM on: {DEVICE}")

    if args.synthetic:
        df = generate_synthetic(8000)
    elif args.csv:
        df = load_from_csv(args.csv)
    else:
        try:
            df = load_from_db()
        except Exception as e:
            logger.warning(f"DB load failed ({e}) — using synthetic data")
            df = generate_synthetic(8000)

    if len(df) < 500:
        logger.warning(f"Only {len(df)} rows — using synthetic data instead")
        df = generate_synthetic(8000)

    train(df)
