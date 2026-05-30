"""
Feature engineering for traffic time-series data.
Used by both LSTM and XGBoost training pipelines.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os


def make_label(index: float) -> int:
    """Map congestion index (0-100) → class label (0-3)."""
    if index <= 25: return 0   # Free Flow
    if index <= 50: return 1   # Moderate
    if index <= 75: return 2   # Heavy
    return 3                   # Gridlock


LABEL_NAMES = {0: 'free_flow', 1: 'moderate', 2: 'heavy', 3: 'gridlock'}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Cyclical time encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Lag features (5-min intervals)
    for lag in [1, 2, 3, 6, 9, 12]:
        df[f'count_lag_{lag}'] = df['vehicle_count'].shift(lag)
        df[f'speed_lag_{lag}'] = df['avg_speed'].shift(lag)
        df[f'index_lag_{lag}'] = df['congestion_index'].shift(lag)

    # Rolling statistics
    df['count_roll5_mean']  = df['vehicle_count'].rolling(5,  min_periods=1).mean()
    df['count_roll15_mean'] = df['vehicle_count'].rolling(15, min_periods=1).mean()
    df['speed_roll5_std']   = df['avg_speed'].rolling(5,      min_periods=1).std().fillna(0)
    df['index_roll5_mean']  = df['congestion_index'].rolling(5, min_periods=1).mean()

    # Rate of change
    df['index_delta'] = df['congestion_index'].diff().fillna(0)
    df['count_delta'] = df['vehicle_count'].diff().fillna(0)

    # Binary flags
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['is_raining']   = (df.get('rainfall_mm', pd.Series(0, index=df.index)) > 0.5).astype(int)

    # Targets
    df['label']        = df['congestion_index'].apply(make_label)
    df['target_15min'] = df['congestion_index'].shift(-3)
    df['target_30min'] = df['congestion_index'].shift(-6)
    df['target_60min'] = df['congestion_index'].shift(-12)

    return df.dropna()


def get_feature_cols() -> list:
    return [
        'vehicle_count', 'avg_speed', 'congestion_index',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'count_lag_1', 'count_lag_2', 'count_lag_3', 'count_lag_6', 'count_lag_12',
        'speed_lag_1', 'speed_lag_3', 'speed_lag_6',
        'index_lag_1', 'index_lag_3', 'index_lag_6', 'index_lag_12',
        'count_roll5_mean', 'count_roll15_mean', 'speed_roll5_std', 'index_roll5_mean',
        'index_delta', 'count_delta',
        'is_weekend', 'is_rush_hour', 'is_raining',
    ]


def make_sequences(df, seq_len=12, feat_cols=None):
    if feat_cols is None:
        feat_cols = get_feature_cols()
    X, y = [], []
    feats   = df[feat_cols].values.astype(np.float32)
    targets = df[['target_15min', 'target_30min', 'target_60min']].values.astype(np.float32)
    for i in range(len(feats) - seq_len):
        X.append(feats[i: i + seq_len])
        y.append(targets[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def chronological_split(df, train=0.70, val=0.15):
    n  = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def fit_scaler(train_df, feat_cols, save_path=None):
    scaler = MinMaxScaler()
    scaler.fit(train_df[feat_cols])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    return scaler
