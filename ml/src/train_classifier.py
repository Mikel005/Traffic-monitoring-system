"""
Train congestion classifier using scikit-learn GradientBoostingClassifier.
Produces the same xgb_classifier.pkl + xgb_scaler.pkl files that
apps/predictions/services.py expects — no XGBoost or network required.

Usage:
    python ml/src/train_classifier.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

SAVE_DIR    = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
LABEL_NAMES = ['free_flow', 'moderate', 'heavy', 'gridlock']
N_ROWS      = 12_000   # synthetic rows — enough for solid generalisation


# ── Synthetic data generator ─────────────────────────────────────────────────

def _generate_synthetic(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    # Simulate 5-minute intervals starting from a fixed anchor
    base = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = [base + pd.Timedelta(minutes=5*i) for i in range(n)]

    rows = []
    for ts in timestamps:
        hour = ts.hour
        dow  = ts.dayofweek

        # Rush-hour amplification
        rush = hour in (7, 8, 9, 16, 17, 18, 19)
        eve  = hour in (10, 11, 12, 13, 14, 15)
        base_count = rng.integers(80, 160) if rush else (
                     rng.integers(40, 100) if eve else rng.integers(10, 50))
        base_speed = rng.uniform(5, 25)   if rush else (
                     rng.uniform(25, 55)  if eve  else rng.uniform(45, 80))
        rain_mm    = float(rng.choice([0]*8 + [rng.uniform(0.5, 15)]))

        if rain_mm > 1:
            base_count = int(base_count * 1.15)
            base_speed = base_speed * 0.75

        ci = float(np.clip(base_count * 0.6 + max(0, 80 - base_speed), 0, 100))
        rows.append({
            'timestamp':       ts,
            'vehicle_count':   int(base_count),
            'avg_speed':       round(float(base_speed), 1),
            'congestion_index':round(ci, 1),
            'rainfall_mm':     round(rain_mm, 2),
        })
    return pd.DataFrame(rows)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    from ml.src.features import engineer_features, get_feature_cols, chronological_split

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                            'traffic_data.csv')
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        print(f"[data]  loaded {len(raw)} rows from CSV")
    else:
        print(f"[data]  no CSV found — generating {N_ROWS} synthetic rows")
        raw = _generate_synthetic(N_ROWS)

    df   = engineer_features(raw)
    feat = get_feature_cols()
    train_df, val_df, test_df = chronological_split(df)

    X_train = train_df[feat].values;  y_train = train_df['label'].values.astype(int)
    X_val   = val_df[feat].values;    y_val   = val_df['label'].values.astype(int)
    X_test  = test_df[feat].values;   y_test  = test_df['label'].values.astype(int)

    print(f"[split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
    print(f"[label] distribution: { {i: int((y_train==i).sum()) for i in range(4)} }")

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Combine train+val for final fit (common practice after hparam selection)
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    print("[train] fitting GradientBoostingClassifier …")
    model = GradientBoostingClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.08,
        subsample         = 0.8,
        min_samples_leaf  = 10,
        random_state      = 42,
        verbose           = 1,
    )
    model.fit(X_tv, y_tv)

    # Evaluate
    preds = model.predict(X_test)
    f1    = f1_score(y_test, preds, average='macro')
    print(classification_report(y_test, preds, target_names=LABEL_NAMES))
    print(f"[eval]  Macro F1 = {f1:.3f}")

    # Save — same filenames services.py expects
    clf_path    = os.path.join(SAVE_DIR, 'xgb_classifier.pkl')
    scaler_path = os.path.join(SAVE_DIR, 'xgb_scaler.pkl')
    joblib.dump(model,  clf_path)
    joblib.dump(scaler, scaler_path)
    print(f"[save]  {clf_path}")
    print(f"[save]  {scaler_path}")
    print("[done]  classifier ready")


if __name__ == '__main__':
    train()
