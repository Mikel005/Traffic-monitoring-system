"""
Train XGBoost congestion level classifier.
Run: python ml/src/train_xgb.py

Outputs:
  ml/saved_models/xgb_classifier.pkl
  ml/saved_models/xgb_scaler.pkl
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

SAVE_DIR    = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
LABEL_NAMES = ['free_flow', 'moderate', 'heavy', 'gridlock']

PARAMS = {
    'n_estimators':         500,
    'max_depth':            6,
    'learning_rate':        0.05,
    'subsample':            0.8,
    'colsample_bytree':     0.8,
    'use_label_encoder':    False,
    'eval_metric':          'mlogloss',
    'early_stopping_rounds': 20,
    'random_state':         42,
}


def train():
    from ml.src.features import (engineer_features, get_feature_cols,
                                  chronological_split, generate_synthetic)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'traffic_data.csv')
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(raw)} rows from CSV")
    else:
        logger.warning("No CSV found — using synthetic data")
        from ml.src.train_lstm import generate_synthetic
        raw = generate_synthetic(8000)

    df   = engineer_features(raw)
    feat = get_feature_cols()
    train_df, val_df, test_df = chronological_split(df)

    X_train = train_df[feat].values;  y_train = train_df['label'].values
    X_val   = val_df[feat].values;    y_val   = val_df['label'].values
    X_test  = test_df[feat].values;   y_test  = test_df['label'].values

    # Handle class imbalance with SMOTE (if available)
    try:
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        logger.info(f"SMOTE applied. Train size: {len(X_train)}")
    except ImportError:
        logger.warning("imbalanced-learn not installed — skipping SMOTE")

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Train
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate
    preds   = model.predict(X_test)
    f1      = f1_score(y_test, preds, average='macro')
    logger.info(f"\n{classification_report(y_test, preds, target_names=LABEL_NAMES)}")
    logger.info(f"✅ Macro F1 = {f1:.3f}")

    # Save
    joblib.dump(model,  os.path.join(SAVE_DIR, 'xgb_classifier.pkl'))
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'xgb_scaler.pkl'))
    logger.info(f"✅ XGBoost saved → {SAVE_DIR}/xgb_classifier.pkl")


if __name__ == '__main__':
    train()
