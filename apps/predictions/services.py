"""
Prediction service: loads ONNX LSTM and XGBoost models,
runs inference, falls back to rule-based estimate if models not trained yet.
"""
import numpy as np
from loguru import logger
from django.conf import settings

_lstm_session = None
_xgb_model    = None
_xgb_scaler   = None


def _load_lstm():
    global _lstm_session
    if _lstm_session is None:
        try:
            import onnxruntime as ort
            _lstm_session = ort.InferenceSession(str(settings.LSTM_MODEL_PATH))
            logger.info("✅ LSTM ONNX model loaded")
        except Exception as e:
            logger.warning(f"⚠️  LSTM not found ({e}) — using rule-based fallback")
    return _lstm_session


def _load_xgb():
    global _xgb_model, _xgb_scaler
    if _xgb_model is None:
        try:
            import joblib
            _xgb_model  = joblib.load(str(settings.XGB_MODEL_PATH))
            scaler_path = str(settings.XGB_MODEL_PATH).replace(
                'xgb_classifier.pkl', 'xgb_scaler.pkl')
            _xgb_scaler = joblib.load(scaler_path)
            logger.info("✅ XGBoost model loaded")
        except Exception as e:
            logger.warning(f"⚠️  XGBoost not found ({e})")
    return _xgb_model, _xgb_scaler


def _rule_based(current_index: float) -> dict:
    """Simple heuristic fallback before training is complete."""
    noise = lambda: float(np.random.normal(0, 3))
    clamp = lambda v: round(max(0.0, min(100.0, v)), 2)
    return {
        'pred_15min':     clamp(current_index + noise()),
        'pred_30min':     clamp(current_index + noise() * 1.5),
        'pred_60min':     clamp(current_index + noise() * 2),
        'model_version':  'rule_based',
        'confidence':     0.50,
    }


def predict(feature_sequence: np.ndarray, current_index: float) -> dict:
    """
    Run LSTM prediction.
    Args:
        feature_sequence: shape (1, 12, n_features) float32
        current_index: current congestion index (fallback use)
    Returns:
        dict with pred_15min, pred_30min, pred_60min, model_version, confidence
    """
    session = _load_lstm()
    if session is None:
        return _rule_based(current_index)
    try:
        inp   = session.get_inputs()[0].name
        preds = session.run(None, {inp: feature_sequence.astype(np.float32)})[0][0]
        return {
            'pred_15min':    round(float(np.clip(preds[0], 0, 100)), 2),
            'pred_30min':    round(float(np.clip(preds[1], 0, 100)), 2),
            'pred_60min':    round(float(np.clip(preds[2], 0, 100)), 2),
            'model_version': 'lstm_v1',
            'confidence':    0.88,
        }
    except Exception as e:
        logger.error(f"LSTM inference error: {e}")
        return _rule_based(current_index)
