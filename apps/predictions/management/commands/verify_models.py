"""
Management command: verify all ML models load correctly.
Usage: python manage.py verify_models
"""
from django.core.management.base import BaseCommand
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Verify that all ML model files exist and load correctly'

    def handle(self, *args, **options):
        self.stdout.write('\n=== ML Model Verification ===\n')
        all_ok = True

        # ── 1. Check file existence ────────────────────────────────
        model_files = {
            'LSTM (ONNX)':      settings.LSTM_MODEL_PATH,
            'XGBoost model':    settings.XGB_MODEL_PATH,
            'XGBoost scaler':   settings.BASE_DIR / 'ml/saved_models/xgb_scaler.pkl',
            'LSTM scaler':      settings.BASE_DIR / 'ml/saved_models/scaler.pkl',
            'YOLOv8 weights':   settings.YOLO_WEIGHTS_PATH,
        }

        for name, path in model_files.items():
            exists = os.path.isfile(path)
            size   = f'{os.path.getsize(path) / 1024:.1f} KB' if exists else 'N/A'
            status = self.style.SUCCESS('✅ FOUND') if exists else self.style.ERROR('❌ MISSING')
            self.stdout.write(f'  {status}  {name:<20} {size:<12} {path}')
            if not exists:
                all_ok = False

        self.stdout.write('')

        # ── 2. Try loading LSTM ────────────────────────────────────
        self.stdout.write('Loading LSTM (ONNX)...')
        try:
            import onnxruntime as ort
            import numpy as np
            session = ort.InferenceSession(str(settings.LSTM_MODEL_PATH))
            inp_name  = session.get_inputs()[0].name
            inp_shape = session.get_inputs()[0].shape
            out_shape = session.get_outputs()[0].shape
            # smoke test with zeros
            n_feat = inp_shape[2] if len(inp_shape) == 3 else 28
            dummy  = np.zeros((1, 12, n_feat), dtype=np.float32)
            result = session.run(None, {inp_name: dummy})[0]
            self.stdout.write(
                self.style.SUCCESS(
                    f'  ✅ LSTM loaded — input {inp_shape}, output {out_shape}, '
                    f'test preds: {result[0].tolist()}'
                )
            )
        except FileNotFoundError:
            self.stdout.write(self.style.WARNING('  ⚠️  LSTM file not found — will use rule-based fallback'))
            all_ok = False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ LSTM load error: {e}'))
            all_ok = False

        # ── 3. Try loading XGBoost ─────────────────────────────────
        self.stdout.write('Loading XGBoost...')
        try:
            import joblib
            import numpy as np
            model  = joblib.load(str(settings.XGB_MODEL_PATH))
            scaler = joblib.load(str(settings.BASE_DIR / 'ml/saved_models/xgb_scaler.pkl'))
            dummy  = np.zeros((1, 28))
            scaled = scaler.transform(dummy)
            pred   = model.predict(scaled)
            self.stdout.write(
                self.style.SUCCESS(
                    f'  ✅ XGBoost loaded — classes: {getattr(model, "n_classes_", "?")} '
                    f'estimators: {getattr(model, "n_estimators", "?")}, '
                    f'test pred: {pred.tolist()}'
                )
            )
        except FileNotFoundError:
            self.stdout.write(self.style.WARNING('  ⚠️  XGBoost file not found — classification unavailable'))
            all_ok = False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ XGBoost load error: {e}'))
            all_ok = False

        # ── 4. Try loading YOLOv8 ─────────────────────────────────
        self.stdout.write('Loading YOLOv8...')
        try:
            from ultralytics import YOLO
            model = YOLO(str(settings.YOLO_WEIGHTS_PATH))
            info  = model.info(verbose=False)
            self.stdout.write(self.style.SUCCESS(f'  ✅ YOLOv8 loaded — {settings.YOLO_WEIGHTS_PATH.name}'))
        except FileNotFoundError:
            self.stdout.write(self.style.WARNING('  ⚠️  YOLOv8 file not found — vision detection unavailable'))
            all_ok = False
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  ❌ YOLOv8 load error: {e}'))
            all_ok = False

        # ── Summary ───────────────────────────────────────────────
        self.stdout.write('')
        if all_ok:
            self.stdout.write(self.style.SUCCESS('All models verified successfully.\n'))
        else:
            self.stdout.write(
                self.style.WARNING(
                    'Some models are missing or failed to load.\n'
                    'Place your trained model files in: ml/saved_models/\n'
                    '  • lstm_best.onnx  — LSTM time-series model\n'
                    '  • xgb_classifier.pkl + xgb_scaler.pkl — XGBoost classifier\n'
                    '  • scaler.pkl      — LSTM feature scaler\n'
                    '  • yolov8n.pt      — YOLOv8 vehicle detection weights\n'
                )
            )
