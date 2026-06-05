"""
export_onnx.py — Export YOLOv8 to ONNX for faster CPU inference.

ONNX Runtime on CPU is typically 2-4× faster than PyTorch.
Run once after you have your trained .pt model:

    python ml/src/export_onnx.py

Or with custom paths:

    python ml/src/export_onnx.py --input ml/saved_models/yolov8n.pt --imgsz 320
"""

import argparse
import time
from pathlib import Path


def export(input_path: str, imgsz: int = 320, opset: int = 12):
    from ultralytics import YOLO
    import numpy as np

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")

    print(f"Loading {src.name} …")
    model = YOLO(str(src))

    print(f"Exporting to ONNX (imgsz={imgsz}, opset={opset}) …")
    t0   = time.perf_counter()
    out  = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        opset    = opset,
        simplify = True,
        dynamic  = False,
    )
    elapsed = time.perf_counter() - t0
    out_path = Path(out)
    print(f"Exported in {elapsed:.1f}s → {out_path}")

    # Quick speed comparison
    import numpy as np
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    print("\nBenchmarking .pt model …")
    for _ in range(3):            # discard cold starts
        model.predict(dummy, verbose=False, imgsz=imgsz)
    times_pt = []
    for _ in range(8):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, imgsz=imgsz, conf=0.4, max_det=50)
        times_pt.append((time.perf_counter() - t0) * 1000)
    print(f"  .pt   avg {sum(times_pt)/len(times_pt):.1f}ms  min {min(times_pt):.1f}ms")

    print("Benchmarking .onnx model …")
    onnx_model = YOLO(str(out_path))
    for _ in range(3):
        onnx_model.predict(dummy, verbose=False, imgsz=imgsz)
    times_onnx = []
    for _ in range(8):
        t0 = time.perf_counter()
        onnx_model.predict(dummy, verbose=False, imgsz=imgsz, conf=0.4, max_det=50)
        times_onnx.append((time.perf_counter() - t0) * 1000)
    print(f"  .onnx avg {sum(times_onnx)/len(times_onnx):.1f}ms  min {min(times_onnx):.1f}ms")

    speedup = sum(times_pt)/len(times_pt) / (sum(times_onnx)/len(times_onnx))
    print(f"\nSpeedup: {speedup:.2f}×")
    print(f"\nTo use the ONNX model, set in .env:")
    print(f"  YOLO_WEIGHTS_PATH={out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt → .onnx")
    parser.add_argument("--input",  default="ml/saved_models/yolov8n.pt")
    parser.add_argument("--imgsz",  type=int, default=320)
    parser.add_argument("--opset",  type=int, default=12)
    args = parser.parse_args()
    export(args.input, args.imgsz, args.opset)
