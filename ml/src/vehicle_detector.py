"""
vehicle_detector.py — YOLOv8 + ByteTrack vehicle detection pipeline.

Detection & Tracking
────────────────────
  .pt  model  → model.track(persist=True, tracker="bytetrack.yaml")
                Ultralytics built-in ByteTrack (best accuracy + speed)
  .torchscript
  .onnx model → model.predict() + ByteTracker (ml/src/byte_tracker.py)
                Kalman-filter two-stage IoU association, CPU-friendly

Counting
────────
  Virtual horizontal line at 60% of frame height.
  Each unique track_id is counted once (set-based, no double-counting).
  Direction detected: INBOUND  (top→bottom, y increasing)
                      OUTBOUND (bottom→top, y decreasing)

CSV Export
──────────
  Every vehicle crossing is logged to media/counts/<session>.csv
  Summary row appended on completion.

Architecture
────────────
  stream_inference()
    ├── FrameReader thread    — I/O decoupled from inference
    ├── InferenceWorker thread — YOLO + tracker + counting
    └── AsyncDBWriter thread  — all DB/CSV writes off the hot path

  process_video()  — non-streaming background batch (same logic)
"""

from __future__ import annotations

import csv
import math
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch    = None
    HAS_CUDA = False

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ml.src.byte_tracker import ByteTracker, STrack, iou_batch

# ── Constants ─────────────────────────────────────────────────────────────────

VEHICLE_CLASS_MAP = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES   = list(VEHICLE_CLASS_MAP.keys())
LEVEL_MAP = {
    "FREE FLOW": "free_flow", "MODERATE": "moderate",
    "HEAVY":     "heavy",     "GRIDLOCK": "gridlock",
}

STRIDE       = 2      # process every Nth frame
IMGSZ        = 320    # YOLO input resolution
CONF_THRESH  = 0.40
MAX_DET      = 50
STREAM_WIDTH = 640
JPEG_QUALITY = 72
DB_SAVE_EVERY  = 90   # frames between DB saves
DB_SYNC_EVERY  = 150  # frames between TrafficReading creates
XGB_CACHE_SEC  = 2.0
LINE_RATIO     = 0.60  # counting line at 60 % of frame height
CROSS_MARGIN   = 20    # px tolerance around counting line

_ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

COUNTS_DIR = Path("media") / "counts"


def _model_supports_track(path: str) -> bool:
    return str(path).lower().endswith(".pt")


# ── CSV helper ────────────────────────────────────────────────────────────────

def _csv_path(session_tag: str) -> Path:
    COUNTS_DIR.mkdir(parents=True, exist_ok=True)
    return COUNTS_DIR / f"{session_tag}.csv"


def _append_csv(path: Path, row: dict):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


# ── Background threads ────────────────────────────────────────────────────────

class FrameReader(threading.Thread):
    def __init__(self, source, maxsize: int = 2):
        super().__init__(daemon=True)
        self.cap    = cv2.VideoCapture(source)
        self._q     = queue.Queue(maxsize=maxsize)
        self._stop  = threading.Event()
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)              or 25.0
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))or 480

    def run(self):
        while not self._stop.is_set() and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)
        self._q.put(None)
        self.cap.release()

    def read(self, timeout: float = 2.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


class AsyncDBWriter(threading.Thread):
    """Executes callable tasks off the inference hot-path with retry on lock."""

    def __init__(self):
        super().__init__(daemon=True)
        self._q = queue.Queue()
        self.start()

    def submit(self, fn, *args, **kwargs):
        self._q.put((fn, args, kwargs))

    def stop(self):
        self._q.put(None)

    def run(self):
        from django.db import close_old_connections
        from django.db.utils import OperationalError
        while True:
            item = self._q.get()
            if item is None:
                break
            fn, args, kwargs = item
            for attempt in range(5):
                try:
                    fn(*args, **kwargs)
                    break
                except OperationalError as e:
                    if "locked" in str(e).lower():
                        time.sleep(0.05 * (2 ** attempt))
                    else:
                        break
                except Exception:
                    break
            try:
                close_old_connections()
            except Exception:
                pass


# ── Inference worker ──────────────────────────────────────────────────────────

class InferenceWorker(threading.Thread):
    """
    Runs YOLO detection + ByteTrack tracking in a background thread.
    Accepts frames via submit(); exposes latest results via .results.

    Counting logic
    ──────────────
    Each unique track_id is counted exactly once when the bbox bottom-centre
    crosses the virtual line (within CROSS_MARGIN pixels).
    Direction is determined from the track's previous y-position:
      - was_above=True  → now crosses line → INBOUND  (moving downward)
      - was_above=False → now crosses line → OUTBOUND (moving upward)
    """

    def __init__(self, detector: "VehicleDetector",
                 speed_history: dict, counted_ids: set,
                 cc: dict, inbound: list, outbound: list,
                 total_speed: list, speed_count: list,
                 fps: float, use_native_track: bool,
                 csv_path: Optional[Path] = None):
        super().__init__(daemon=True)
        self.det             = detector
        self.speed_history   = speed_history
        self.counted_ids     = counted_ids
        self.cc              = cc
        self.inbound         = inbound    # [count]
        self.outbound        = outbound   # [count]
        self.total_speed     = total_speed
        self.speed_count     = speed_count
        self.fps             = fps
        self.use_native_track= use_native_track
        self.csv_path        = csv_path

        self._in_q   = queue.Queue(maxsize=1)
        self._stop   = threading.Event()
        self._bt     = ByteTracker(high_thresh=0.50, low_thresh=0.10,
                                    iou_thresh=0.30, max_age=30, min_hits=1)
        # track_id → 'above' or 'below' (relative to counting line)
        self._prev_side: Dict[int, str] = {}

        self.results = {
            "boxes": [], "n": 0, "level": "FREE FLOW",
            "avg_speed": 0.0, "queue_len": 0.0, "ready": False,
        }

    def submit(self, frame: np.ndarray, w: int, h: int, line_y: int):
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        self._in_q.put((frame, w, h, line_y))

    def stop(self):
        self._stop.set()
        self._in_q.put(None)

    def run(self):
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._process(*item)
            except Exception as exc:
                print(f"[InferenceWorker] {exc}")
                time.sleep(0.05)

    def _process(self, frame: np.ndarray, w: int, h: int, line_y: int):
        model = self.det.model
        mid   = w / 2.0

        # ── Detection + tracking ───────────────────────────────────────
        if self.use_native_track:
            results = model.track(
                frame, classes=self.det.vehicle_classes,
                persist=True, verbose=False,
                imgsz=IMGSZ, conf=CONF_THRESH, max_det=MAX_DET,
                half=HAS_CUDA, tracker="bytetrack.yaml",
            )
            res   = results[0]
            track_list: List[Tuple] = []
            if res.boxes is not None and len(res.boxes):
                boxes  = res.boxes.xyxy.cpu().numpy().astype(int)
                cids   = res.boxes.cls.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                tids   = (res.boxes.id.cpu().numpy().astype(int)
                          if res.boxes.id is not None
                          else np.arange(len(boxes)))
                for box, tid, cid, sc in zip(boxes, tids, cids, scores):
                    track_list.append((tuple(box.tolist()), int(tid), int(cid), float(sc)))
        else:
            results = model.predict(
                frame, classes=self.det.vehicle_classes,
                verbose=False, imgsz=IMGSZ, conf=CONF_THRESH,
                max_det=MAX_DET, half=False,
            )
            res = results[0]
            dets: List[Tuple] = []
            if res.boxes is not None and len(res.boxes):
                boxes  = res.boxes.xyxy.cpu().numpy()
                cids   = res.boxes.cls.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                for box, cid, sc in zip(boxes, cids, scores):
                    dets.append((box.astype(int), float(sc), int(cid)))

            bt_boxes  = np.array([d[0] for d in dets], dtype=np.float32) if dets else np.empty((0, 4))
            bt_scores = np.array([d[1] for d in dets], dtype=np.float32) if dets else np.empty((0,))
            bt_cids   = np.array([d[2] for d in dets], dtype=np.int32)   if dets else np.empty((0,), dtype=np.int32)
            active    = self._bt.update(bt_boxes, bt_scores, bt_cids)
            track_list = [
                (tuple(t.tlbr.astype(int).tolist()), t.track_id, t.cls_id, t.score)
                for t in active
            ]

        # ── Per-track processing ───────────────────────────────────────
        boxes_out  = []
        bev_pts    = []

        for (x1, y1, x2, y2), tid, cid, score in track_list:
            cx, cy = (x1 + x2) // 2, y2        # bottom-centre
            M      = self.det.ML if cx < mid else self.det.MR
            bev_pt = self.det._to_bev(M, cx, cy)
            bev_pts.append(bev_pt)

            # Speed via BEV history
            if tid not in self.speed_history:
                self.speed_history[tid] = deque(maxlen=12)
            self.speed_history[tid].append(bev_pt)
            kph = 0.0
            hist = self.speed_history[tid]
            if len(hist) >= 3:
                dx     = hist[-1][0] - hist[0][0]
                dy     = hist[-1][1] - hist[0][1]
                dist_m = math.hypot(dx, dy) / self.det.BEV_SCALE
                dt_s   = (len(hist) - 1) * STRIDE / self.fps
                kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                if 1.0 < kph < 200.0:
                    self.total_speed[0] += kph
                    self.speed_count[0] += 1

            # Direction + counting line crossing
            cy_centre = (y1 + y2) / 2.0
            side      = "above" if cy_centre < line_y else "below"
            prev_side = self._prev_side.get(tid)

            if prev_side and prev_side != side and tid not in self.counted_ids:
                # Vehicle just crossed the line
                self.counted_ids.add(tid)
                lbl = VEHICLE_CLASS_MAP.get(cid, "car")
                self.cc[lbl] = self.cc.get(lbl, 0) + 1

                direction = "INBOUND" if prev_side == "above" else "OUTBOUND"
                if direction == "INBOUND":
                    self.inbound[0]  += 1
                else:
                    self.outbound[0] += 1

                # Log crossing to CSV
                if self.csv_path:
                    _append_csv(self.csv_path, {
                        "timestamp":    datetime.now().isoformat(timespec="seconds"),
                        "track_id":     tid,
                        "vehicle_type": lbl,
                        "direction":    direction,
                        "speed_kph":    round(kph, 1),
                    })

            self._prev_side[tid] = side
            boxes_out.append({
                "box": (x1, y1, x2, y2), "tid": tid,
                "cid": cid, "kph": kph, "side": side,
            })

        n_on_screen = len(track_list)
        avg_speed   = (self.total_speed[0] / self.speed_count[0]
                       if self.speed_count[0] else 0.0)
        queue_len   = self.det._queue_length(bev_pts)
        level       = self.det._xgb_classify(
            n_on_screen, avg_speed, min(n_on_screen * 2, 100)
        )

        self.results = {
            "boxes":     boxes_out,
            "n":         n_on_screen,
            "level":     level,
            "avg_speed": avg_speed,
            "queue_len": queue_len,
            "ready":     True,
        }


# ── VehicleDetector ───────────────────────────────────────────────────────────

class VehicleDetector:

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path       = str(model_path)
        self.use_native_track = _model_supports_track(self.model_path)
        self.model            = YOLO(self.model_path) if YOLO else None
        if self.model and HAS_CUDA:
            self.model.to("cuda")
        self.vehicle_classes = VEHICLE_CLASSES

        # Bird's-Eye-View calibration
        self.BEV_SCALE        = 18
        self.VISIBLE_LENGTH_M = 60
        self.SRC_ROAD_L = np.float32([[130,390],[415,390],[680,720],[-70,720]])
        self.SRC_ROAD_R = np.float32([[415,390],[610,390],[960,720],[680,720]])
        self.ML, self.bev_wL, _ = self._bev_matrix(self.SRC_ROAD_L, 3.75 * 5)
        self.MR, self.bev_wR, _ = self._bev_matrix(self.SRC_ROAD_R, 3.75 * 3)

        self._xgb_last_run: float = 0.0
        self._xgb_cached:   str   = ""

    # ── BEV ───────────────────────────────────────────────────────────

    def _bev_matrix(self, src, road_width_m):
        w   = int(road_width_m * self.BEV_SCALE)
        h   = int(self.VISIBLE_LENGTH_M * self.BEV_SCALE)
        dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
        return cv2.getPerspectiveTransform(src, dst), w, h

    def _to_bev(self, M, cx, cy):
        pt = np.float32([[[cx, cy]]])
        t  = cv2.perspectiveTransform(pt, M)
        return float(t[0,0,0]), float(t[0,0,1])

    def _queue_length(self, bev_pts) -> float:
        if len(bev_pts) < 2:
            return 0.0
        ys = [p[1] for p in bev_pts]
        return round((max(ys) - min(ys)) / self.BEV_SCALE, 1)

    # ── Congestion ────────────────────────────────────────────────────

    def _xgb_classify(self, count: int, speed: float, idx: int) -> str:
        now = time.monotonic()
        if now - self._xgb_last_run < XGB_CACHE_SEC and self._xgb_cached:
            return self._xgb_cached
        try:
            from apps.predictions.services import classify
            dt    = datetime.now()
            feats = np.zeros((1, 28), dtype=np.float32)
            feats[0, 0] = count;  feats[0, 1] = speed; feats[0, 2] = idx
            feats[0, 3] = math.sin(2*math.pi*dt.hour/24)
            feats[0, 4] = math.cos(2*math.pi*dt.hour/24)
            result = classify(feats)
            if result.get("label") is not None:
                label_map = {0:"FREE FLOW",1:"MODERATE",2:"HEAVY",3:"GRIDLOCK"}
                self._xgb_cached   = label_map[result["label"]]
                self._xgb_last_run = now
                return self._xgb_cached
        except Exception:
            pass
        if   count > 20: level = "GRIDLOCK"
        elif count > 12: level = "HEAVY"
        elif count > 5:  level = "MODERATE"
        else:            level = "FREE FLOW"
        self._xgb_cached   = level
        self._xgb_last_run = now
        return level

    # ── Drawing ───────────────────────────────────────────────────────

    @staticmethod
    def _speed_color(kph: float):
        if kph < 40:  return (0, 220,   0)
        if kph < 100: return (0, 220, 220)
        return (0, 60, 255)

    def _draw_hud(self, frame, total_v, speed, level, cc, ib, ob, queue_len):
        colour = {
            "FREE FLOW":(0,220,0), "MODERATE":(0,220,220),
            "HEAVY":(0,140,255),   "GRIDLOCK":(0,50,255),
        }.get(level, (200,200,200))
        cv2.rectangle(frame, (8,8),   (430,175), (0,0,0), -1)
        cv2.rectangle(frame, (8,8),   (430,175), (60,60,60), 1)
        cv2.putText(frame, f"Vehicles : {total_v}",
                    (18, 38),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Speed    : {speed:.1f} km/h",
                    (18, 68),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,220,0),    2, cv2.LINE_AA)
        cv2.putText(frame, f"Status   : {level}",
                    (18, 98),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, colour,       2, cv2.LINE_AA)
        cv2.putText(frame, f"IN:{ib}  OUT:{ob}  Q:{queue_len}m",
                    (18,128),  cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180,180,255),2, cv2.LINE_AA)
        detail = (f"C:{cc.get('car',0)} T:{cc.get('truck',0)} "
                  f"B:{cc.get('bus',0)} M:{cc.get('motorcycle',0)}")
        cv2.putText(frame, detail,
                    (18,158),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160),1, cv2.LINE_AA)

    # ── DB helpers ────────────────────────────────────────────────────

    def _save_video_obj(self, video_obj, vehicles, speed, level, cc,
                        inbound, outbound, status=None):
        video_obj.vehicle_count              = vehicles
        video_obj.average_speed              = round(speed, 1)
        video_obj.predicted_congestion_level = level
        video_obj.car_count                  = cc.get("car", 0)
        video_obj.truck_count                = cc.get("truck", 0)
        video_obj.motorcycle_count           = cc.get("motorcycle", 0)
        video_obj.bus_count                  = cc.get("bus", 0)
        fields = ["vehicle_count","average_speed","predicted_congestion_level",
                  "car_count","truck_count","motorcycle_count","bus_count"]
        if status:
            video_obj.status = status
            fields.append("status")
        video_obj.save(update_fields=fields)

    def _sync_to_traffic(self, video_obj, vehicles, speed, level, cc, queue_len):
        from apps.traffic.models import TrafficReading, CongestionLevel
        from apps.alerts.services import NotificationService
        if not getattr(video_obj, "location", None):
            return
        db_level = CongestionLevel(LEVEL_MAP.get(level, "free_flow"))
        reading  = TrafficReading.objects.create(
            location         = video_obj.location,
            vehicle_count    = vehicles,
            car_count        = cc.get("car", 0),
            truck_count      = cc.get("truck", 0),
            motorcycle_count = cc.get("motorcycle", 0),
            bus_count        = cc.get("bus", 0),
            avg_speed        = speed,
            queue_length     = queue_len,
            congestion_index = min(vehicles * 2, 100),
            congestion_level = db_level,
            source           = "vision",
        )
        if db_level in (CongestionLevel.HEAVY, CongestionLevel.GRIDLOCK):
            try:
                NotificationService.send_alert(reading)
            except Exception:
                pass

    def _save_session(self, session_id, video_obj, cc, inbound, outbound,
                      avg_speed, level, csv_path: Optional[Path]):
        """Create/update VehicleCountSession in the DB."""
        try:
            from apps.vision.models import VehicleCountSession
            from django.utils import timezone
            total_v = sum(cc.values())
            session, _ = VehicleCountSession.objects.update_or_create(
                session_tag=session_id,
                defaults=dict(
                    location         = getattr(video_obj, "location", None),
                    total_count      = total_v,
                    car_count        = cc.get("car", 0),
                    truck_count      = cc.get("truck", 0),
                    bus_count        = cc.get("bus", 0),
                    motorcycle_count = cc.get("motorcycle", 0),
                    inbound_count    = inbound,
                    outbound_count   = outbound,
                    avg_speed        = round(avg_speed, 1),
                    peak_congestion  = level,
                    ended_at         = timezone.now(),
                ),
            )
            if csv_path and csv_path.exists():
                rel = str(csv_path.relative_to(Path("media")))
                session.csv_file.name = rel
                session.save(update_fields=["csv_file"])
        except Exception:
            pass

    # ── Public: MJPEG streaming ───────────────────────────────────────

    def stream_inference(self, video_source, output_video_obj=None):
        """
        Yield MJPEG byte-frames with YOLO + ByteTrack overlays.
        Suitable for Django StreamingHttpResponse.
        """
        if not self.model:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
            return

        session_tag = f"stream_{int(time.time())}"
        csv_path    = _csv_path(session_tag)

        reader    = FrameReader(video_source)
        reader.start()
        db_writer = AsyncDBWriter() if output_video_obj else None

        fps    = reader.fps
        w      = STREAM_WIDTH
        h      = int(reader.height * (STREAM_WIDTH / reader.width))
        line_y = int(h * LINE_RATIO)

        frame_n      = 0
        counted_ids  = set()
        speed_history= {}
        cc           = {}
        inbound      = [0]
        outbound     = [0]
        total_speed  = [0.0]
        speed_count  = [0]
        last_db_save = 0
        target_delay = 1.0 / fps

        worker = InferenceWorker(
            self, speed_history, counted_ids, cc, inbound, outbound,
            total_speed, speed_count, fps,
            use_native_track=self.use_native_track,
            csv_path=csv_path,
        )
        worker.start()

        while True:
            t0    = time.time()
            frame = reader.read(timeout=2.0)
            if frame is None:
                break

            frame_n   += 1
            disp      = cv2.resize(frame, (w, h))

            if frame_n % STRIDE == 0:
                worker.submit(disp.copy(), w, h, line_y)

            res = worker.results
            if res["ready"]:
                # Counting line
                cv2.line(disp, (0, line_y), (w, line_y), (0, 200, 255), 2)
                cv2.putText(disp, "COUNTING LINE", (w//2 - 70, line_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1, cv2.LINE_AA)

                for b in res["boxes"]:
                    x1, y1, x2, y2 = b["box"]
                    color  = self._speed_color(b["kph"])
                    label  = (f"#{b['tid']} "
                              f"{VEHICLE_CLASS_MAP.get(b['cid'],'?')} "
                              f"{b['kph']:.0f}km/h")
                    cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(disp, label, (x1, max(y1-8,12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

                total_v = sum(cc.values())
                self._draw_hud(disp, total_v, res["avg_speed"], res["level"],
                               cc, inbound[0], outbound[0], res["queue_len"])

                if output_video_obj and db_writer:
                    if frame_n - last_db_save >= DB_SAVE_EVERY:
                        last_db_save = frame_n
                        db_writer.submit(self._save_video_obj, output_video_obj,
                                         total_v, res["avg_speed"], res["level"],
                                         cc.copy(), inbound[0], outbound[0])
                    if frame_n % DB_SYNC_EVERY == 0:
                        db_writer.submit(self._sync_to_traffic, output_video_obj,
                                         total_v, res["avg_speed"], res["level"],
                                         cc.copy(), res["queue_len"])

            ok, buf = cv2.imencode(".jpg", disp, _ENCODE_PARAM)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

            elapsed = time.time() - t0
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

        worker.stop()
        reader.stop()

        if output_video_obj and db_writer:
            res     = worker.results
            total_v = sum(cc.values())
            db_writer.submit(self._save_video_obj, output_video_obj,
                             total_v, res["avg_speed"], res["level"],
                             cc.copy(), inbound[0], outbound[0], status="completed")
            if getattr(output_video_obj, "location", None):
                db_writer.submit(self._sync_to_traffic, output_video_obj,
                                 total_v, res["avg_speed"], res["level"],
                                 cc.copy(), res["queue_len"])
            db_writer.submit(self._save_session, session_tag, output_video_obj,
                             cc.copy(), inbound[0], outbound[0],
                             res["avg_speed"], res["level"], csv_path)
            time.sleep(0.5)
            db_writer.stop()

    # ── Public: background batch ──────────────────────────────────────

    def process_video(self, video_path: str, video_obj=None):
        """Non-streaming batch processing for the background upload thread."""
        if not self.model:
            if video_obj:
                video_obj.status = "failed"
                video_obj.save(update_fields=["status"])
            return

        session_tag = f"batch_{int(time.time())}"
        csv_path    = _csv_path(session_tag)

        db_writer    = AsyncDBWriter()
        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)               or 25.0
        orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        proc_w       = 640
        proc_h       = int(orig_h * (proc_w / orig_w))
        line_y       = int(proc_h * LINE_RATIO)
        mid          = proc_w / 2.0

        frame_n      = 0
        counted_ids  = set()
        speed_history= {}
        cc: dict     = {}
        inbound      = 0
        outbound     = 0
        total_speed  = 0.0
        speed_count  = 0
        level        = "FREE FLOW"
        queue_len    = 0.0
        bt           = ByteTracker(high_thresh=0.50, low_thresh=0.10,
                                    iou_thresh=0.30, max_age=30, min_hits=1)
        prev_side: Dict[int, str] = {}

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1
            if frame_n % STRIDE != 0:
                continue

            small = cv2.resize(frame, (proc_w, proc_h))

            # Detect
            if self.use_native_track:
                results = self.model.track(
                    small, classes=self.vehicle_classes, persist=True,
                    verbose=False, imgsz=IMGSZ, conf=CONF_THRESH,
                    max_det=MAX_DET, half=HAS_CUDA, tracker="bytetrack.yaml",
                )
                res = results[0]
                tracks = []
                if res.boxes is not None and len(res.boxes):
                    boxes  = res.boxes.xyxy.cpu().numpy().astype(int)
                    cids   = res.boxes.cls.cpu().numpy().astype(int)
                    scores = res.boxes.conf.cpu().numpy()
                    tids   = (res.boxes.id.cpu().numpy().astype(int)
                              if res.boxes.id is not None
                              else np.arange(len(boxes)))
                    for box, tid, cid, sc in zip(boxes, tids, cids, scores):
                        tracks.append((tuple(box.tolist()), int(tid), int(cid)))
            else:
                results = self.model.predict(
                    small, classes=self.vehicle_classes, verbose=False,
                    imgsz=IMGSZ, conf=CONF_THRESH, max_det=MAX_DET, half=False,
                )
                res  = results[0]
                dets = []
                if res.boxes is not None and len(res.boxes):
                    boxes  = res.boxes.xyxy.cpu().numpy()
                    cids   = res.boxes.cls.cpu().numpy().astype(int)
                    scores = res.boxes.conf.cpu().numpy()
                    for box, cid, sc in zip(boxes, cids, scores):
                        dets.append((box, float(sc), int(cid)))
                bt_boxes  = np.array([d[0] for d in dets], dtype=np.float32) if dets else np.empty((0,4))
                bt_scores = np.array([d[1] for d in dets], dtype=np.float32) if dets else np.empty((0,))
                bt_cids   = np.array([d[2] for d in dets], dtype=np.int32)   if dets else np.empty((0,),dtype=np.int32)
                active    = bt.update(bt_boxes, bt_scores, bt_cids)
                tracks    = [(tuple(t.tlbr.astype(int).tolist()), t.track_id, t.cls_id) for t in active]

            # Process tracks
            bev_pts = []
            for (x1, y1, x2, y2), tid, cid in tracks:
                cx, cy = (x1+x2)//2, y2
                M      = self.ML if cx < mid else self.MR
                bev_pt = self._to_bev(M, cx, cy)
                bev_pts.append(bev_pt)

                if tid not in speed_history:
                    speed_history[tid] = deque(maxlen=12)
                speed_history[tid].append(bev_pt)

                kph = 0.0
                hist = speed_history[tid]
                if len(hist) >= 3:
                    dx     = hist[-1][0] - hist[0][0]
                    dy     = hist[-1][1] - hist[0][1]
                    dist_m = math.hypot(dx, dy) / self.BEV_SCALE
                    dt_s   = (len(hist) - 1) * STRIDE / fps
                    kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                    if 1.0 < kph < 200.0:
                        total_speed += kph
                        speed_count += 1

                # Direction detection
                cy_centre = (y1 + y2) / 2.0
                side      = "above" if cy_centre < line_y else "below"
                p_side    = prev_side.get(tid)
                if p_side and p_side != side and tid not in counted_ids:
                    counted_ids.add(tid)
                    lbl = VEHICLE_CLASS_MAP.get(cid, "car")
                    cc[lbl] = cc.get(lbl, 0) + 1
                    direction = "INBOUND" if p_side == "above" else "OUTBOUND"
                    if direction == "INBOUND":
                        inbound  += 1
                    else:
                        outbound += 1
                    _append_csv(csv_path, {
                        "timestamp":    datetime.now().isoformat(timespec="seconds"),
                        "track_id":     tid,
                        "vehicle_type": lbl,
                        "direction":    direction,
                        "speed_kph":    round(kph, 1),
                    })
                prev_side[tid] = side

            n_on_screen = len(tracks)
            avg_speed   = total_speed / speed_count if speed_count else 0.0
            queue_len   = self._queue_length(bev_pts)
            level       = self._xgb_classify(n_on_screen, avg_speed, min(n_on_screen*2, 100))
            total_v     = sum(cc.values())

            if video_obj and frame_n % DB_SAVE_EVERY == 0:
                db_writer.submit(self._save_video_obj, video_obj,
                                 total_v, avg_speed, level, cc.copy(),
                                 inbound, outbound)
            if video_obj and frame_n % DB_SYNC_EVERY == 0 and video_obj.location:
                db_writer.submit(self._sync_to_traffic, video_obj,
                                 total_v, avg_speed, level, cc.copy(), queue_len)

        cap.release()

        avg_speed = total_speed / speed_count if speed_count else 0.0
        total_v   = sum(cc.values())

        if video_obj:
            db_writer.submit(self._save_video_obj, video_obj,
                             total_v, avg_speed, level, cc.copy(),
                             inbound, outbound, status="completed")
            if video_obj.location:
                db_writer.submit(self._sync_to_traffic, video_obj,
                                 total_v, avg_speed, level, cc.copy(), queue_len)
            db_writer.submit(self._save_session, session_tag, video_obj,
                             cc.copy(), inbound, outbound,
                             avg_speed, level, csv_path)
            time.sleep(0.6)
        db_writer.stop()

        return {
            "total": total_v, "car": cc.get("car",0),
            "truck": cc.get("truck",0), "bus": cc.get("bus",0),
            "motorcycle": cc.get("motorcycle",0),
            "inbound": inbound, "outbound": outbound,
            "avg_speed": avg_speed, "level": level,
            "csv_path": str(csv_path) if csv_path.exists() else None,
        }
