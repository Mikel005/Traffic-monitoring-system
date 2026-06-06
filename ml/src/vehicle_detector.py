"""
vehicle_detector.py  —  YOLOv8 + SORT vehicle detection & counting pipeline.

Design principles
─────────────────
1. Thread safety  — InferenceWorker owns ALL mutable counting state.
                    Results are published as immutable snapshots protected
                    by a threading.Lock.  The MJPEG loop only reads snapshots.
                    No shared mutable containers between threads.

2. Single responsibility  — detection, tracking, counting are separated.
   _detect()      →  YOLO raw boxes (no tracker state)
   SortTracker    →  Kalman + Hungarian assignment (sort_tracker.py)
   InferenceWorker._tally() →  counting + speed (uses tracker output)

3. Consistent code paths  — streaming and batch use the same _detect()
                            and the same SortTracker logic.

4. model.predict() always  — works with .pt, .onnx, .torchscript.
                             Tracking is handled by SortTracker, not YOLO.

Public API (unchanged — views.py does not need editing)
──────────
    d = VehicleDetector('yolov8n.pt')

    # MJPEG streaming (Django StreamingHttpResponse)
    yield from d.stream_inference(video_path_or_rtsp_url, video_obj)

    # Background batch processing
    result = d.process_video(video_path, video_obj)
    # result: {'total', 'car', 'truck', 'bus', 'motorcycle',
    #          'inbound', 'outbound', 'avg_speed', 'level', 'csv_path'}
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
    if not HAS_CUDA:
        torch.set_num_threads(os.cpu_count() or 4)
        torch.set_num_interop_threads(max(2, (os.cpu_count() or 4) // 2))
except ImportError:
    torch    = None
    HAS_CUDA = False

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ml.src.sort_tracker import SortTracker, Track

# ── Vehicle classes (COCO) ────────────────────────────────────────────────────
CLASS_IDS  = [2, 3, 5, 7]                 # car, motorcycle, bus, truck
CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

LEVEL_MAP = {
    'FREE FLOW': 'free_flow', 'MODERATE': 'moderate',
    'HEAVY':     'heavy',     'GRIDLOCK': 'gridlock',
}

# ── Tunable constants ─────────────────────────────────────────────────────────
# Inference
IMGSZ_STREAM   = 192    # resolution sent to YOLO for streaming frames
IMGSZ_BATCH    = 320    # resolution for batch (background) processing
STRIDE_STREAM  = 2      # run YOLO every N frames in streaming mode
STRIDE_BATCH   = 3      # run YOLO every N frames in batch mode

# Detection quality
CONF_THRESH       = 0.40  # YOLO detection confidence floor
COUNT_CONF_THRESH = 0.60  # track must have this score to count at the line
NMS_IOU_THRESH    = 0.40  # NMS IoU — lower removes more duplicates

# Temporal smoothing — track must survive this many frames before counting
MIN_TRACK_HITS = 3   # built into SortTracker(min_hits=3)

# Region-of-interest — black out top fraction to exclude sky/buildings
ROI_SKIP_TOP = 0.20

# Vehicle size bounds (pixels)
MIN_BOX_AREA       = 400    # 20×20 minimum — filters noise
MAX_BOX_AREA_RATIO = 0.25   # reject boxes > 25 % of frame area

# Counting line
LINE_RATIO = 0.60    # line at 60 % of frame height

# MJPEG output
STREAM_WIDTH  = 640
JPEG_QUALITY  = 65
_ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

# DB write intervals
DB_SAVE_EVERY = 90    # frames
DB_SYNC_EVERY = 150   # frames

# XGBoost congestion cache
XGB_CACHE_SEC = 2.0

# CSV output directory
CSV_DIR = Path('media') / 'counts'


# ── Pure helpers (no state) ───────────────────────────────────────────────────

def _apply_roi(frame: np.ndarray) -> np.ndarray:
    """Black out the top ROI_SKIP_TOP fraction of the frame."""
    if ROI_SKIP_TOP <= 0:
        return frame
    out     = frame.copy()
    cut     = int(frame.shape[0] * ROI_SKIP_TOP)
    out[:cut] = 0
    return out


def _size_ok(x1: int, y1: int, x2: int, y2: int,
             frame_w: int, frame_h: int) -> bool:
    """True if the box area is within the expected vehicle size range."""
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return MIN_BOX_AREA <= area <= frame_w * frame_h * MAX_BOX_AREA_RATIO


def _queue_length(bev_pts: list, bev_scale: float) -> float:
    if len(bev_pts) < 2:
        return 0.0
    ys = [p[1] for p in bev_pts]
    return round((max(ys) - min(ys)) / bev_scale, 1)


def _classify_congestion(n: int, speed: float, xgb_cache: dict) -> str:
    """Classify congestion level. XGBoost if available, heuristic fallback."""
    now = time.monotonic()
    if now - xgb_cache.get('ts', 0) < XGB_CACHE_SEC:
        return xgb_cache.get('level', 'FREE FLOW')
    try:
        from apps.predictions.services import classify
        dt    = datetime.now()
        feats = np.zeros((1, 28), dtype=np.float32)
        feats[0, 0] = n;      feats[0, 1] = speed
        feats[0, 2] = min(n * 2, 100)
        feats[0, 3] = math.sin(2 * math.pi * dt.hour / 24)
        feats[0, 4] = math.cos(2 * math.pi * dt.hour / 24)
        result = classify(feats)
        if result.get('label') is not None:
            lm = {0: 'FREE FLOW', 1: 'MODERATE', 2: 'HEAVY', 3: 'GRIDLOCK'}
            level = lm[result['label']]
            xgb_cache.update({'level': level, 'ts': now})
            return level
    except Exception:
        pass
    if   n > 20: level = 'GRIDLOCK'
    elif n > 12: level = 'HEAVY'
    elif n > 5:  level = 'MODERATE'
    else:        level = 'FREE FLOW'
    xgb_cache.update({'level': level, 'ts': now})
    return level


def _append_csv(path: Path, row: dict):
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _csv_for(tag: str) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    return CSV_DIR / f'{tag}.csv'


# ── Thread: frame reading ─────────────────────────────────────────────────────

class FrameReader(threading.Thread):
    """Reads any OpenCV source in a thread; keeps only the freshest frame."""

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
                try: self._q.get_nowait()
                except queue.Empty: pass
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


# ── Thread: async DB / CSV writer ────────────────────────────────────────────

class AsyncDBWriter(threading.Thread):
    """
    Executes DB/CSV writes in a dedicated thread.
    Keeps the inference loop unblocked. Retries on SQLite lock.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._q = queue.Queue()
        self.start()

    def submit(self, fn, *args, **kwargs):
        self._q.put((fn, args, kwargs))

    def stop(self):
        self._q.put(None)

    def run(self):
        from django.db import connection as db_conn
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
                    if 'locked' in str(e).lower():
                        time.sleep(0.05 * (2 ** attempt))
                    else:
                        break
                except Exception:
                    break
            # Close after each task so stale connections don't accumulate
            try:
                db_conn.close()
            except Exception:
                pass


# ── Thread: inference + tracking + counting ──────────────────────────────────

class _Snapshot:
    """Thread-safe container for the latest inference result."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Optional[dict] = None

    def publish(self, data: dict):
        with self._lock:
            self._data = data

    def read(self) -> Optional[dict]:
        with self._lock:
            return self._data


class InferenceWorker(threading.Thread):
    """
    Owns ALL mutable counting state. Publishes immutable snapshots to a
    _Snapshot object that the MJPEG loop reads safely.

    State that lives here (and ONLY here):
        _tracker        SortTracker instance
        _counted_ids    set of track IDs already counted
        _prev_side      dict[track_id → 'above'|'below']
        _speed_history  dict[track_id → deque of BEV points]
        _cc             dict[vehicle_type → count]
        _inbound        int
        _outbound       int
        _total_speed    float
        _speed_count    int
    """

    def __init__(self, detector: 'VehicleDetector', snap: _Snapshot,
                 line_y: int, fps: float,
                 imgsz: int      = IMGSZ_STREAM,
                 stride: int     = STRIDE_STREAM,
                 csv_path: Optional[Path] = None,
                 db_writer: Optional[AsyncDBWriter] = None,
                 video_obj=None):
        super().__init__(daemon=True)
        self.det       = detector
        self.snap      = snap
        self.line_y    = line_y
        self.fps       = fps
        self.imgsz     = imgsz
        self.stride    = stride
        self.csv_path  = csv_path
        self.db_writer = db_writer
        self.video_obj = video_obj

        self._in_q  = queue.Queue(maxsize=1)
        self._stop  = threading.Event()

        # ── Counting state — PRIVATE ───────────────────────────────────
        self._tracker      = SortTracker(min_hits=MIN_TRACK_HITS, max_age=30)
        self._counted_ids  : set          = set()
        self._prev_side    : Dict[int,str] = {}
        self._speed_history: Dict[int, deque] = {}
        self._cc           : Dict[str,int] = {}
        self._inbound      = 0
        self._outbound     = 0
        self._total_speed  = 0.0
        self._speed_count  = 0
        self._frame_n      = 0
        self._last_db_save = 0
        self._xgb_cache    : dict = {}

    # ── Thread interface ───────────────────────────────────────────────

    def submit(self, frame: np.ndarray):
        """Send a frame for processing; drop if worker is already busy."""
        if self._in_q.full():
            try: self._in_q.get_nowait()
            except queue.Empty: pass
        self._in_q.put(frame)

    def stop(self):
        self._stop.set()
        self._in_q.put(None)

    def final_snapshot(self) -> dict:
        """Return a copy of the final counting state (called after stopping)."""
        avg = self._total_speed / self._speed_count if self._speed_count else 0.0
        return dict(
            cc       = dict(self._cc),
            inbound  = self._inbound,
            outbound = self._outbound,
            avg_speed= round(avg, 1),
            level    = _classify_congestion(0, avg, self._xgb_cache),
        )

    def run(self):
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if frame is None:
                break
            self._frame_n += 1
            try:
                self.snap.publish(self._process(frame))
            except Exception as exc:
                print(f'[InferenceWorker] {exc}')

    # ── Core processing ────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        mid  = w / 2.0

        # 1. Detect
        dets = self.det._detect(frame, self.imgsz)   # (N, 6)

        # 2. Track
        tracks = self._tracker.update(dets)

        # 3. Tally
        boxes_out = []
        bev_pts   = []

        for t in tracks:
            x1, y1, x2, y2 = t.tlbr.astype(int)
            cx, cy = (x1 + x2) // 2, y2

            M      = self.det.ML if cx < mid else self.det.MR
            bev_pt = self.det._to_bev(M, cx, cy)
            bev_pts.append(bev_pt)

            kph = self._calc_speed(t.track_id, bev_pt)
            self._count_crossing(t, x1, y1, x2, y2, kph)

            boxes_out.append({
                'box': (x1, y1, x2, y2),
                'tid': t.track_id,
                'cid': t.cls_id,
                'kph': kph,
            })

        n         = len(tracks)
        avg_speed = (self._total_speed / self._speed_count
                     if self._speed_count else 0.0)
        total_v   = sum(self._cc.values())
        level     = _classify_congestion(n, avg_speed, self._xgb_cache)
        queue_len = _queue_length(bev_pts, self.det.BEV_SCALE)

        # 4. Flush to DB (async, off the hot path)
        if self.db_writer and self.video_obj:
            if self._frame_n - self._last_db_save >= DB_SAVE_EVERY:
                self._last_db_save = self._frame_n
                self.db_writer.submit(
                    self.det._save_video_obj,
                    self.video_obj, total_v, avg_speed, level,
                    dict(self._cc), self._inbound, self._outbound,
                )
            if self._frame_n % DB_SYNC_EVERY == 0 and self.video_obj.location:
                self.db_writer.submit(
                    self.det._sync_to_traffic,
                    self.video_obj, total_v, avg_speed, level,
                    dict(self._cc), queue_len,
                )

        return {
            'boxes':     boxes_out,
            'n':         n,
            'total':     total_v,
            'cc':        dict(self._cc),
            'inbound':   self._inbound,
            'outbound':  self._outbound,
            'level':     level,
            'avg_speed': round(avg_speed, 1),
            'queue_len': queue_len,
            'ready':     True,
        }

    def _calc_speed(self, tid: int, bev_pt: Tuple[float, float]) -> float:
        if tid not in self._speed_history:
            self._speed_history[tid] = deque(maxlen=12)
        self._speed_history[tid].append(bev_pt)
        hist = self._speed_history[tid]
        if len(hist) < 3:
            return 0.0
        dx     = hist[-1][0] - hist[0][0]
        dy     = hist[-1][1] - hist[0][1]
        dist_m = math.hypot(dx, dy) / self.det.BEV_SCALE
        dt_s   = (len(hist) - 1) * self.stride / self.fps
        kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
        if 1.0 < kph < 200.0:
            self._total_speed += kph
            self._speed_count += 1
            return round(kph, 1)
        return 0.0

    def _count_crossing(self, t: Track,
                         x1: int, y1: int, x2: int, y2: int,
                         kph: float):
        cy       = (y1 + y2) / 2.0
        side     = 'above' if cy < self.line_y else 'below'
        prev     = self._prev_side.get(t.track_id)
        self._prev_side[t.track_id] = side

        if (prev and prev != side
                and t.track_id not in self._counted_ids
                and t.score >= COUNT_CONF_THRESH):
            self._counted_ids.add(t.track_id)
            lbl = CLASS_NAMES.get(t.cls_id, 'car')
            self._cc[lbl] = self._cc.get(lbl, 0) + 1
            direction = 'INBOUND' if prev == 'above' else 'OUTBOUND'
            if direction == 'INBOUND':
                self._inbound  += 1
            else:
                self._outbound += 1
            if self.csv_path:
                _append_csv(self.csv_path, {
                    'timestamp':    datetime.now().isoformat(timespec='seconds'),
                    'track_id':     t.track_id,
                    'vehicle_type': lbl,
                    'direction':    direction,
                    'speed_kph':    kph,
                })


# ── VehicleDetector ───────────────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv8 + SORT vehicle detection, tracking, and counting pipeline.

    Works with any model format (.pt, .onnx, .torchscript).
    Always uses model.predict() so behaviour is consistent across formats.
    Tracking and counting are handled internally by SortTracker.
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path = str(model_path)
        self.model      = YOLO(self.model_path) if YOLO else None
        if self.model and HAS_CUDA:
            self.model.to('cuda')

        # Bird's-Eye-View calibration matrices (default for standard intersection)
        self.BEV_SCALE        = 18
        self.VISIBLE_LENGTH_M = 60
        self._SRC_L = np.float32([[130,390],[415,390],[680,720],[-70,720]])
        self._SRC_R = np.float32([[415,390],[610,390],[960,720],[680,720]])
        self.ML, _, _ = self._bev_matrix(self._SRC_L, 3.75 * 5)
        self.MR, _, _ = self._bev_matrix(self._SRC_R, 3.75 * 3)

        # Warm up JIT so first real frame is not penalised
        if self.model:
            dummy = np.zeros((IMGSZ_STREAM, IMGSZ_STREAM, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, imgsz=IMGSZ_STREAM,
                               conf=CONF_THRESH, max_det=1)

    # ── BEV ───────────────────────────────────────────────────────────

    def _bev_matrix(self, src, road_width_m):
        w   = int(road_width_m * self.BEV_SCALE)
        h   = int(self.VISIBLE_LENGTH_M * self.BEV_SCALE)
        dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M   = cv2.getPerspectiveTransform(src, dst)
        return M, w, h

    def _to_bev(self, M, cx, cy) -> Tuple[float, float]:
        pt = np.float32([[[cx, cy]]])
        t  = cv2.perspectiveTransform(pt, M)
        return float(t[0, 0, 0]), float(t[0, 0, 1])

    # ── Detection ─────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray, imgsz: int) -> np.ndarray:
        """
        Run YOLO on frame. Returns (N, 6) ndarray [x1,y1,x2,y2,score,cls_id].
        Applies ROI masking, NMS IoU tuning, and size filtering.
        Returns empty (0,6) array when nothing passes the filters.
        """
        if not self.model:
            return np.empty((0, 6), dtype=np.float32)

        h, w = frame.shape[:2]
        roi  = _apply_roi(frame)

        results = self.model.predict(
            roi,
            classes   = CLASS_IDS,
            verbose   = False,
            imgsz     = imgsz,
            conf      = CONF_THRESH,
            iou       = NMS_IOU_THRESH,
            max_det   = 50,
            half      = HAS_CUDA,
        )
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes  = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        cids   = res.boxes.cls.cpu().numpy().astype(int)

        rows = []
        for box, sc, cid in zip(boxes, scores, cids):
            x1, y1, x2, y2 = map(int, box)
            if _size_ok(x1, y1, x2, y2, w, h):
                rows.append([x1, y1, x2, y2, float(sc), int(cid)])

        return np.array(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)

    # ── DB persistence ────────────────────────────────────────────────

    def _save_video_obj(self, video_obj, total, speed, level,
                         cc, inbound, outbound, status=None):
        video_obj.vehicle_count              = total
        video_obj.average_speed              = round(speed, 1)
        video_obj.predicted_congestion_level = level
        video_obj.car_count                  = cc.get('car', 0)
        video_obj.truck_count                = cc.get('truck', 0)
        video_obj.motorcycle_count           = cc.get('motorcycle', 0)
        video_obj.bus_count                  = cc.get('bus', 0)
        fields = ['vehicle_count', 'average_speed', 'predicted_congestion_level',
                  'car_count', 'truck_count', 'motorcycle_count', 'bus_count']
        if status:
            video_obj.status = status
            fields.append('status')
        video_obj.save(update_fields=fields)

    def _sync_to_traffic(self, video_obj, total, speed, level, cc, queue_len):
        from apps.traffic.models import TrafficReading, CongestionLevel
        from apps.alerts.services import NotificationService
        if not getattr(video_obj, 'location', None):
            return
        db_level = CongestionLevel(LEVEL_MAP.get(level, 'free_flow'))
        reading  = TrafficReading.objects.create(
            location         = video_obj.location,
            vehicle_count    = total,
            car_count        = cc.get('car', 0),
            truck_count      = cc.get('truck', 0),
            motorcycle_count = cc.get('motorcycle', 0),
            bus_count        = cc.get('bus', 0),
            avg_speed        = speed,
            queue_length     = queue_len,
            congestion_index = min(total * 2, 100),
            congestion_level = db_level,
            source           = 'vision',
        )
        if db_level in (CongestionLevel.HEAVY, CongestionLevel.GRIDLOCK):
            try:
                NotificationService.send_alert(reading)
            except Exception:
                pass

    def _save_session(self, tag, video_obj, final: dict,
                       csv_path: Optional[Path]):
        try:
            from apps.vision.models import VehicleCountSession
            from django.utils import timezone
            cc = final['cc']
            s, _ = VehicleCountSession.objects.update_or_create(
                session_tag = tag,
                defaults    = dict(
                    location         = getattr(video_obj, 'location', None),
                    total_count      = sum(cc.values()),
                    car_count        = cc.get('car', 0),
                    truck_count      = cc.get('truck', 0),
                    bus_count        = cc.get('bus', 0),
                    motorcycle_count = cc.get('motorcycle', 0),
                    inbound_count    = final['inbound'],
                    outbound_count   = final['outbound'],
                    avg_speed        = final['avg_speed'],
                    peak_congestion  = final['level'],
                    ended_at         = timezone.now(),
                ),
            )
            if csv_path and csv_path.exists():
                s.csv_file.name = str(csv_path.relative_to(Path('media')))
                s.save(update_fields=['csv_file'])
        except Exception:
            pass

    # ── Drawing ───────────────────────────────────────────────────────

    @staticmethod
    def _speed_color(kph: float) -> tuple:
        if kph < 40:  return (0, 220, 0)
        if kph < 100: return (0, 220, 220)
        return (0, 60, 255)

    def _draw(self, frame: np.ndarray, snap: dict, line_y: int) -> np.ndarray:
        """Annotate a display frame with bounding boxes and HUD."""
        lc = {'FREE FLOW':(0,220,0), 'MODERATE':(0,220,220),
              'HEAVY':(0,140,255), 'GRIDLOCK':(0,50,255)}.get(
                  snap['level'], (200,200,200))

        # Counting line
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), (0, 200, 255), 2)
        cv2.putText(frame, 'COUNTING LINE',
                    (w // 2 - 65, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,200,255), 1, cv2.LINE_AA)

        # Bounding boxes
        for b in snap['boxes']:
            x1, y1, x2, y2 = b['box']
            color = self._speed_color(b['kph'])
            label = f"#{b['tid']} {CLASS_NAMES.get(b['cid'],'?')} {b['kph']:.0f}km/h"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # HUD panel
        cc = snap['cc']
        cv2.rectangle(frame, (8, 8), (430, 182), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (430, 182), (60, 60, 60), 1)
        cv2.putText(frame, f"Vehicles : {snap['total']}",
                    (16, 38),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Speed    : {snap['avg_speed']:.1f} km/h",
                    (16, 68),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,220,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Status   : {snap['level']}",
                    (16, 98),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, lc, 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"IN:{snap['inbound']}  OUT:{snap['outbound']}  Q:{snap['queue_len']}m",
                    (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180,180,255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    f"C:{cc.get('car',0)}  T:{cc.get('truck',0)}  "
                    f"B:{cc.get('bus',0)}  M:{cc.get('motorcycle',0)}",
                    (16, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)
        return frame

    # ── Public: MJPEG streaming ───────────────────────────────────────

    def stream_inference(self, video_source, output_video_obj=None):
        """
        Yield MJPEG byte-frames suitable for Django StreamingHttpResponse.

        InferenceWorker runs in a background thread and publishes immutable
        result snapshots.  This loop reads snapshots and yields frames at
        maximum speed — no artificial sleep.
        """
        if not self.model:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            return

        tag      = f'stream_{int(time.time())}'
        csv_path = _csv_for(tag)
        reader   = FrameReader(video_source)
        reader.start()

        fps    = reader.fps
        w      = STREAM_WIDTH
        h      = int(reader.height * (STREAM_WIDTH / reader.width))
        line_y = int(h * LINE_RATIO)

        snap      = _Snapshot()
        db_writer = AsyncDBWriter() if output_video_obj else None

        worker = InferenceWorker(
            detector   = self,
            snap       = snap,
            line_y     = line_y,
            fps        = fps,
            imgsz      = IMGSZ_STREAM,
            stride     = STRIDE_STREAM,
            csv_path   = csv_path,
            db_writer  = db_writer,
            video_obj  = output_video_obj,
        )
        worker.start()

        frame_n = 0
        while True:
            frame = reader.read(timeout=2.0)
            if frame is None:
                break

            frame_n += 1
            disp = cv2.resize(frame, (w, h))

            if frame_n % STRIDE_STREAM == 0:
                worker.submit(disp.copy())

            result = snap.read()
            if result:
                disp = self._draw(disp, result, line_y)

            ok, buf = cv2.imencode('.jpg', disp, _ENCODE_PARAM)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')

        worker.stop()
        worker.join(timeout=3.0)
        reader.stop()

        if output_video_obj and db_writer:
            final = worker.final_snapshot()
            total = sum(final['cc'].values())
            db_writer.submit(
                self._save_video_obj,
                output_video_obj, total, final['avg_speed'],
                final['level'], final['cc'],
                final['inbound'], final['outbound'], status='completed',
            )
            if getattr(output_video_obj, 'location', None):
                db_writer.submit(
                    self._sync_to_traffic,
                    output_video_obj, total, final['avg_speed'],
                    final['level'], final['cc'], 0.0,
                )
            db_writer.submit(self._save_session, tag, output_video_obj,
                             final, csv_path)
            time.sleep(0.5)
            db_writer.stop()

    # ── Public: background batch ──────────────────────────────────────

    def process_video(self, video_path: str, video_obj=None) -> dict:
        """
        Process an entire video file in the calling thread (runs in background
        via _run_inference_thread in views.py).  Uses STRIDE_BATCH and the
        same detection/tracking pipeline as streaming.
        """
        if not self.model:
            if video_obj:
                video_obj.status = 'failed'
                video_obj.save(update_fields=['status'])
            return {}

        tag      = f'batch_{int(time.time())}'
        csv_path = _csv_for(tag)
        db_writer = AsyncDBWriter()

        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)               or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        proc_w = 640
        proc_h = int(orig_h * (proc_w / orig_w))
        line_y = int(proc_h * LINE_RATIO)

        # All counting state is local to this function (no threads)
        tracker      = SortTracker(min_hits=MIN_TRACK_HITS, max_age=30)
        counted_ids  : set          = set()
        prev_side    : Dict[int,str]= {}
        speed_history: Dict[int, deque] = {}
        cc           : Dict[str,int]= {}
        inbound  = outbound = 0
        total_speed = 0.0;  speed_count = 0
        level    = 'FREE FLOW';  queue_len = 0.0
        frame_n  = 0;  last_db = 0
        xgb_cache: dict = {}

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1
            if frame_n % STRIDE_BATCH != 0:
                continue

            small  = cv2.resize(frame, (proc_w, proc_h))
            dets   = self._detect(small, IMGSZ_BATCH)
            tracks = tracker.update(dets)

            bev_pts = []
            for t in tracks:
                x1, y1, x2, y2 = t.tlbr.astype(int)
                cx, cy = (x1 + x2) // 2, y2
                M      = self.ML if cx < proc_w / 2 else self.MR
                bev_pt = self._to_bev(M, cx, cy)
                bev_pts.append(bev_pt)

                # Speed
                if t.track_id not in speed_history:
                    speed_history[t.track_id] = deque(maxlen=12)
                speed_history[t.track_id].append(bev_pt)
                hist = speed_history[t.track_id]
                kph  = 0.0
                if len(hist) >= 3:
                    dx     = hist[-1][0] - hist[0][0]
                    dy     = hist[-1][1] - hist[0][1]
                    dist_m = math.hypot(dx, dy) / self.BEV_SCALE
                    dt_s   = (len(hist) - 1) * STRIDE_BATCH / fps
                    kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                    if 1.0 < kph < 200.0:
                        total_speed += kph;  speed_count += 1

                # Crossing
                cy_c  = (y1 + y2) / 2.0
                side  = 'above' if cy_c < line_y else 'below'
                prev  = prev_side.get(t.track_id)
                if (prev and prev != side
                        and t.track_id not in counted_ids
                        and t.score >= COUNT_CONF_THRESH):
                    counted_ids.add(t.track_id)
                    lbl = CLASS_NAMES.get(t.cls_id, 'car')
                    cc[lbl] = cc.get(lbl, 0) + 1
                    direction = 'INBOUND' if prev == 'above' else 'OUTBOUND'
                    if direction == 'INBOUND': inbound  += 1
                    else:                      outbound += 1
                    _append_csv(csv_path, {
                        'timestamp':    datetime.now().isoformat(timespec='seconds'),
                        'track_id':     t.track_id,
                        'vehicle_type': lbl,
                        'direction':    direction,
                        'speed_kph':    round(kph, 1),
                    })
                prev_side[t.track_id] = side

            n         = len(tracks)
            avg_speed = total_speed / speed_count if speed_count else 0.0
            queue_len = _queue_length(bev_pts, self.BEV_SCALE)
            level     = _classify_congestion(n, avg_speed, xgb_cache)
            total_v   = sum(cc.values())

            if video_obj and frame_n - last_db >= DB_SAVE_EVERY:
                last_db = frame_n
                db_writer.submit(self._save_video_obj,
                                 video_obj, total_v, avg_speed, level,
                                 dict(cc), inbound, outbound)
            if video_obj and frame_n % DB_SYNC_EVERY == 0 and video_obj.location:
                db_writer.submit(self._sync_to_traffic,
                                 video_obj, total_v, avg_speed, level,
                                 dict(cc), queue_len)

        cap.release()

        avg_speed = total_speed / speed_count if speed_count else 0.0
        total_v   = sum(cc.values())
        final = dict(cc=cc, inbound=inbound, outbound=outbound,
                     avg_speed=round(avg_speed, 1), level=level)

        if video_obj:
            db_writer.submit(self._save_video_obj,
                             video_obj, total_v, avg_speed, level,
                             dict(cc), inbound, outbound, status='completed')
            if video_obj.location:
                db_writer.submit(self._sync_to_traffic,
                                 video_obj, total_v, avg_speed, level,
                                 dict(cc), queue_len)
            db_writer.submit(self._save_session, tag, video_obj,
                             final, csv_path)
            time.sleep(0.6)
        db_writer.stop()

        return {
            'total':      total_v,
            'car':        cc.get('car', 0),
            'truck':      cc.get('truck', 0),
            'bus':        cc.get('bus', 0),
            'motorcycle': cc.get('motorcycle', 0),
            'inbound':    inbound,
            'outbound':   outbound,
            'avg_speed':  round(avg_speed, 1),
            'level':      level,
            'csv_path':   str(csv_path) if csv_path.exists() else None,
        }
