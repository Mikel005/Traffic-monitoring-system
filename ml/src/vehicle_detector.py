"""
vehicle_detector.py — YOLOv8 + ByteTrack / IoU-based vehicle pipeline.

Tracking strategy
─────────────────
  .pt model       → model.track(persist=True, tracker="bytetrack.yaml")
                    (built-in ultralytics ByteTrack — best accuracy)
  .torchscript /
  .onnx model     → model.predict() + IoUTracker
                    (IoU Hungarian-style matching — works on any format)

Architecture
────────────
stream_inference()
  ├── FrameReader thread    — decouples I/O from inference
  ├── InferenceWorker thread — YOLO + tracker; cached overlay on skipped frames
  └── AsyncDBWriter thread  — all DB writes off the hot path

process_video()  — non-streaming background batch
"""

import cv2
import math
import time
import queue
import threading
import numpy as np
from collections import deque

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

# ── Pipeline constants ────────────────────────────────────────────────────────
VEHICLE_CLASS_MAP = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES   = list(VEHICLE_CLASS_MAP.keys())

LEVEL_MAP = {
    "FREE FLOW": "free_flow",
    "MODERATE":  "moderate",
    "HEAVY":     "heavy",
    "GRIDLOCK":  "gridlock",
}

# Inference tuning
STRIDE       = 2      # run YOLO every N frames (2 = ~half fps, good tracking)
IMGSZ        = 320    # inference resolution — 320 is the sweet spot on CPU
CONF_THRESH  = 0.40   # confidence threshold
MAX_DET      = 50     # cap detections per frame to avoid CPU spike

# Streaming
STREAM_WIDTH  = 640   # resize output frames before JPEG encode
JPEG_QUALITY  = 72

# DB write intervals (in frames)
DB_SAVE_EVERY = 90
DB_SYNC_EVERY = 150

# XGBoost congestion cache
XGB_CACHE_SEC = 2.0

_ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


def _model_supports_track(model_path: str) -> bool:
    """Only native PyTorch .pt models support .track(); TorchScript/ONNX do not."""
    return str(model_path).lower().endswith('.pt')


# ── IoU Tracker (used when model.track() is unavailable) ─────────────────────

class IoUTracker:
    """
    Lightweight IoU-based multi-object tracker.

    Matches detections to existing tracks using bounding-box IoU and greedy
    assignment (same principle as ByteTrack's low-confidence second pass).
    Tracks survive up to `max_age` missed frames before being dropped —
    this handles brief occlusions cleanly.
    """

    def __init__(self, iou_thresh: float = 0.30, max_age: int = 8):
        self.tracks     = {}   # tid -> {'box', 'age', 'hits', 'cid'}
        self.next_id    = 1
        self.iou_thresh = iou_thresh
        self.max_age    = max_age

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, detections):
        """
        Args:
            detections: list of (box_xyxy_tuple, class_id_int)
        Returns:
            list of (box_xyxy_tuple, track_id, class_id)
        """
        tids = list(self.tracks.keys())

        # ── Build IoU cost matrix ──────────────────────────────────────────
        if tids and detections:
            iou_mat = np.zeros((len(detections), len(tids)), dtype=np.float32)
            for di, (dbox, _) in enumerate(detections):
                for ti, tid in enumerate(tids):
                    iou_mat[di, ti] = self._iou(dbox, self.tracks[tid]['box'])
        else:
            iou_mat = np.zeros((len(detections), len(tids)), dtype=np.float32)

        # ── Greedy assignment (highest IoU first) ──────────────────────────
        matched_d, matched_t = set(), set()
        assignments = []
        work = iou_mat.copy()
        for _ in range(min(len(detections), len(tids))):
            if work.size == 0:
                break
            di, ti = np.unravel_index(np.argmax(work), work.shape)
            if work[di, ti] < self.iou_thresh:
                break
            assignments.append((di, ti))
            matched_d.add(di); matched_t.add(ti)
            work[di, :] = -1; work[:, ti] = -1

        # ── Apply updates ──────────────────────────────────────────────────
        out = []
        for di, ti in assignments:
            tid = tids[ti]
            box, cid = detections[di]
            self.tracks[tid].update({'box': box, 'age': 0, 'cid': cid})
            self.tracks[tid]['hits'] += 1
            out.append((box, tid, cid))

        # New tracks for unmatched detections
        for di, (box, cid) in enumerate(detections):
            if di not in matched_d:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'box': box, 'age': 0, 'hits': 1, 'cid': cid}
                out.append((box, tid, cid))

        # Age / retire unmatched existing tracks
        for ti, tid in enumerate(tids):
            if ti not in matched_t:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        return out


# ── Background helpers ────────────────────────────────────────────────────────

class FrameReader(threading.Thread):
    """Reads any OpenCV source in a thread; drops stale frames to stay current."""

    def __init__(self, source, maxsize: int = 2):
        super().__init__(daemon=True)
        self.cap    = cv2.VideoCapture(source)
        self._q     = queue.Queue(maxsize=maxsize)
        self._stop  = threading.Event()
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)                  or 25.0
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))     or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    or 480

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
    """Fire-and-forget DB writes — never blocks the inference loop."""

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
                    if 'locked' in str(e).lower():
                        time.sleep(0.05 * (2 ** attempt))
                    else:
                        break
                except Exception:
                    break
            try:
                close_old_connections()
            except Exception:
                pass


# ── Inference worker thread ───────────────────────────────────────────────────

class InferenceWorker(threading.Thread):
    """
    Runs detection + tracking in a background thread.
    Accepts frames via submit(); exposes latest results via .results dict.
    Supports both model.track() (ByteTrack, .pt models) and
    model.predict() + IoUTracker (.torchscript / .onnx models).
    """

    def __init__(self, detector, speed_history, counted_ids, cc,
                 total_speed, speed_count, fps, use_native_track):
        super().__init__(daemon=True)
        self.det            = detector
        self.speed_history  = speed_history
        self.counted_ids    = counted_ids
        self.cc             = cc
        self.total_speed    = total_speed
        self.speed_count    = speed_count
        self.fps            = fps
        self.use_native_track = use_native_track

        self._in_q   = queue.Queue(maxsize=1)
        self._stop   = threading.Event()
        self.results = {
            'boxes': [], 'bev_pts': [], 'n': 0,
            'level': 'FREE FLOW', 'avg_speed': 0.0,
            'queue_len': 0.0, 'ready': False,
        }
        self._iou_tracker = IoUTracker(iou_thresh=0.30, max_age=8)

    def submit(self, frame, w, h, line_y):
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

            frame, w, h, line_y = item
            try:
                self._process(frame, w, h, line_y)
            except Exception as e:
                print(f"[InferenceWorker] {e}")
                time.sleep(0.1)

    def _process(self, frame, w, h, line_y):
        mid   = w / 2
        model = self.det.model

        # ── Detection + tracking ──────────────────────────────────────────
        if self.use_native_track:
            # ByteTrack via ultralytics (.pt model)
            results = model.track(
                frame,
                classes   = self.det.vehicle_classes,
                persist   = True,
                verbose   = False,
                imgsz     = IMGSZ,
                conf      = CONF_THRESH,
                max_det   = MAX_DET,
                half      = HAS_CUDA,
                tracker   = "bytetrack.yaml",
            )
            res = results[0]
            detections_tracked = []   # (box, tid, cid)

            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                cids  = res.boxes.cls.cpu().numpy().astype(int)
                tids  = (res.boxes.id.cpu().numpy().astype(int)
                         if res.boxes.id is not None
                         else np.arange(len(boxes)))
                for box, tid, cid in zip(boxes, tids, cids):
                    detections_tracked.append((tuple(box), int(tid), int(cid)))

        else:
            # model.predict() + IoUTracker (.torchscript / .onnx)
            results = model.predict(
                frame,
                classes = self.det.vehicle_classes,
                verbose = False,
                imgsz   = IMGSZ,
                conf    = CONF_THRESH,
                max_det = MAX_DET,
                half    = False,     # half=True unsupported on CPU TorchScript
            )
            res  = results[0]
            dets = []
            if res.boxes is not None and len(res.boxes):
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                cids  = res.boxes.cls.cpu().numpy().astype(int)
                for box, cid in zip(boxes, cids):
                    dets.append((tuple(box), int(cid)))
            tracked = self._iou_tracker.update(dets)
            detections_tracked = [(box, tid, cid) for box, tid, cid in tracked]

        # ── Speed, counting, annotation ──────────────────────────────────
        boxes_out   = []
        bev_pts     = []
        n_on_screen = len(detections_tracked)

        for (x1, y1, x2, y2), tid, cid in detections_tracked:
            cx, cy = (x1 + x2) // 2, y2
            M      = self.det.ML if cx < mid else self.det.MR
            bev_pt = self.det._to_bev(M, cx, cy)
            bev_pts.append(bev_pt)

            # Speed via BEV displacement history
            if tid not in self.speed_history:
                self.speed_history[tid] = deque(maxlen=12)
            self.speed_history[tid].append(bev_pt)

            kph = 0.0
            hist = self.speed_history[tid]
            if len(hist) >= 3:
                dx     = hist[-1][0] - hist[0][0]
                dy     = hist[-1][1] - hist[0][1]
                dist_m = math.hypot(dx, dy) / self.det.BEV_SCALE
                # time covered = (samples-1) * STRIDE frames / fps
                dt_s   = (len(hist) - 1) * STRIDE / self.fps
                kph    = (dist_m / dt_s) * 3.6 if dt_s > 0 else 0.0
                if 1.0 < kph < 200.0:   # sanity clamp
                    self.total_speed[0] += kph
                    self.speed_count[0] += 1

            # Counting line crossing (bottom-centre of bbox)
            cy_centre = (y1 + y2) / 2
            if tid not in self.counted_ids and abs(cy_centre - line_y) < 20:
                self.counted_ids.add(tid)
                label = VEHICLE_CLASS_MAP.get(cid, 'car')
                self.cc[label] = self.cc.get(label, 0) + 1

            boxes_out.append({'box': (x1, y1, x2, y2), 'tid': tid,
                               'cid': cid, 'kph': kph})

        avg_speed = (self.total_speed[0] / self.speed_count[0]
                     if self.speed_count[0] else 0.0)
        queue_len = self.det._queue_length(bev_pts)
        level     = self.det._xgb_classify(
            n_on_screen, avg_speed, min(n_on_screen * 2, 100)
        )

        self.results = {
            'boxes':     boxes_out,
            'bev_pts':   bev_pts,
            'n':         n_on_screen,
            'level':     level,
            'avg_speed': avg_speed,
            'queue_len': queue_len,
            'ready':     True,
        }


# ── Main detector ─────────────────────────────────────────────────────────────

class VehicleDetector:

    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path       = str(model_path)
        self.use_native_track = _model_supports_track(self.model_path)
        self.model            = YOLO(self.model_path) if YOLO else None
        if self.model and HAS_CUDA:
            self.model.to('cuda')
        self.vehicle_classes = VEHICLE_CLASSES

        # Bird's-Eye-View calibration (default for a standard intersection camera)
        self.BEV_SCALE        = 18
        self.VISIBLE_LENGTH_M = 60
        self.SRC_ROAD_L = np.float32([[130,390],[415,390],[680,720],[-70,720]])
        self.SRC_ROAD_R = np.float32([[415,390],[610,390],[960,720],[680,720]])
        self.ML, self.bev_wL, self.bev_hL = self._bev_matrix(self.SRC_ROAD_L, 3.75*5)
        self.MR, self.bev_wR, self.bev_hR = self._bev_matrix(self.SRC_ROAD_R, 3.75*3)

        # Cached XGBoost congestion classification
        self._xgb_last_run: float = 0.0
        self._xgb_cached:   str   = ''

    # ── BEV helpers ───────────────────────────────────────────────────────────

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

    # ── Congestion classification ─────────────────────────────────────────────

    def _xgb_classify(self, count: int, speed: float, idx: int) -> str:
        now = time.monotonic()
        if now - self._xgb_last_run < XGB_CACHE_SEC and self._xgb_cached:
            return self._xgb_cached

        try:
            from apps.predictions.services import classify
            import datetime
            dt    = datetime.datetime.now()
            feats = np.zeros((1, 28), dtype=np.float32)
            feats[0, 0] = count
            feats[0, 1] = speed
            feats[0, 2] = idx
            feats[0, 3] = math.sin(2 * math.pi * dt.hour / 24)
            feats[0, 4] = math.cos(2 * math.pi * dt.hour / 24)
            feats[0, 5] = math.sin(2 * math.pi * dt.weekday() / 7)
            feats[0, 6] = math.cos(2 * math.pi * dt.weekday() / 7)
            result = classify(feats)
            if result.get('label') is not None:
                label_map = {0: "FREE FLOW", 1: "MODERATE", 2: "HEAVY", 3: "GRIDLOCK"}
                self._xgb_cached   = label_map[result['label']]
                self._xgb_last_run = now
                return self._xgb_cached
        except Exception:
            pass

        # Heuristic fallback (used until XGBoost model is trained)
        if   count > 20: level = "GRIDLOCK"
        elif count > 12: level = "HEAVY"
        elif count > 5:  level = "MODERATE"
        else:            level = "FREE FLOW"
        self._xgb_cached   = level
        self._xgb_last_run = now
        return level

    # ── Visual helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _speed_color(kph: float):
        if kph < 40:  return (0, 220,   0)
        if kph < 100: return (0, 220, 220)
        return (0, 60, 255)

    def _draw_hud(self, frame, vehicles, speed, level, cc, queue_len):
        colour = {
            "FREE FLOW": (0, 220, 0), "MODERATE": (0, 220, 220),
            "HEAVY":     (0, 140, 255), "GRIDLOCK": (0, 50, 255),
        }.get(level, (200, 200, 200))
        cv2.rectangle(frame, (8, 8), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (400, 150), (60, 60, 60), 1)
        cv2.putText(frame, f"Vehicles : {vehicles}",
                    (18,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Avg Speed: {speed:.1f} km/h",
                    (18,  74), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 0),   2, cv2.LINE_AA)
        cv2.putText(frame, f"Status   : {level}",
                    (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.70, colour,        2, cv2.LINE_AA)
        detail = (f"C:{cc.get('car',0)}  T:{cc.get('truck',0)}  "
                  f"B:{cc.get('bus',0)}  M:{cc.get('motorcycle',0)}  Q:{queue_len}m")
        cv2.putText(frame, detail,
                    (18, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1, cv2.LINE_AA)

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _save_video_obj(self, video_obj, vehicles, speed, level, cc, status=None):
        video_obj.vehicle_count              = vehicles
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

    def _sync_to_traffic(self, video_obj, vehicles, speed, level, cc, queue_len):
        from apps.traffic.models import TrafficReading, CongestionLevel
        from apps.alerts.services import NotificationService
        if not getattr(video_obj, 'location', None):
            return
        db_level = CongestionLevel(LEVEL_MAP.get(level, 'free_flow'))
        reading  = TrafficReading.objects.create(
            location         = video_obj.location,
            vehicle_count    = vehicles,
            car_count        = cc.get('car', 0),
            truck_count      = cc.get('truck', 0),
            motorcycle_count = cc.get('motorcycle', 0),
            bus_count        = cc.get('bus', 0),
            avg_speed        = speed,
            queue_length     = queue_len,
            congestion_index = min(vehicles * 2, 100),
            congestion_level = db_level,
            source           = 'vision',
        )
        if db_level in (CongestionLevel.HEAVY, CongestionLevel.GRIDLOCK):
            try:
                NotificationService.send_alert(reading)
            except Exception:
                pass

    # ── Public: MJPEG streaming ───────────────────────────────────────────────

    def stream_inference(self, video_source, output_video_obj=None):
        """
        Yield MJPEG frames with real-time YOLO + ByteTrack/IoU overlays.
        FrameReader and InferenceWorker run in parallel threads so the
        main loop never blocks on I/O or inference.
        """
        if not self.model:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n'
            return

        reader    = FrameReader(video_source)
        reader.start()
        db_writer = AsyncDBWriter() if output_video_obj else None

        fps    = reader.fps
        w      = STREAM_WIDTH
        h      = int(reader.height * (STREAM_WIDTH / reader.width))
        line_y = int(h * 0.60)

        frame_n      = 0
        counted_ids  = set()
        speed_history= {}
        cc           = {}
        total_speed  = [0.0]
        speed_count  = [0]
        last_db_save = 0
        target_delay = 1.0 / fps

        worker = InferenceWorker(
            self, speed_history, counted_ids, cc,
            total_speed, speed_count, fps,
            use_native_track=self.use_native_track,
        )
        worker.start()

        while True:
            t0    = time.time()
            frame = reader.read(timeout=2.0)
            if frame is None:
                break

            frame_n   += 1
            disp_frame = cv2.resize(frame, (w, h))

            if frame_n % STRIDE == 0:
                worker.submit(disp_frame.copy(), w, h, line_y)

            # Draw last inference results onto display frame
            res = worker.results
            if res['ready']:
                cv2.line(disp_frame, (0, line_y), (w, line_y), (200, 0, 200), 1)
                for b in res['boxes']:
                    x1, y1, x2, y2 = b['box']
                    color = self._speed_color(b['kph'])
                    label = (f"#{b['tid']} "
                             f"{VEHICLE_CLASS_MAP.get(b['cid'], '?')} "
                             f"{b['kph']:.0f}km/h")
                    cv2.rectangle(disp_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(disp_frame, label, (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

                total_v = sum(cc.values())
                self._draw_hud(disp_frame, total_v, res['avg_speed'],
                               res['level'], cc, res['queue_len'])

                if output_video_obj and db_writer:
                    if frame_n - last_db_save >= DB_SAVE_EVERY:
                        last_db_save = frame_n
                        db_writer.submit(self._save_video_obj, output_video_obj,
                                         total_v, res['avg_speed'], res['level'], cc.copy())
                    if frame_n % DB_SYNC_EVERY == 0:
                        db_writer.submit(self._sync_to_traffic, output_video_obj,
                                         total_v, res['avg_speed'], res['level'],
                                         cc.copy(), res['queue_len'])

            ok, buf = cv2.imencode('.jpg', disp_frame, _ENCODE_PARAM)
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')

            elapsed = time.time() - t0
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

        worker.stop()
        reader.stop()

        if output_video_obj and db_writer:
            res     = worker.results
            total_v = sum(cc.values())
            db_writer.submit(self._save_video_obj, output_video_obj,
                             total_v, res['avg_speed'], res['level'],
                             cc.copy(), status='completed')
            if getattr(output_video_obj, 'location', None):
                db_writer.submit(self._sync_to_traffic, output_video_obj,
                                 total_v, res['avg_speed'], res['level'],
                                 cc.copy(), res['queue_len'])
            time.sleep(0.5)
            db_writer.stop()

    # ── Public: background batch processing ───────────────────────────────────

    def process_video(self, video_path: str, video_obj=None):
        """
        Non-streaming batch processing for the background upload thread.
        Uses the same IoUTracker / ByteTrack logic as stream_inference.
        """
        if not self.model:
            if video_obj:
                video_obj.status = 'failed'
                video_obj.save(update_fields=['status'])
            return

        db_writer    = AsyncDBWriter()
        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)                 or 25.0
        orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    or 640
        orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   or 480
        proc_w       = 640
        proc_h       = int(orig_h * (proc_w / orig_w))
        line_y       = int(proc_h * 0.60)

        frame_n      = 0
        counted_ids  = set()
        speed_history= {}
        cc           = {}
        total_speed  = 0.0
        speed_count  = 0
        level        = "FREE FLOW"
        queue_len    = 0.0
        iou_tracker  = IoUTracker(iou_thresh=0.30, max_age=8)
        mid          = proc_w / 2

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
                tracked = []
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                    cids  = res.boxes.cls.cpu().numpy().astype(int)
                    tids  = (res.boxes.id.cpu().numpy().astype(int)
                             if res.boxes.id is not None
                             else np.arange(len(boxes)))
                    for box, tid, cid in zip(boxes, tids, cids):
                        tracked.append((tuple(box), int(tid), int(cid)))
            else:
                results = self.model.predict(
                    small, classes=self.vehicle_classes, verbose=False,
                    imgsz=IMGSZ, conf=CONF_THRESH, max_det=MAX_DET, half=False,
                )
                res  = results[0]
                dets = []
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                    cids  = res.boxes.cls.cpu().numpy().astype(int)
                    for box, cid in zip(boxes, cids):
                        dets.append((tuple(box), int(cid)))
                tracked = [(b, tid, c) for b, tid, c in iou_tracker.update(dets)]

            # Process tracks
            bev_pts = []
            for (x1, y1, x2, y2), tid, cid in tracked:
                cx, cy = (x1 + x2) // 2, y2
                M      = self.ML if cx < mid else self.MR
                bev_pt = self._to_bev(M, cx, cy)
                bev_pts.append(bev_pt)

                if tid not in speed_history:
                    speed_history[tid] = deque(maxlen=12)
                speed_history[tid].append(bev_pt)

                hist = speed_history[tid]
                if len(hist) >= 3:
                    dx     = hist[-1][0] - hist[0][0]
                    dy     = hist[-1][1] - hist[0][1]
                    dist_m = math.hypot(dx, dy) / self.BEV_SCALE
                    dt_s   = (len(hist) - 1) * STRIDE / fps
                    kph    = (dist_m / dt_s) * 3.6 if dt_s > 0 else 0.0
                    if 1.0 < kph < 200.0:
                        total_speed += kph
                        speed_count += 1

                cy_centre = (y1 + y2) / 2
                if tid not in counted_ids and abs(cy_centre - line_y) < 20:
                    counted_ids.add(tid)
                    lbl = VEHICLE_CLASS_MAP.get(cid, 'car')
                    cc[lbl] = cc.get(lbl, 0) + 1

            n_on_screen = len(tracked)
            avg_speed   = total_speed / speed_count if speed_count else 0.0
            queue_len   = self._queue_length(bev_pts)
            level       = self._xgb_classify(n_on_screen, avg_speed, min(n_on_screen * 2, 100))
            total_v     = sum(cc.values())

            if video_obj and frame_n % DB_SAVE_EVERY == 0:
                db_writer.submit(self._save_video_obj,
                                 video_obj, total_v, avg_speed, level, cc.copy())
            if video_obj and frame_n % DB_SYNC_EVERY == 0 and video_obj.location:
                db_writer.submit(self._sync_to_traffic,
                                 video_obj, total_v, avg_speed, level, cc.copy(), queue_len)

        cap.release()

        if video_obj:
            avg_speed = total_speed / speed_count if speed_count else 0.0
            total_v   = sum(cc.values())
            db_writer.submit(self._save_video_obj,
                             video_obj, total_v, avg_speed, level, cc.copy(),
                             status='completed')
            if video_obj.location:
                db_writer.submit(self._sync_to_traffic,
                                 video_obj, total_v, avg_speed, level, cc.copy(), queue_len)
            time.sleep(0.6)
        db_writer.stop()
