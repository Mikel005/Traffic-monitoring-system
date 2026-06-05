"""
byte_tracker.py  —  ByteTrack multi-object tracker with Kalman filter.

Based on: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
          (Zhang et al., ECCV 2022)

Designed to run on CPU with any detector that outputs raw bounding boxes
(i.e. when model.track() is not available — TorchScript / ONNX models).

Usage
─────
    tracker = ByteTracker()
    for frame in video:
        boxes, scores, class_ids = detector(frame)
        tracks = tracker.update(boxes, scores, class_ids)
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            print(t.track_id, t.cls_id, t.score)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Constant-velocity Kalman filter for axis-aligned bounding boxes.

    State  (8-dim): [cx, cy, w, h,  vcx, vcy, vw, vh]
    Measurement (4-dim): [cx, cy, w, h]
    """

    def __init__(self):
        n = 4   # observation dims
        dt = 1  # time step (frames)

        # Transition matrix F (constant velocity)
        self.F = np.eye(2 * n)
        for i in range(n):
            self.F[i, n + i] = dt

        # Observation matrix H
        self.H = np.eye(n, 2 * n)

        # Process noise Q (scales with box size — tuned empirically)
        self._pos_std = 1.0 / 20.0
        self._vel_std = 1.0 / 160.0

    def _process_noise(self, w: float, h: float) -> np.ndarray:
        s = np.array([
            self._pos_std * w,
            self._pos_std * h,
            self._pos_std * w,
            self._pos_std * h,
            self._vel_std * w,
            self._vel_std * h,
            self._vel_std * w,
            self._vel_std * h,
        ])
        return np.diag(s ** 2)

    def _meas_noise(self, w: float, h: float) -> np.ndarray:
        s = np.array([
            self._pos_std * w,
            self._pos_std * h,
            self._pos_std * w,
            self._pos_std * h,
        ])
        return np.diag((s * 2) ** 2)

    def initiate(self, meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """meas: [cx, cy, w, h]"""
        x = np.zeros(8)
        x[:4] = meas
        w, h = meas[2], meas[3]
        P = np.diag([
            (2 * self._pos_std * w) ** 2,
            (2 * self._pos_std * h) ** 2,
            (2 * self._pos_std * w) ** 2,
            (2 * self._pos_std * h) ** 2,
            (10 * self._vel_std * w) ** 2,
            (10 * self._vel_std * h) ** 2,
            (10 * self._vel_std * w) ** 2,
            (10 * self._vel_std * h) ** 2,
        ])
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray):
        Q = self._process_noise(x[2], x[3])
        x_p = self.F @ x
        P_p = self.F @ P @ self.F.T + Q
        # Clamp width/height to stay positive
        x_p[2] = max(x_p[2], 1.0)
        x_p[3] = max(x_p[3], 1.0)
        return x_p, P_p

    def update(self, x: np.ndarray, P: np.ndarray, meas: np.ndarray):
        R = self._meas_noise(x[2], x[3])
        S = self.H @ P @ self.H.T + R
        K = P @ self.H.T @ np.linalg.inv(S)
        y = meas - self.H @ x
        x_new = x + K @ y
        P_new = (np.eye(8) - K @ self.H) @ P
        return x_new, P_new


# ── Track state ────────────────────────────────────────────────────────────────

class TrackState:
    NEW      = 0   # first frame — not yet confirmed
    TRACKED  = 1   # confirmed active track
    LOST     = 2   # temporarily missing
    REMOVED  = 3   # deleted


_KF = KalmanFilter()   # shared instance (stateless)


class STrack:
    """
    Single tracked object.

    Each track maintains its own Kalman state (x, P) and is uniquely
    identified by track_id.  The track is "confirmed" after min_hits
    consecutive detections and "lost" after max_age missed frames.
    """

    _next_id: int = 1

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1

    def __init__(self, tlbr: np.ndarray, score: float, cls_id: int):
        self.track_id = STrack._next_id
        STrack._next_id += 1
        self.state    = TrackState.NEW
        self.score    = float(score)
        self.cls_id   = int(cls_id)
        self.hits     = 1
        self.age      = 0        # frames since last successful match

        cx = (tlbr[0] + tlbr[2]) / 2.0
        cy = (tlbr[1] + tlbr[3]) / 2.0
        w  = float(tlbr[2] - tlbr[0])
        h  = float(tlbr[3] - tlbr[1])
        self._x, self._P = _KF.initiate(np.array([cx, cy, w, h]))

    # ── Kalman interface ───────────────────────────────────────────────

    def predict(self):
        self._x, self._P = _KF.predict(self._x, self._P)
        self.age += 1

    def update(self, tlbr: np.ndarray, score: float, cls_id: int):
        cx = (tlbr[0] + tlbr[2]) / 2.0
        cy = (tlbr[1] + tlbr[3]) / 2.0
        w  = float(tlbr[2] - tlbr[0])
        h  = float(tlbr[3] - tlbr[1])
        self._x, self._P = _KF.update(self._x, self._P,
                                       np.array([cx, cy, w, h]))
        self.score  = float(score)
        self.cls_id = int(cls_id)
        self.hits  += 1
        self.age    = 0
        self.state  = TrackState.TRACKED

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def tlbr(self) -> np.ndarray:
        """Return predicted [x1, y1, x2, y2]."""
        cx, cy, w, h = self._x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    @property
    def center(self) -> Tuple[float, float]:
        return float(self._x[0]), float(self._x[1])

    def __repr__(self):
        return (f"STrack(id={self.track_id}, "
                f"cls={self.cls_id}, hits={self.hits}, state={self.state})")


# ── Utilities ─────────────────────────────────────────────────────────────────

def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between two sets of boxes.
    boxes: shape (N, 4) in [x1, y1, x2, y2] format.
    Returns: (N, M) IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]   # N×1×4
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]   # 1×M×4

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union  = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def _linear_assign(cost: np.ndarray):
    """
    Optimal linear assignment (Hungarian algorithm).
    Falls back to greedy if scipy is not installed.
    Returns list of (row, col) pairs.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:
        pairs, mat = [], cost.copy()
        for _ in range(min(mat.shape)):
            i, j = np.unravel_index(np.argmin(mat), mat.shape)
            if mat[i, j] >= 1e9:
                break
            pairs.append((int(i), int(j)))
            mat[i, :] = 1e9
            mat[:, j] = 1e9
        return pairs


def _associate(detections, tracks, iou_thresh: float):
    """
    Match detections to tracks using IoU cost + linear assignment.

    Args:
        detections : list of (tlbr, score, cls_id)
        tracks     : list of STrack
        iou_thresh : minimum IoU for a valid match

    Returns:
        matches            : [(det_idx, trk_idx), ...]
        unmatched_det_idxs : [int, ...]
        unmatched_trk_idxs : [int, ...]
    """
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))

    det_boxes = np.array([d[0] for d in detections])
    trk_boxes = np.array([t.tlbr for t in tracks])
    iou       = iou_batch(det_boxes, trk_boxes)
    cost      = 1.0 - iou

    pairs = _linear_assign(cost)

    matched_d, matched_t = set(), set()
    matches = []
    for di, ti in pairs:
        if iou[di, ti] >= iou_thresh:
            matches.append((di, ti))
            matched_d.add(di)
            matched_t.add(ti)

    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    unmatched_t = [i for i in range(len(tracks))     if i not in matched_t]
    return matches, unmatched_d, unmatched_t


# ── ByteTracker ────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Two-stage association:
    1. High-confidence detections  → all active + recently-lost tracks  (IoU)
    2. Low-confidence  detections  → remaining unmatched active tracks  (IoU)
    Unmatched high-conf detections → new tracks.
    Tracks not seen for > max_age frames are removed.

    Parameters
    ──────────
    high_thresh   Detection score threshold for "high confidence"  (default 0.50)
    low_thresh    Minimum score to consider a detection at all     (default 0.10)
    iou_thresh    Minimum IoU required to accept an assignment     (default 0.30)
    max_age       Frames a lost track survives without a match     (default 30)
    min_hits      Detections before a track is output as confirmed (default 1)
    """

    def __init__(self,
                 high_thresh: float = 0.50,
                 low_thresh:  float = 0.10,
                 iou_thresh:  float = 0.30,
                 max_age:     int   = 30,
                 min_hits:    int   = 1):
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh
        self.iou_thresh  = iou_thresh
        self.max_age     = max_age
        self.min_hits    = min_hits

        self._active: List[STrack] = []   # confirmed + tracked
        self._lost:   List[STrack] = []   # temporarily missing
        STrack.reset_ids()

    # ── Public API ─────────────────────────────────────────────────────

    def update(self,
               boxes:   np.ndarray,
               scores:  np.ndarray,
               cls_ids: np.ndarray) -> List[STrack]:
        """
        Process one frame.

        Args:
            boxes   : (N, 4) float32 [x1, y1, x2, y2]
            scores  : (N,)   float32  detection confidence
            cls_ids : (N,)   int      class indices

        Returns:
            List of active STrack objects (track_id, tlbr, cls_id, score).
        """
        # Split detections by confidence
        all_dets  = list(zip(boxes, scores, cls_ids))
        high_dets = [(b, s, c) for b, s, c in all_dets if s >= self.high_thresh]
        low_dets  = [(b, s, c) for b, s, c in all_dets
                     if self.low_thresh <= s < self.high_thresh]

        # Predict all existing tracks forward
        for t in self._active + self._lost:
            t.predict()

        # ── Stage 1: high-conf ↔ active + lost ────────────────────────
        pool = self._active + self._lost
        m1, unm_d1, unm_t1 = _associate(high_dets, pool, self.iou_thresh)

        matched_high: List[STrack] = []
        for di, ti in m1:
            pool[ti].update(*high_dets[di])
            matched_high.append(pool[ti])

        # ── Stage 2: low-conf ↔ remaining active (not lost) ───────────
        rem_active = [pool[ti] for ti in unm_t1
                      if pool[ti].state != TrackState.LOST]
        m2, _, unm_t2 = _associate(low_dets, rem_active, self.iou_thresh)

        for di, ti in m2:
            rem_active[ti].update(*low_dets[di])
            matched_high.append(rem_active[ti])

        still_unmatched = [rem_active[ti] for ti in unm_t2]

        # ── New tracks from unmatched high-conf detections ─────────────
        new_tracks: List[STrack] = []
        for di in unm_d1:
            t = STrack(*high_dets[di])
            t.state = TrackState.TRACKED
            new_tracks.append(t)

        # ── Move unmatched tracks to lost / remove old ones ────────────
        new_lost: List[STrack] = []
        for t in still_unmatched:
            t.state = TrackState.LOST
            if t.age <= self.max_age:
                new_lost.append(t)

        # lost tracks from pool that weren't in active (already lost)
        for ti in unm_t1:
            t = pool[ti]
            if t.state == TrackState.LOST:
                t.age += 1
                if t.age <= self.max_age:
                    new_lost.append(t)

        # ── Update collections ─────────────────────────────────────────
        active_set = set(id(t) for t in matched_high)
        self._active = [t for t in matched_high + new_tracks
                        if t.state == TrackState.TRACKED
                        and t.hits >= self.min_hits]
        self._lost   = new_lost

        return self._active
