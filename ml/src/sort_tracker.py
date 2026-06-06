"""
sort_tracker.py  —  SORT (Simple Online and Realtime Tracking)
Based on: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.

Kalman filter + Hungarian assignment per frame.
No neural-network feature extractor required — works on CPU.

Public API
──────────
    tracker = SortTracker(min_hits=3, max_age=30, iou_thresh=0.30)
    tracker.reset()                         # call between videos/sessions
    tracks = tracker.update(detections)     # detections: (N, 6) ndarray
                                            # columns: x1,y1,x2,y2,score,cls_id
    for t in tracks:                        # only confirmed tracks returned
        print(t.track_id, t.tlbr, t.cls_id, t.score)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter for one bounding box.

    State    (8): [cx, cy, w, h,  vcx, vcy, vw, vh]
    Measurement (4): [cx, cy, w, h]

    Noise is scaled by the box dimensions so that large, fast vehicles and
    small slow ones are modelled with the same relative uncertainty.
    """

    # Shared transition and observation matrices
    _F = np.eye(8, dtype=np.float64)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0
    _H = np.eye(4, 8, dtype=np.float64)

    def __init__(self, tlbr: np.ndarray):
        x1, y1, x2, y2 = tlbr
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = float(x2 - x1)
        h  = float(y2 - y1)

        # State: position known, velocity unknown
        self._x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)

        # Initial covariance — very uncertain about velocity
        self._P = np.diag([
            (w * 0.10) ** 2, (h * 0.10) ** 2,
            (w * 0.10) ** 2, (h * 0.10) ** 2,
            (w * 1.00) ** 2, (h * 1.00) ** 2,
            (w * 1.00) ** 2, (h * 1.00) ** 2,
        ])

        # Process noise (per-step uncertainty) — scale with box size
        self._q = np.array([
            w * 0.01, h * 0.01, w * 0.01, h * 0.01,
            w * 0.05, h * 0.05, w * 0.05, h * 0.05,
        ], dtype=np.float64)

        # Measurement noise — YOLO localization error ~ 5% of box dims
        self._r = np.array(
            [w * 0.05, h * 0.05, w * 0.05, h * 0.05], dtype=np.float64
        )

    # ── Kalman steps ──────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Advance one frame. Returns predicted [cx, cy, w, h]."""
        Q = np.diag(self._q ** 2)
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + Q
        # Keep w, h ≥ 1 to avoid degenerate boxes
        self._x[2] = max(1.0, self._x[2])
        self._x[3] = max(1.0, self._x[3])
        return self._x[:4].copy()

    def update(self, tlbr: np.ndarray):
        """Update with a new detected bounding box [x1,y1,x2,y2]."""
        x1, y1, x2, y2 = tlbr
        z = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2,
             float(x2 - x1), float(y2 - y1)],
            dtype=np.float64,
        )
        R = np.diag(self._r ** 2)
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._H @ self._x)
        self._P = (np.eye(8) - K @ self._H) @ self._P

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def tlbr(self) -> np.ndarray:
        """Predicted [x1, y1, x2, y2] (float)."""
        cx, cy, w, h = self._x[:4]
        return np.array([cx - w / 2, cy - h / 2,
                         cx + w / 2, cy + h / 2], dtype=np.float64)

    @property
    def center(self) -> Tuple[float, float]:
        return float(self._x[0]), float(self._x[1])


# ── Track object ──────────────────────────────────────────────────────────────

class Track:
    """
    A single tracked vehicle.

    Lifecycle
    ─────────
    NEW        first detection
    CONFIRMED  seen in ≥ min_hits consecutive frames
    LOST       unmatched but kept alive for max_age frames
    """

    # Instance-level counter — reset via SortTracker.reset()
    _counter: int = 0

    def __init__(self, tlbr: np.ndarray, score: float, cls_id: int):
        Track._counter += 1
        self.track_id  = Track._counter
        self.kf        = KalmanBoxFilter(tlbr)
        self.score     = float(score)
        self.cls_id    = int(cls_id)
        self.hits      = 1    # consecutive matched frames
        self.age       = 0    # frames since last successful match
        self.confirmed = False

    def predict(self):
        """Kalman predict + increment age."""
        self.kf.predict()
        self.age += 1

    def match(self, tlbr: np.ndarray, score: float, cls_id: int):
        """Kalman update after a successful detection match."""
        self.kf.update(tlbr)
        self.score  = float(score)
        self.cls_id = int(cls_id)
        self.hits  += 1
        self.age    = 0

    @property
    def tlbr(self) -> np.ndarray:
        return self.kf.tlbr

    @property
    def center(self) -> Tuple[float, float]:
        return self.kf.center

    def __repr__(self):
        return f"Track(id={self.track_id}, cls={self.cls_id}, hits={self.hits})"


# ── Assignment utilities ──────────────────────────────────────────────────────

def _iou_matrix(det_boxes: np.ndarray, trk_boxes: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between N detections and M tracks.
    Returns (N, M) matrix.  All boxes in [x1, y1, x2, y2] format.
    """
    if det_boxes.shape[0] == 0 or trk_boxes.shape[0] == 0:
        return np.zeros((det_boxes.shape[0], trk_boxes.shape[0]), dtype=np.float32)

    d = det_boxes[:, None, :]   # N × 1 × 4
    t = trk_boxes[None, :, :]   # 1 × M × 4

    ix1 = np.maximum(d[..., 0], t[..., 0])
    iy1 = np.maximum(d[..., 1], t[..., 1])
    ix2 = np.minimum(d[..., 2], t[..., 2])
    iy2 = np.minimum(d[..., 3], t[..., 3])

    inter  = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    union  = area_d + area_t - inter

    return (inter / np.maximum(union, 1e-8)).astype(np.float32)


def _assign(
    iou_mat: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Optimal linear assignment using scipy if available, greedy otherwise.

    Returns
    ───────
    matches        : list of (det_idx, trk_idx) with IoU ≥ thresh
    unmatched_dets : detection indices with no valid match
    unmatched_trks : track indices with no valid match
    """
    n_dets, n_trks = iou_mat.shape
    if n_dets == 0 or n_trks == 0:
        return [], list(range(n_dets)), list(range(n_trks))

    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-iou_mat)
        pairs  = list(zip(ri.tolist(), ci.tolist()))
    except ImportError:
        # Greedy fallback — O(min(N,M)²) but acceptable for N≤50
        pairs, work = [], iou_mat.copy()
        for _ in range(min(n_dets, n_trks)):
            r, c = np.unravel_index(np.argmax(work), work.shape)
            if work[r, c] < thresh:
                break
            pairs.append((int(r), int(c)))
            work[r, :] = -1.0
            work[:, c] = -1.0

    matched_d, matched_t = set(), set()
    matches = []
    for r, c in pairs:
        if iou_mat[r, c] >= thresh:
            matches.append((r, c))
            matched_d.add(r)
            matched_t.add(c)

    unmatched_dets = [i for i in range(n_dets) if i not in matched_d]
    unmatched_trks = [i for i in range(n_trks) if i not in matched_t]
    return matches, unmatched_dets, unmatched_trks


# ── SORT Tracker ──────────────────────────────────────────────────────────────

class SortTracker:
    """
    SORT multi-object tracker.

    Parameters
    ──────────
    min_hits   Frames a track must be seen before being returned as confirmed.
               Setting this to 3 suppresses single-frame false positives.
    max_age    Frames a track may go undetected before being deleted.
               Allows tracks to survive brief occlusions.
    iou_thresh Minimum IoU required to match a detection to an existing track.

    Usage
    ─────
        tracker = SortTracker(min_hits=3, max_age=30)
        tracker.reset()               # call once per video/session
        tracks = tracker.update(dets) # dets: (N,6) [x1,y1,x2,y2,score,cls_id]
        for t in tracks:              # only confirmed tracks
            ...
    """

    def __init__(self,
                 min_hits:   int   = 3,
                 max_age:    int   = 30,
                 iou_thresh: float = 0.30):
        self.min_hits  = min_hits
        self.max_age   = max_age
        self.iou_thresh= iou_thresh
        self._tracks: List[Track] = []

    def reset(self):
        """Clear all tracks and reset the ID counter. Call between videos."""
        self._tracks.clear()
        Track._counter = 0

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Process one frame.

        Args:
            detections: ndarray shape (N, 6), columns [x1,y1,x2,y2,score,cls_id].
                        Pass np.empty((0, 6)) when there are no detections.

        Returns:
            List of confirmed Track objects (hits ≥ min_hits, age == 0).
        """
        # 1. Kalman predict all tracks forward one frame
        for t in self._tracks:
            t.predict()

        if len(detections) == 0:
            # Age out dead tracks, keep rest as lost
            self._tracks = [t for t in self._tracks if t.age <= self.max_age]
            return [t for t in self._tracks if t.confirmed]

        det_boxes = detections[:, :4]
        trk_boxes = np.array([t.tlbr for t in self._tracks]) if self._tracks \
                    else np.empty((0, 4))

        # 2. Match detections to existing tracks via IoU
        iou     = _iou_matrix(det_boxes, trk_boxes)
        matches, unm_dets, unm_trks = _assign(iou, self.iou_thresh)

        # 3. Update matched tracks
        for di, ti in matches:
            self._tracks[ti].match(
                detections[di, :4],
                float(detections[di, 4]),
                int(detections[di, 5]),
            )

        # 4. Create new tracks for unmatched detections
        for di in unm_dets:
            self._tracks.append(Track(
                detections[di, :4],
                float(detections[di, 4]),
                int(detections[di, 5]),
            ))

        # 5. Remove tracks that have been lost too long
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        # 6. Mark confirmed tracks (hit streak ≥ min_hits)
        for t in self._tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True

        # 7. Return only confirmed tracks
        return [t for t in self._tracks if t.confirmed]
