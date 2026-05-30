import cv2
import math
import os
import numpy as np
from collections import deque
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path) if YOLO else None
        if self.model and HAS_CUDA:
            self.model.to('cuda')
        # COCO classes for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]

        # BEV Calibration
        self.LANE_WIDTH_M   = 3.75
        self.VISIBLE_LENGTH_M = 60
        self.BEV_SCALE      = 18

        self.SRC_ROAD_L = np.float32([
            [130, 390], [415, 390], [680, 720], [-70, 720]
        ])
        self.SRC_ROAD_R = np.float32([
            [415, 390], [610, 390], [960, 720], [680, 720]
        ])

        self.ML, self.bev_wL, self.bev_hL = self._build_bev_matrix(self.SRC_ROAD_L, self.LANE_WIDTH_M * 5)
        self.MR, self.bev_wR, self.bev_hR = self._build_bev_matrix(self.SRC_ROAD_R, self.LANE_WIDTH_M * 3)

    # ── helpers ──────────────────────────────────────────────────────

    def _build_bev_matrix(self, src_pts, road_width_m):
        bev_w = int(road_width_m * self.BEV_SCALE)
        bev_h = int(self.VISIBLE_LENGTH_M * self.BEV_SCALE)
        dst   = np.float32([[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]])
        M     = cv2.getPerspectiveTransform(src_pts, dst)
        return M, bev_w, bev_h

    def _to_bev(self, M, pt):
        p = np.float32([[pt[0], pt[1]]]).reshape(-1, 1, 2)
        t = cv2.perspectiveTransform(p, M)
        return float(t[0, 0, 0]), float(t[0, 0, 1])

    def _get_speed_color(self, kph):
        if kph < 40:  return (0, 255, 0)
        if kph < 100: return (0, 255, 255)
        return (0, 0, 255)

    @staticmethod
    def _congestion_level(current_on_screen: int) -> str:
        """Classify congestion from the number of vehicles visible in current frame."""
        if current_on_screen > 20:  return "GRIDLOCK"
        if current_on_screen > 12:  return "HEAVY"
        if current_on_screen > 5:   return "MODERATE"
        return "FREE FLOW"

    def _sync_to_traffic(self, video_obj, vehicles, speed, level):
        """Push vision analysis results into the main traffic system."""
        from apps.traffic.models import TrafficReading, CongestionLevel
        if not video_obj or not video_obj.location:
            return

        level_map = {
            "FREE FLOW": CongestionLevel.FREE_FLOW,
            "MODERATE":  CongestionLevel.MODERATE,
            "HEAVY":     CongestionLevel.HEAVY,
            "GRIDLOCK":  CongestionLevel.GRIDLOCK,
        }
        TrafficReading.objects.create(
            location         = video_obj.location,
            vehicle_count    = vehicles,
            avg_speed        = speed,
            congestion_index = min(vehicles * 2, 100),
            congestion_level = level_map.get(level, CongestionLevel.FREE_FLOW),
            source           = 'vision',
        )

    # ── background processing (non-streaming) ────────────────────────

    def process_video(self, video_path, video_obj=None):
        """
        Process an entire video file in a background thread.
        Saves statistics to video_obj and syncs readings to the traffic DB.
        Does NOT yield frames — call this from a thread, not a view.
        """
        if not self.model:
            if video_obj:
                video_obj.status = 'failed'
                video_obj.save(update_fields=['status'])
            return

        cap          = cv2.VideoCapture(video_path)
        frame_count  = 0
        counted_ids  = set()
        total_vehicles = 0
        speed_history  = {}
        HISTORY_LEN  = 10
        total_speed  = 0.0
        speed_count  = 0
        level        = "FREE FLOW"
        STRIDE       = 2
        IMGSZ        = 320

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        line_y = int(height * 0.6)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % STRIDE != 0:
                continue

            results = self.model.track(
                frame, classes=self.vehicle_classes,
                persist=True, verbose=False, imgsz=IMGSZ
            )
            result = results[0]

            current_on_screen = 0
            if result.boxes and result.boxes.id is not None:
                boxes     = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()
                current_on_screen = len(track_ids)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int(y2)

                    mid   = width / 2
                    M     = self.ML if cx < mid else self.MR
                    bev_pt = self._to_bev(M, (cx, cy))

                    if track_id not in speed_history:
                        speed_history[track_id] = deque(maxlen=HISTORY_LEN)
                    speed_history[track_id].append(bev_pt)

                    if len(speed_history[track_id]) >= 2:
                        hist   = speed_history[track_id]
                        dx     = hist[-1][0] - hist[0][0]
                        dy     = hist[-1][1] - hist[0][1]
                        dist_m = math.hypot(dx, dy) / self.BEV_SCALE
                        dt_s   = (len(hist) - 1) * STRIDE / fps
                        kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                        if kph > 0:
                            total_speed += kph
                            speed_count += 1

                    cy_mid = (y1 + y2) / 2
                    if track_id not in counted_ids and abs(cy_mid - line_y) < 15:
                        counted_ids.add(track_id)
                        total_vehicles += 1

            level     = self._congestion_level(current_on_screen)
            avg_speed = total_speed / speed_count if speed_count > 0 else 0.0

            # Persist stats every 60 processed frames to reduce SQLite write contention
            if video_obj and frame_count % 60 == 0:
                try:
                    video_obj.vehicle_count              = total_vehicles
                    video_obj.predicted_congestion_level = level
                    video_obj.average_speed              = round(avg_speed, 1)
                    video_obj.save(update_fields=[
                        'vehicle_count', 'predicted_congestion_level', 'average_speed'
                    ])
                    if video_obj.location:
                        self._sync_to_traffic(video_obj, total_vehicles, avg_speed, level)
                except Exception:
                    pass  # SQLite lock; next save will catch up

        cap.release()

        if video_obj:
            avg_speed = total_speed / speed_count if speed_count > 0 else 0.0
            video_obj.vehicle_count              = total_vehicles
            video_obj.average_speed              = round(avg_speed, 1)
            video_obj.predicted_congestion_level = level
            video_obj.status                     = 'completed'
            video_obj.save()
            if video_obj.location:
                self._sync_to_traffic(video_obj, total_vehicles, avg_speed, level)

    # ── MJPEG streaming (live view) ───────────────────────────────────

    def stream_inference(self, video_path, output_video_obj=None):
        """Yield annotated MJPEG frames with YOLO overlays for live browser viewing."""
        if not self.model:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            return

        cap          = cv2.VideoCapture(video_path)
        frame_count  = 0
        counted_ids  = set()
        total_vehicles = 0
        speed_history  = {}
        HISTORY_LEN  = 10
        total_speed  = 0.0
        speed_count  = 0
        level        = "FREE FLOW"
        STRIDE       = 2
        IMGSZ        = 320

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        line_y = int(height * 0.6)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            if frame_count % STRIDE != 0:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes() + b'\r\n')
                continue

            results = self.model.track(
                frame, classes=self.vehicle_classes,
                persist=True, verbose=False, imgsz=IMGSZ
            )
            result          = results[0]
            annotated_frame = frame.copy()
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 0, 255), 2)

            current_on_screen = 0
            if result.boxes and result.boxes.id is not None:
                boxes     = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()
                current_on_screen = len(track_ids)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int(y2)

                    mid    = width / 2
                    M      = self.ML if cx < mid else self.MR
                    bev_pt = self._to_bev(M, (cx, cy))

                    if track_id not in speed_history:
                        speed_history[track_id] = deque(maxlen=HISTORY_LEN)
                    speed_history[track_id].append(bev_pt)

                    kph = 0.0
                    if len(speed_history[track_id]) >= 2:
                        hist   = speed_history[track_id]
                        dx     = hist[-1][0] - hist[0][0]
                        dy     = hist[-1][1] - hist[0][1]
                        dist_m = math.hypot(dx, dy) / self.BEV_SCALE
                        dt_s   = (len(hist) - 1) * STRIDE / fps
                        kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                        if kph > 0:
                            total_speed += kph
                            speed_count += 1

                    color      = self._get_speed_color(kph)
                    class_name = self.model.names[cls_id]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"#{track_id} {class_name} {kph:.0f}km/h",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    cy_mid = (y1 + y2) / 2
                    if track_id not in counted_ids and abs(cy_mid - line_y) < 15:
                        counted_ids.add(track_id)
                        total_vehicles += 1

            level     = self._congestion_level(current_on_screen)
            avg_speed = total_speed / speed_count if speed_count > 0 else 0.0

            # HUD overlay
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Vehicles: {total_vehicles}",
                        (20, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Avg Speed: {avg_speed:.1f} km/h",
                        (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Status: {level}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if output_video_obj and frame_count % 10 == 0:
                try:
                    output_video_obj.vehicle_count              = total_vehicles
                    output_video_obj.predicted_congestion_level = level
                    output_video_obj.average_speed              = round(avg_speed, 1)
                    output_video_obj.save(update_fields=[
                        'vehicle_count', 'predicted_congestion_level', 'average_speed'
                    ])
                except Exception:
                    pass

            if output_video_obj and frame_count % 30 == 0 and output_video_obj.location:
                self._sync_to_traffic(output_video_obj, total_vehicles, avg_speed, level)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')

        cap.release()

        if output_video_obj:
            output_video_obj.status = 'completed'
            output_video_obj.save(update_fields=['status'])
            if output_video_obj.location:
                self._sync_to_traffic(output_video_obj, total_vehicles, avg_speed, level)
