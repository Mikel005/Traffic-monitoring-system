"""
run_detector — CLI management command for vehicle detection & counting.

Usage examples
──────────────
  # Process a recorded video file
  python manage.py run_detector --video media/videos/footage.mp4

  # Process a live RTSP camera stream
  python manage.py run_detector --camera rtsp://user:pass@192.168.1.10/stream1

  # Process with a specific location and save CSV
  python manage.py run_detector --video footage.mp4 --location 3 --csv output.csv

  # Show a live OpenCV window while processing (requires a display)
  python manage.py run_detector --video footage.mp4 --display

  # Use a specific YOLO model
  python manage.py run_detector --video footage.mp4 --model ml/saved_models/yolov8n.pt
"""

import sys
import os
import csv
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = "Detect, track, and count vehicles from a video file or camera stream."

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--video", "-v", type=str,
            help="Path to a recorded video file (MP4, AVI, MOV, MKV …).",
        )
        group.add_argument(
            "--camera", "-c", type=str,
            help="Camera source: RTSP URL, HTTP stream URL, or integer device index.",
        )
        parser.add_argument(
            "--location", "-l", type=int, default=None,
            help="Location ID to associate counts with (optional).",
        )
        parser.add_argument(
            "--model", "-m", type=str, default=None,
            help="Path to YOLO model weights. Defaults to YOLO_WEIGHTS_PATH in settings.",
        )
        parser.add_argument(
            "--csv", type=str, default=None,
            help="Path for the output CSV file. Auto-generated if omitted.",
        )
        parser.add_argument(
            "--display", "-d", action="store_true",
            help="Show an OpenCV preview window (requires a local display).",
        )
        parser.add_argument(
            "--conf", type=float, default=0.40,
            help="Detection confidence threshold (default 0.40).",
        )

    def handle(self, *args, **options):
        # ── Imports inside handle() so Django is fully initialised ────
        import cv2
        import numpy as np
        sys.path.insert(0, str(Path(settings.BASE_DIR)))

        from ml.src.vehicle_detector import VehicleDetector, VEHICLE_CLASS_MAP
        from ml.src.byte_tracker import ByteTracker, STrack
        import math, time
        from collections import deque
        from datetime import datetime

        # ── Resolve source ────────────────────────────────────────────
        video_src = options["video"]
        cam_src   = options["camera"]
        source    = video_src or cam_src

        if video_src and not Path(video_src).exists():
            raise CommandError(f"Video file not found: {video_src}")

        if cam_src and cam_src.isdigit():
            source = int(cam_src)

        # ── Model ─────────────────────────────────────────────────────
        model_path = options["model"] or str(settings.YOLO_WEIGHTS_PATH)
        self.stdout.write(f"Loading model: {model_path}")
        detector = VehicleDetector(model_path=model_path)
        if not detector.model:
            raise CommandError("YOLO model could not be loaded. Check YOLO_WEIGHTS_PATH.")

        # ── Location ─────────────────────────────────────────────────
        location = None
        if options["location"]:
            from apps.traffic.models import Location
            try:
                location = Location.objects.get(pk=options["location"])
                self.stdout.write(f"Location: {location.name}")
            except Location.DoesNotExist:
                self.stdout.write(self.style.WARNING(
                    f"Location {options['location']} not found — continuing without."
                ))

        # ── CSV output ────────────────────────────────────────────────
        if options["csv"]:
            csv_path = Path(options["csv"])
        else:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = Path(settings.MEDIA_ROOT) / "counts" / f"cli_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout.write(f"CSV output  : {csv_path}")

        # ── Open capture ──────────────────────────────────────────────
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise CommandError(f"Cannot open video source: {source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        proc_w = 640
        proc_h = int(orig_h * (proc_w / orig_w))
        line_y = int(proc_h * 0.60)
        mid    = proc_w / 2.0

        self.stdout.write(
            f"Source      : {source}\n"
            f"Resolution  : {orig_w}×{orig_h}  →  {proc_w}×{proc_h}\n"
            f"FPS         : {fps:.1f}\n"
            f"Counting line at y={line_y} ({int(line_y/proc_h*100)}% of height)\n"
            f"Tracking    : {'ByteTrack (ultralytics)' if detector.use_native_track else 'ByteTracker (Kalman + IoU)'}\n"
        )

        # ── Detection & tracking state ────────────────────────────────
        from ml.src.byte_tracker import ByteTracker
        bt            = ByteTracker()
        counted_ids   = set()
        speed_history : dict = {}
        prev_side     : dict = {}
        cc            : dict = {}
        inbound       = 0
        outbound      = 0
        total_speed   = 0.0
        speed_count   = 0
        frame_n       = 0
        STRIDE        = 2
        IMGSZ         = 320
        CONF          = options["conf"]
        MAX_DET       = 50

        # CSV header
        csv_fields = ["timestamp","track_id","vehicle_type","direction","speed_kph","frame"]
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        self.stdout.write(self.style.SUCCESS("Processing…  (Ctrl-C to stop early)"))
        t_start = time.time()

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_n += 1
                if frame_n % STRIDE != 0:
                    continue

                small = cv2.resize(frame, (proc_w, proc_h))

                # ── Detect ────────────────────────────────────────────
                if detector.use_native_track:
                    results = detector.model.track(
                        small, classes=detector.vehicle_classes,
                        persist=True, verbose=False,
                        imgsz=IMGSZ, conf=CONF, max_det=MAX_DET,
                        half=False, tracker="bytetrack.yaml",
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
                        for box, tid, cid in zip(boxes, tids, cids):
                            tracks.append((tuple(box.tolist()), int(tid), int(cid)))
                else:
                    results = detector.model.predict(
                        small, classes=detector.vehicle_classes,
                        verbose=False, imgsz=IMGSZ, conf=CONF,
                        max_det=MAX_DET, half=False,
                    )
                    res  = results[0]
                    dets = []
                    if res.boxes is not None and len(res.boxes):
                        boxes  = res.boxes.xyxy.cpu().numpy()
                        cids   = res.boxes.cls.cpu().numpy().astype(int)
                        scores = res.boxes.conf.cpu().numpy()
                        for box, cid, sc in zip(boxes, cids, scores):
                            dets.append((box, float(sc), int(cid)))
                    bt_b = np.array([d[0] for d in dets],dtype=np.float32) if dets else np.empty((0,4))
                    bt_s = np.array([d[1] for d in dets],dtype=np.float32) if dets else np.empty((0,))
                    bt_c = np.array([d[2] for d in dets],dtype=np.int32)   if dets else np.empty((0,),dtype=np.int32)
                    active = bt.update(bt_b, bt_s, bt_c)
                    tracks = [(tuple(t.tlbr.astype(int).tolist()), t.track_id, t.cls_id) for t in active]

                # ── Per-track processing ──────────────────────────────
                if options["display"]:
                    display = small.copy()
                    cv2.line(display, (0,line_y), (proc_w,line_y), (0,200,255), 2)

                for (x1,y1,x2,y2), tid, cid in tracks:
                    cx, cy = (x1+x2)//2, y2
                    M      = detector.ML if cx < mid else detector.MR
                    bev_pt = detector._to_bev(M, cx, cy)

                    if tid not in speed_history:
                        speed_history[tid] = deque(maxlen=12)
                    speed_history[tid].append(bev_pt)
                    kph = 0.0
                    hist = speed_history[tid]
                    if len(hist) >= 3:
                        dx     = hist[-1][0] - hist[0][0]
                        dy     = hist[-1][1] - hist[0][1]
                        dist_m = math.hypot(dx, dy) / detector.BEV_SCALE
                        dt_s   = (len(hist) - 1) * STRIDE / fps
                        kph    = (dist_m / dt_s * 3.6) if dt_s > 0 else 0.0
                        if 1.0 < kph < 200.0:
                            total_speed += kph
                            speed_count += 1

                    # Direction + count
                    cy_c  = (y1 + y2) / 2.0
                    side  = "above" if cy_c < line_y else "below"
                    p_side = prev_side.get(tid)
                    if p_side and p_side != side and tid not in counted_ids:
                        counted_ids.add(tid)
                        lbl  = VEHICLE_CLASS_MAP.get(cid, "car")
                        cc[lbl] = cc.get(lbl, 0) + 1
                        direction = "INBOUND" if p_side == "above" else "OUTBOUND"
                        if direction == "INBOUND": inbound  += 1
                        else:                      outbound += 1
                        csv_writer.writerow({
                            "timestamp":    datetime.now().isoformat(timespec="seconds"),
                            "track_id":     tid,
                            "vehicle_type": lbl,
                            "direction":    direction,
                            "speed_kph":    round(kph, 1),
                            "frame":        frame_n,
                        })
                        self.stdout.write(
                            f"  [{datetime.now().strftime('%H:%M:%S')}] "
                            f"ID#{tid:<4} {lbl:<12} {direction}  "
                            f"{kph:5.1f} km/h  "
                            f"total={sum(cc.values())}"
                        )
                    prev_side[tid] = side

                    if options["display"]:
                        color = (0,220,0) if kph < 40 else (0,220,220) if kph < 100 else (0,60,255)
                        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(display,
                                    f"#{tid} {VEHICLE_CLASS_MAP.get(cid,'?')} {kph:.0f}",
                                    (x1, max(y1-6,12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                # ── Display ───────────────────────────────────────────
                if options["display"]:
                    total_v = sum(cc.values())
                    avg_v   = total_speed/speed_count if speed_count else 0.0
                    cv2.rectangle(display, (8,8), (420,130), (0,0,0), -1)
                    cv2.putText(display, f"Total: {total_v}  IN:{inbound}  OUT:{outbound}",
                                (14,36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(display, f"C:{cc.get('car',0)} T:{cc.get('truck',0)} B:{cc.get('bus',0)} M:{cc.get('motorcycle',0)}",
                                (14,66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,255), 1, cv2.LINE_AA)
                    cv2.putText(display, f"Avg speed: {avg_v:.1f} km/h",
                                (14,96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,0), 1, cv2.LINE_AA)
                    cv2.imshow("TrafficIQ — Vehicle Detector", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            self.stdout.write("\nStopped by user.")
        finally:
            cap.release()
            csv_file.close()
            if options["display"]:
                cv2.destroyAllWindows()

        # ── Summary ───────────────────────────────────────────────────
        elapsed   = time.time() - t_start
        total_v   = sum(cc.values())
        avg_speed = total_speed / speed_count if speed_count else 0.0

        self.stdout.write("\n" + "─" * 50)
        self.stdout.write(self.style.SUCCESS("SUMMARY"))
        self.stdout.write(f"  Duration     : {elapsed:.1f}s  ({frame_n} frames @ {frame_n/elapsed:.1f} fps effective)")
        self.stdout.write(f"  Total vehicles: {total_v}")
        self.stdout.write(f"  Cars          : {cc.get('car', 0)}")
        self.stdout.write(f"  Trucks        : {cc.get('truck', 0)}")
        self.stdout.write(f"  Buses         : {cc.get('bus', 0)}")
        self.stdout.write(f"  Motorcycles   : {cc.get('motorcycle', 0)}")
        self.stdout.write(f"  Inbound       : {inbound}")
        self.stdout.write(f"  Outbound      : {outbound}")
        self.stdout.write(f"  Avg speed     : {avg_speed:.1f} km/h")
        self.stdout.write(f"  CSV saved     : {csv_path}")

        # ── Save session to DB ────────────────────────────────────────
        try:
            from apps.vision.models import VehicleCountSession
            from django.utils import timezone
            session_tag = f"cli_{int(t_start)}"
            VehicleCountSession.objects.update_or_create(
                session_tag=session_tag,
                defaults=dict(
                    location         = location,
                    total_count      = total_v,
                    car_count        = cc.get("car", 0),
                    truck_count      = cc.get("truck", 0),
                    bus_count        = cc.get("bus", 0),
                    motorcycle_count = cc.get("motorcycle", 0),
                    inbound_count    = inbound,
                    outbound_count   = outbound,
                    avg_speed        = round(avg_speed, 1),
                    ended_at         = timezone.now(),
                ),
            )
            self.stdout.write(self.style.SUCCESS("Session saved to database."))
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"DB save skipped: {exc}"))
