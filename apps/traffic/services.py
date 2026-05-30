"""
Services implementing UML controller classes:
  - SystemController
  - TrafficProcessor
  - DatabaseManager
"""
import random
import math
from datetime import timedelta
from django.utils import timezone
from django.db.models import Avg, Max, Min, Count

from apps.traffic.models import (
    Location, TrafficReading, CongestionLevel, Report, AlternativeRoute
)


# ─────────────────────────────────────────────────────────────────
# SystemController  (UML: + validateUser, + storeRawData,
#                         + predictCongestion, + updatePredictiveFocus,
#                         + createCameraFacade, + storeMwData …)
# ─────────────────────────────────────────────────────────────────
class SystemController:

    @staticmethod
    def validate_user(user):
        """UML: + validateUser()  — checks role and active status."""
        return user is not None and user.is_authenticated and user.is_active

    @staticmethod
    def store_raw_data(reading_dict: dict) -> TrafficReading:
        """
        UML: + storeRawData()
        Persists a raw reading dict to TrafficReading.
        """
        location = Location.objects.get(pk=reading_dict['location_id'])
        idx = reading_dict.get('congestion_index', 0.0)
        reading = TrafficReading.objects.create(
            location         = location,
            vehicle_count    = reading_dict.get('vehicle_count', 0),
            car_count        = reading_dict.get('car_count', 0),
            truck_count      = reading_dict.get('truck_count', 0),
            motorcycle_count = reading_dict.get('motorcycle_count', 0),
            bus_count        = reading_dict.get('bus_count', 0),
            avg_speed        = reading_dict.get('avg_speed', 0.0),
            free_flow_speed  = reading_dict.get('free_flow_speed', 60.0),
            queue_length     = reading_dict.get('queue_length', 0.0),
            congestion_index = idx,
            congestion_level = TrafficReading.index_to_level(idx),
            rainfall_mm      = reading_dict.get('rainfall_mm', 0.0),
            temperature_c    = reading_dict.get('temperature_c', 28.0),
            source           = reading_dict.get('source', 'mock'),
        )
        # Trigger alert if gridlock/heavy
        if reading.congestion_level in (CongestionLevel.HEAVY, CongestionLevel.GRIDLOCK):
            SystemController._raise_alert(reading)
        return reading

    @staticmethod
    def _raise_alert(reading: TrafficReading):
        """Internal: create an alert via NotificationService."""
        from apps.alerts.services import NotificationService
        NotificationService.send_alert(reading)

    @staticmethod
    def predict_congestion(location: Location) -> dict:
        """UML: + predictCongestion()  — delegates to TrafficProcessor."""
        return TrafficProcessor.predict_congestion_for(location)

    @staticmethod
    def update_predictive_focus(location: Location):
        """UML: + updatePredictiveFocus()  — stores a Prediction record."""
        from apps.predictions.models import Prediction
        from apps.predictions.ml import CongestionPredictor
        pred = CongestionPredictor.predict(location)
        Prediction.objects.create(
            location      = location,
            pred_15min    = pred.get('minutes_15'),
            pred_30min    = pred.get('minutes_30'),
            pred_60min    = pred.get('minutes_60'),
            model_version = pred.get('model', 'rule_based'),
            confidence    = pred.get('confidence', 0.7),
        )
        return pred

    @staticmethod
    def create_camera_facade(camera):
        """UML: + createCameraFacade()  — returns camera capture data."""
        return camera.capture_video()

    @staticmethod
    def store_mw_data(data: dict):
        """UML: + storeMwData()  — middleware data ingestion point."""
        return SystemController.store_raw_data(data)

    @staticmethod
    def generate_mock_reading(location: Location) -> dict:
        """Generates realistic mock sensor data for a location."""
        hour = timezone.now().hour
        # Enugu rush hours: 7-9 AM, 5-8 PM
        is_rush = (7 <= hour <= 9) or (17 <= hour <= 20)
        base_vehicles = random.randint(80, 160) if is_rush else random.randint(20, 80)
        avg_speed     = random.uniform(5, 25) if is_rush else random.uniform(30, 70)
        congestion    = min(100, max(0, (1 - avg_speed / location.speed_limit) * 100
                         + random.gauss(0, 5)))
        return {
            'location_id':    location.pk,
            'vehicle_count':  base_vehicles,
            'car_count':      int(base_vehicles * 0.60),
            'truck_count':    int(base_vehicles * 0.10),
            'motorcycle_count': int(base_vehicles * 0.20),
            'bus_count':      int(base_vehicles * 0.10),
            'avg_speed':      round(max(2, avg_speed), 1),
            'free_flow_speed': float(location.speed_limit),
            'queue_length':   round(random.uniform(0, 500) if is_rush else 0, 1),
            'congestion_index': round(congestion, 1),
            'rainfall_mm':    round(random.uniform(0, 5) if random.random() < 0.2 else 0, 2),
            'temperature_c':  round(random.uniform(25, 38), 1),
            'source':         'mock',
        }


# ─────────────────────────────────────────────────────────────────
# TrafficProcessor  (UML: + detectVehicles, + analyzeTrafficData,
#                         + predictCongestion, + calculateAlternativeRoutes)
# ─────────────────────────────────────────────────────────────────
class TrafficProcessor:

    @staticmethod
    def detect_vehicles(image_path: str) -> dict:
        """
        UML: + detectVehicles()
        Uses OpenCV to count vehicles in an image/frame.
        Falls back to mock count if CV not available.
        """
        try:
            import cv2
            import numpy as np
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not load image")
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vehicle_contours = [c for c in contours if cv2.contourArea(c) > 500]
            return {
                'vehicle_count': len(vehicle_contours),
                'method': 'opencv_contour',
                'image_path': str(image_path),
            }
        except Exception:
            return {
                'vehicle_count': random.randint(5, 50),
                'method': 'mock',
                'image_path': str(image_path),
            }

    @staticmethod
    def analyze_traffic_data(location: Location, hours: int = 1) -> dict:
        """
        UML: + analyzeTrafficData()
        Returns aggregated stats for a location over the last `hours` hours.
        """
        since = timezone.now() - timedelta(hours=hours)
        qs    = TrafficReading.objects.filter(location=location, timestamp__gte=since)
        agg   = qs.aggregate(
            avg_speed        = Avg('avg_speed'),
            avg_congestion   = Avg('congestion_index'),
            max_congestion   = Max('congestion_index'),
            min_congestion   = Min('congestion_index'),
            avg_vehicles     = Avg('vehicle_count'),
            total_readings   = Count('id'),
        )
        return {
            'location':       location.name,
            'period_hours':   hours,
            'readings_count': agg['total_readings'] or 0,
            'avg_speed':      round(agg['avg_speed'] or 0, 1),
            'avg_congestion': round(agg['avg_congestion'] or 0, 1),
            'max_congestion': round(agg['max_congestion'] or 0, 1),
            'min_congestion': round(agg['min_congestion'] or 0, 1),
            'avg_vehicles':   round(agg['avg_vehicles'] or 0),
        }

    @staticmethod
    def predict_congestion_for(location: Location) -> dict:
        """
        UML: + predictCongestion()
        Lightweight rule-based predictor (no external model).
        """
        from apps.predictions.ml import CongestionPredictor
        return CongestionPredictor.predict(location)

    @staticmethod
    def calculate_alternative_routes(location: Location) -> list:
        """
        UML: + calculateAlternativeRoutes()
        Returns saved alternative routes for a congested location.
        If none stored, generates mock routes.
        """
        saved = list(AlternativeRoute.objects.filter(
            origin=location, is_active=True
        ).values(
            'id', 'description', 'distance_km', 'est_time_min', 'via_roads'
        ))
        if saved:
            return saved
        # Generate mock alternatives
        mock_routes = [
            {
                'id': None,
                'description': f'Via {loc} avoiding {location.road_name or location.name}',
                'distance_km': round(random.uniform(2, 15), 1),
                'est_time_min': random.randint(10, 45),
                'via_roads': f'{loc} Road',
            }
            for loc in ['Ogui Road', 'Enugu-Onitsha Expressway', 'Independence Layout']
        ]
        return sorted(mock_routes, key=lambda r: r['est_time_min'])


# ─────────────────────────────────────────────────────────────────
# DatabaseManager  (UML: + saveReport, + getHistoricalData)
# ─────────────────────────────────────────────────────────────────
class DatabaseManager:

    @staticmethod
    def save_report(title: str, report_type: str, location,
                    period_start, period_end, generated_by, summary: dict = None) -> Report:
        """UML: + saveReport()"""
        report = Report.objects.create(
            title        = title,
            report_type  = report_type,
            location     = location,
            generated_by = generated_by,
            period_start = period_start,
            period_end   = period_end,
            summary_json = summary or DatabaseManager._build_summary(location, period_start, period_end),
        )
        return report

    @staticmethod
    def get_historical_data(location=None, days: int = 7) -> dict:
        """UML: + getHistoricalData()"""
        since = timezone.now() - timedelta(days=days)
        qs = TrafficReading.objects.filter(timestamp__gte=since)
        if location:
            qs = qs.filter(location=location)
        agg = qs.aggregate(
            avg_congestion = Avg('congestion_index'),
            avg_speed      = Avg('avg_speed'),
            total_readings = Count('id'),
        )
        return {
            'days':           days,
            'location':       str(location) if location else 'All',
            'total_readings': agg['total_readings'] or 0,
            'avg_congestion': round(agg['avg_congestion'] or 0, 1),
            'avg_speed':      round(agg['avg_speed'] or 0, 1),
        }

    @staticmethod
    def _build_summary(location, period_start, period_end) -> dict:
        qs = TrafficReading.objects.filter(
            location=location,
            timestamp__gte=period_start,
            timestamp__lte=period_end,
        )
        agg = qs.aggregate(
            avg_c = Avg('congestion_index'),
            max_c = Max('congestion_index'),
            avg_s = Avg('avg_speed'),
            cnt   = Count('id'),
        )
        level_counts = {}
        for row in qs.values('congestion_level').annotate(n=Count('id')):
            level_counts[row['congestion_level']] = row['n']
        return {
            'avg_congestion':   round(agg['avg_c'] or 0, 1),
            'max_congestion':   round(agg['max_c'] or 0, 1),
            'avg_speed':        round(agg['avg_s'] or 0, 1),
            'total_readings':   agg['cnt'] or 0,
            'level_breakdown':  level_counts,
        }
