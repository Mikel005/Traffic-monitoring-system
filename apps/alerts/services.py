"""
NotificationService — UML: + sendAlert() + updatePredictiveData()
"""
from apps.alerts.models import Alert
from apps.traffic.models import CongestionLevel, TrafficReading


class NotificationService:

    @staticmethod
    def send_alert(reading: TrafficReading) -> Alert:
        """
        UML: + sendAlert()
        Creates an Alert for heavy/gridlock readings.
        Deduplicates: won't create a new alert if one already exists
        within the last 10 minutes for the same location.
        """
        from django.utils import timezone
        from datetime import timedelta

        recent_cutoff = timezone.now() - timedelta(minutes=10)
        exists = Alert.objects.filter(
            location   = reading.location,
            is_resolved= False,
            timestamp__gte = recent_cutoff,
        ).exists()
        if exists:
            return None

        severity = 'critical' if reading.congestion_level == CongestionLevel.GRIDLOCK else 'warning'
        level_labels = {
            'heavy':    'Heavy Traffic',
            'gridlock': '🚨 GRIDLOCK',
        }
        msg = (
            f"{level_labels.get(reading.congestion_level, 'Alert')} detected at "
            f"{reading.location.name}. "
            f"Congestion index: {reading.congestion_index:.1f}/100. "
            f"Average speed: {reading.avg_speed:.1f} km/h."
        )
        alert = Alert.objects.create(
            location         = reading.location,
            level            = reading.congestion_level,
            severity         = severity,
            message          = msg,
            congestion_index = reading.congestion_index,
        )
        return alert

    @staticmethod
    def update_predictive_data(location) -> dict:
        """
        UML: + updatePredictiveData()
        Runs predictor and stores result; returns prediction dict.
        """
        from apps.traffic.services import SystemController
        return SystemController.update_predictive_focus(location)
