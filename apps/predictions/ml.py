"""
CongestionPredictor — rule-based + trend predictor.
No external model files required. Uses recent readings + time-of-day heuristics.
"""
import random
from django.utils import timezone
from django.db.models import Avg


class CongestionPredictor:

    # Enugu congestion multipliers by hour (0–23)
    HOUR_MULTIPLIER = {
        0: 0.1, 1: 0.08, 2: 0.07, 3: 0.06, 4: 0.07, 5: 0.15,
        6: 0.45, 7: 0.85, 8: 0.95, 9: 0.75, 10: 0.55, 11: 0.50,
        12: 0.60, 13: 0.55, 14: 0.50, 15: 0.55, 16: 0.70, 17: 0.90,
        18: 0.95, 19: 0.85, 20: 0.65, 21: 0.40, 22: 0.25, 23: 0.15,
    }

    @classmethod
    def predict(cls, location) -> dict:
        """Predict congestion for 15-, 30-, and 60-minute horizons."""
        from apps.traffic.models import TrafficReading

        now    = timezone.now()
        recent = TrafficReading.objects.filter(
            location=location
        ).order_by('-timestamp').values('congestion_index')[:10]

        if recent:
            current = sum(r['congestion_index'] for r in recent) / len(recent)
        else:
            current = 30.0   # default moderate

        predictions = {}
        for minutes, label in [(15, 'minutes_15'), (30, 'minutes_30'), (60, 'minutes_60')]:
            future_hour = (now.hour + minutes // 60) % 24
            multiplier  = cls.HOUR_MULTIPLIER[future_hour]
            jitter      = random.gauss(0, 3)
            # Trend: congestion decays slightly over time unless rush hour
            trend_factor = 1.0 if multiplier > 0.7 else 0.95
            pred_value   = min(100, max(0, current * trend_factor * multiplier / 0.5 + jitter))
            predictions[label] = round(pred_value, 1)

        predictions['current']    = round(current, 1)
        predictions['model']      = 'rule_based_v2'
        predictions['confidence'] = round(min(0.95, 0.6 + 0.1 * min(len(recent), 4)), 2)
        return predictions
