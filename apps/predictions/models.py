"""
Prediction model — stores CongestionPredictor output.
"""
from django.db import models
from django.utils import timezone
from apps.traffic.models import Location


class Prediction(models.Model):
    """Stores multi-horizon congestion forecasts per location."""
    timestamp     = models.DateTimeField(default=timezone.now, db_index=True)
    location      = models.ForeignKey(Location, on_delete=models.CASCADE,
                      related_name='predictions')
    pred_15min    = models.FloatField(null=True, blank=True)
    pred_30min    = models.FloatField(null=True, blank=True)
    pred_60min    = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='rule_based_v2')
    confidence    = models.FloatField(default=0.7)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes  = [models.Index(fields=['location', '-timestamp'])]

    def __str__(self):
        return f"Pred @ {self.location.name} {self.timestamp:%H:%M}"

    @property
    def max_horizon(self):
        return max(
            v for v in [self.pred_15min, self.pred_30min, self.pred_60min]
            if v is not None
        ) if any([self.pred_15min, self.pred_30min, self.pred_60min]) else 0
