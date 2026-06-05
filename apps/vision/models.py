from django.db import models
from django.utils import timezone
from apps.traffic.models import Location


class Camera(models.Model):
    name       = models.CharField(max_length=200)
    location   = models.ForeignKey(Location, on_delete=models.CASCADE,
                                   related_name='vision_cameras')
    stream_url = models.CharField(
        max_length=500, blank=True,
        help_text="Local video file path, RTSP URL (rtsp://...), or HTTP stream URL"
    )
    is_active  = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} @ {self.location.name}"

    @property
    def stream_type(self):
        if self.stream_url.startswith('rtsp://'):
            return 'rtsp'
        if self.stream_url.startswith(('http://', 'https://')):
            return 'http'
        if self.stream_url:
            return 'file'
        return 'none'


class UploadedVideo(models.Model):
    video_file  = models.FileField(upload_to='videos/')
    location    = models.ForeignKey(Location, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name='vision_videos')
    camera      = models.ForeignKey(Camera, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name='videos')
    status      = models.CharField(max_length=20, default='pending', choices=[
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('completed',  'Completed'),
        ('failed',     'Failed'),
    ])

    # Aggregate detection results
    predicted_congestion_level = models.CharField(max_length=50, blank=True)
    vehicle_count    = models.IntegerField(default=0)
    car_count        = models.IntegerField(default=0)
    truck_count      = models.IntegerField(default=0)
    motorcycle_count = models.IntegerField(default=0)
    bus_count        = models.IntegerField(default=0)
    average_speed    = models.FloatField(default=0.0)
    queue_length     = models.FloatField(default=0.0)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"Video {self.id} — {self.status}"


class VehicleCountSession(models.Model):
    """
    One counting session = one video processed or one camera live-stream segment.
    Stores aggregate counts, direction totals, and a link to the per-crossing CSV.
    """
    session_tag   = models.CharField(max_length=64, unique=True)
    source_video  = models.ForeignKey(UploadedVideo, on_delete=models.SET_NULL,
                                      null=True, blank=True, related_name='count_sessions')
    source_camera = models.ForeignKey('Camera', on_delete=models.SET_NULL,
                                      null=True, blank=True, related_name='count_sessions')
    location      = models.ForeignKey(Location, on_delete=models.SET_NULL,
                                      null=True, blank=True)

    started_at    = models.DateTimeField(default=timezone.now)
    ended_at      = models.DateTimeField(null=True, blank=True)

    total_count      = models.IntegerField(default=0)
    car_count        = models.IntegerField(default=0)
    truck_count      = models.IntegerField(default=0)
    bus_count        = models.IntegerField(default=0)
    motorcycle_count = models.IntegerField(default=0)
    inbound_count    = models.IntegerField(default=0)
    outbound_count   = models.IntegerField(default=0)

    avg_speed       = models.FloatField(default=0.0)
    peak_congestion = models.CharField(max_length=20, default='FREE FLOW')

    csv_file = models.FileField(upload_to='counts/', null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f"Session {self.session_tag} — {self.total_count} vehicles"

    @property
    def csv_url(self):
        return self.csv_file.url if self.csv_file else None
