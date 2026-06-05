from django.db import models
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
