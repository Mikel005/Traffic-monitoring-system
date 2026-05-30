"""
Device models implementing the UML hierarchy:

    SensorUnit (abstract)
        ├── Camera        + captureVideo()
        └── IoTSensor     + collectData()

    DeviceController      + pollAllDevices()
"""
from django.db import models
from django.utils import timezone


SENSOR_TYPE_CHOICES = [
    ('speed',      'Speed Sensor'),
    ('volume',     'Volume Counter'),
    ('occupancy',  'Occupancy Sensor'),
    ('weather',    'Weather Station'),
]


# ── SensorUnit (Abstract Base) ────────────────────────────────────
class SensorUnit(models.Model):
    """UML: SensorUnit — abstract base for Camera and IoTSensor."""
    name       = models.CharField(max_length=200)
    location   = models.ForeignKey(
        'traffic.Location', on_delete=models.CASCADE, related_name='%(class)ss'
    )
    is_active  = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

    def __str__(self):
        return f"{self.name} @ {self.location.name}"


# ── Camera ────────────────────────────────────────────────────────
class Camera(SensorUnit):
    """UML: Camera + captureVideo()"""
    RESOLUTION_CHOICES = [
        ('480p',  '480p SD'),
        ('720p',  '720p HD'),
        ('1080p', '1080p Full HD'),
        ('4k',    '4K Ultra HD'),
    ]
    stream_url  = models.CharField(max_length=500, blank=True,
                    help_text='RTSP stream URL or path to local video file')
    resolution  = models.CharField(max_length=10, choices=RESOLUTION_CHOICES, default='1080p')
    fps         = models.IntegerField(default=30, help_text='Frames per second')
    direction   = models.CharField(max_length=100, blank=True,
                    help_text='e.g. Northbound, Southbound')

    class Meta:
        verbose_name = 'Camera'

    def capture_video(self):
        """
        UML: + captureVideo()
        Returns a dict with frame metadata. In mock mode returns simulated data.
        """
        return {
            'camera_id':  self.pk,
            'camera_name': self.name,
            'location':   self.location.name,
            'stream_url': self.stream_url,
            'timestamp':  timezone.now().isoformat(),
            'status':     'active' if self.is_active else 'offline',
        }


# ── IoTSensor ─────────────────────────────────────────────────────
class IoTSensor(SensorUnit):
    """UML: IoTSensor + collectData()"""
    sensor_type   = models.CharField(max_length=20, choices=SENSOR_TYPE_CHOICES,
                      default='volume')
    device_id     = models.CharField(max_length=100, unique=True)
    firmware_ver  = models.CharField(max_length=50, default='1.0.0')
    last_ping     = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'IoT Sensor'

    def collect_data(self):
        """
        UML: + collectData()
        Returns a sensor reading dict. In mock mode returns simulated data.
        """
        import random
        self.last_ping = timezone.now()
        self.save(update_fields=['last_ping'])
        return {
            'sensor_id':   self.pk,
            'device_id':   self.device_id,
            'sensor_type': self.sensor_type,
            'location':    self.location.name,
            'timestamp':   timezone.now().isoformat(),
            'value':       round(random.uniform(0, 100), 2),
            'unit':        self._unit_for_type(),
        }

    def _unit_for_type(self):
        return {
            'speed':     'km/h',
            'volume':    'veh/hr',
            'occupancy': '%',
            'weather':   'mm/hr',
        }.get(self.sensor_type, 'units')

    @property
    def is_online(self):
        if not self.last_ping:
            return False
        return (timezone.now() - self.last_ping).seconds < 300  # 5-min timeout


# ── SensorReading ─────────────────────────────────────────────────
class SensorReading(models.Model):
    """Persisted data point from any IoT sensor."""
    sensor    = models.ForeignKey(IoTSensor, on_delete=models.CASCADE,
                  related_name='readings')
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    value     = models.FloatField()
    raw_json  = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.sensor.name}: {self.value} @ {self.timestamp:%H:%M}"


# ── DeviceController ──────────────────────────────────────────────
class DeviceController(models.Model):
    """
    UML: DeviceController — manages a group of SensorUnits
    Orchestrates polling of cameras and IoT sensors.
    """
    name             = models.CharField(max_length=200)
    managed_cameras  = models.ManyToManyField(Camera, blank=True,
                         related_name='controllers')
    managed_sensors  = models.ManyToManyField(IoTSensor, blank=True,
                         related_name='controllers')
    is_active        = models.BooleanField(default=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Device Controller'

    def __str__(self):
        return self.name

    def poll_all_devices(self):
        """
        UML: + pollAllDevices()
        Calls captureVideo() on each camera and collectData() on each IoT sensor.
        Returns a summary dict.
        """
        results = {'cameras': [], 'sensors': []}
        for cam in self.managed_cameras.filter(is_active=True):
            results['cameras'].append(cam.capture_video())
        for sensor in self.managed_sensors.filter(is_active=True):
            reading = sensor.collect_data()
            # Persist the reading
            SensorReading.objects.create(
                sensor=sensor,
                value=reading['value'],
                raw_json=reading,
            )
            results['sensors'].append(reading)
        return results
