from django.contrib import admin
from apps.devices.models import Camera, IoTSensor, DeviceController, SensorReading


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display  = ('name', 'location', 'resolution', 'fps', 'is_active')
    list_filter   = ('is_active', 'resolution')
    search_fields = ('name',)


@admin.register(IoTSensor)
class IoTSensorAdmin(admin.ModelAdmin):
    list_display  = ('name', 'device_id', 'location', 'sensor_type', 'is_online', 'last_ping')
    list_filter   = ('sensor_type', 'is_active')
    search_fields = ('name', 'device_id')


@admin.register(DeviceController)
class DeviceControllerAdmin(admin.ModelAdmin):
    list_display  = ('name', 'is_active', 'created_at')
    filter_horizontal = ('managed_cameras', 'managed_sensors')


@admin.register(SensorReading)
class SensorReadingAdmin(admin.ModelAdmin):
    list_display  = ('sensor', 'value', 'timestamp')
    list_filter   = ('sensor__sensor_type',)
