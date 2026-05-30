from django.contrib import admin
from .models import Location, TrafficReading


@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    list_display  = ['name', 'city', 'segment_id', 'speed_limit', 'is_active']
    list_filter   = ['city', 'is_active']
    search_fields = ['name', 'segment_id', 'road_name']


@admin.register(TrafficReading)
class TrafficReadingAdmin(admin.ModelAdmin):
    list_display  = ['location', 'timestamp', 'congestion_index',
                     'congestion_level', 'avg_speed', 'vehicle_count', 'source']
    list_filter   = ['congestion_level', 'source', 'location']
    search_fields = ['location__name']
    date_hierarchy = 'timestamp'
    readonly_fields = ['timestamp']
