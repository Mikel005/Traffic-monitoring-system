"""
Celery tasks for the traffic app.
fetch_traffic_data — runs every 60 seconds via django-celery-beat.
"""
import random
import math
from datetime import datetime

from celery import shared_task
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone
from loguru import logger


@shared_task(name='traffic.fetch_traffic_data')
def fetch_traffic_data():
    """
    Main periodic task: fetch traffic data for all active locations.
    Uses TomTom API if key is configured, otherwise generates mock data.
    Runs every 60 seconds.
    """
    from apps.traffic.models import Location, TrafficReading
    from apps.alerts.tasks import check_and_fire_alerts

    locations = Location.objects.filter(is_active=True)
    created   = 0

    for loc in locations:
        try:
            if settings.USE_MOCK_DATA or not settings.TOMTOM_API_KEY:
                data = _generate_mock(loc)
            else:
                data = _fetch_tomtom(loc)

            reading = TrafficReading.objects.create(**data)
            created += 1

            # Check alert thresholds
            check_and_fire_alerts.delay(
                location_id      = loc.id,
                location_name    = loc.name,
                congestion_index = reading.congestion_index,
                congestion_level = reading.congestion_level,
            )

        except Exception as e:
            logger.error(f"Failed to fetch data for {loc.name}: {e}")

    # Invalidate live cache
    cache.delete('api:live')
    cache.delete('api:summary')

    logger.info(f"✅ fetch_traffic_data: created {created} readings")
    return created


def _generate_mock(location) -> dict:
    """Generate a realistic simulated traffic reading."""
    now   = timezone.now()
    hour  = now.hour
    is_wk = now.weekday() >= 5

    if is_wk:
        base = 20 + 15 * math.sin(math.pi * (hour - 10) / 12)
    else:
        am   = 60 * math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        pm   = 70 * math.exp(-0.5 * ((hour - 17) / 1.5) ** 2)
        base = max(am, pm) + 10

    idx   = max(0.0, min(100.0, base + random.gauss(0, 5)))
    speed = max(5.0, 60.0 - (idx * 0.55) + random.gauss(0, 3))
    count = max(0, int(idx * 1.2 + random.gauss(0, 8)))

    from apps.traffic.models import TrafficReading
    return dict(
        location         = location,
        timestamp        = now,
        vehicle_count    = count,
        car_count        = int(count * 0.65),
        truck_count      = int(count * 0.10),
        motorcycle_count = int(count * 0.20),
        bus_count        = int(count * 0.05),
        avg_speed        = round(speed, 1),
        free_flow_speed  = 60.0,
        queue_length     = round(max(0, idx * 2.5), 1),
        congestion_index = round(idx, 2),
        congestion_level = TrafficReading.index_to_level(idx),
        rainfall_mm      = round(random.choices([0, random.uniform(0.5,15)],
                                                 weights=[0.85, 0.15])[0], 1),
        visibility_km    = round(random.uniform(5, 10), 1),
        temperature_c    = round(random.uniform(25, 35), 1),
        source           = 'mock',
    )


def _fetch_tomtom(location) -> dict:
    """Fetch real data from TomTom Traffic Flow API."""
    import httpx
    from apps.traffic.models import TrafficReading

    url    = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        'key':   settings.TOMTOM_API_KEY,
        'point': f"{location.latitude},{location.longitude}",
        'unit':  'KMPH',
    }

    with httpx.Client(timeout=10) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        fd = resp.json().get('flowSegmentData', {})

    curr_speed      = fd.get('currentSpeed', 30)
    free_flow_speed = fd.get('freeFlowSpeed', 60)
    idx             = max(0.0, min(100.0,
        (1 - curr_speed / max(free_flow_speed, 1)) * 100
    ))

    return dict(
        location         = location,
        timestamp        = timezone.now(),
        vehicle_count    = int(idx * 1.2),
        car_count        = 0, truck_count=0, motorcycle_count=0, bus_count=0,
        avg_speed        = float(curr_speed),
        free_flow_speed  = float(free_flow_speed),
        queue_length     = 0.0,
        congestion_index = round(idx, 2),
        congestion_level = TrafficReading.index_to_level(idx),
        rainfall_mm      = 0.0,
        visibility_km    = 10.0,
        temperature_c    = 28.0,
        source           = 'tomtom',
    )
