"""
Seed script — creates initial data matching the Enugu traffic monitoring scenario.
Run with: python manage.py shell < seed_data.py
"""
import django
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'traffic_project.settings')
django.setup()

from django.contrib.auth import get_user_model
from apps.traffic.models import Location, AlternativeRoute
from apps.devices.models import Camera, IoTSensor, DeviceController
from apps.accounts.models import ROLE_ADMIN, ROLE_OFFICER, ROLE_ROAD_USER
from apps.traffic.services import SystemController

User = get_user_model()

# ── Users ─────────────────────────────────────────────────────────
admin = User.objects.filter(username='admin').first()
if not admin:
    admin = User.objects.create_superuser(
        username='admin', email='admin@traffic.local', password='admin123',
        role=ROLE_ADMIN, first_name='System', last_name='Administrator'
    )
    print(f"Created admin: admin / admin123")

if not User.objects.filter(username='officer1').exists():
    User.objects.create_user(
        username='officer1', email='officer@traffic.local', password='officer123',
        role=ROLE_OFFICER, first_name='Emeka', last_name='Okafor'
    )
    print("Created officer: officer1 / officer123")

if not User.objects.filter(username='roaduser1').exists():
    User.objects.create_user(
        username='roaduser1', email='user@traffic.local', password='user123',
        role=ROLE_ROAD_USER, first_name='Amina', last_name='Bello'
    )
    print("Created road user: roaduser1 / user123")

# ── Locations (Enugu major roads) ─────────────────────────────────
LOCATIONS = [
    dict(name='Okpara Avenue', segment_id='ENU-001',
         latitude=6.4355, longitude=7.4920, road_name='Okpara Avenue',
         city='Enugu', speed_limit=50),
    dict(name='Agbani Road', segment_id='ENU-002',
         latitude=6.4020, longitude=7.5020, road_name='Agbani Road',
         city='Enugu', speed_limit=50),
    dict(name='Independence Layout - Bisalla', segment_id='ENU-003',
         latitude=6.4460, longitude=7.5250, road_name='Bisalla Road',
         city='Enugu', speed_limit=60),
    dict(name='Ogui Road', segment_id='ENU-004',
         latitude=6.4360, longitude=7.5020, road_name='Ogui Road',
         city='Enugu', speed_limit=50),
    dict(name='Abakpa Nike', segment_id='ENU-005',
         latitude=6.4710, longitude=7.5220, road_name='Abakpa Nike Road',
         city='Enugu', speed_limit=50),
    dict(name='Emene', segment_id='ENU-006',
         latitude=6.4770, longitude=7.5680, road_name='Emene Old Road',
         city='Enugu', speed_limit=80),
    dict(name='Enugu-Onitsha Expressway', segment_id='ENU-007',
         latitude=6.4390, longitude=7.4720, road_name='Enugu-Onitsha Expressway',
         city='Enugu', speed_limit=100),
    dict(name='Zik Avenue', segment_id='ENU-008',
         latitude=6.4250, longitude=7.4930, road_name='Zik Avenue',
         city='Enugu', speed_limit=50),
]

created_locs = []
for loc_data in LOCATIONS:
    loc, created = Location.objects.get_or_create(
        segment_id=loc_data['segment_id'], defaults=loc_data
    )
    created_locs.append(loc)
    if created:
        print(f"  Created location: {loc.name}")

# ── Alternative Routes ────────────────────────────────────────────
alt_routes = [
    ('ENU-001', 'Via Market Road', 2.5, 10, 'Market Road, Holy Ghost'),
    ('ENU-001', 'Via Ogui Road Junction', 3.0, 15, 'Ogui Road'),
    ('ENU-004', 'Via Nkpokiti Street', 1.5, 8, 'Nkpokiti Street'),
    ('ENU-005', 'Via ESBS Junction', 2.0, 12, 'ESBS Road'),
    ('ENU-008', 'Via Uwani', 2.0, 10, 'Uwani Road, Agbani Road'),
]
for seg_id, desc, dist, time, via in alt_routes:
    loc = Location.objects.filter(segment_id=seg_id).first()
    if loc and not AlternativeRoute.objects.filter(origin=loc, description=desc).exists():
        AlternativeRoute.objects.create(
            origin=loc, description=desc,
            distance_km=dist, est_time_min=time, via_roads=via
        )
        print(f"  Created route: {desc[:40]}")

# ── Cameras ───────────────────────────────────────────────────────
for i, loc in enumerate(created_locs[:4], 1):
    cam, created = Camera.objects.get_or_create(
        name=f'Camera-{i:02d}',
        location=loc,
        defaults=dict(resolution='1080p', fps=30, direction='Northbound')
    )
    if created:
        print(f"  Created camera: {cam.name}")

# ── IoT Sensors ───────────────────────────────────────────────────
sensor_types = ['volume', 'speed', 'occupancy', 'weather']
for i, loc in enumerate(created_locs, 1):
    stype = sensor_types[i % len(sensor_types)]
    s, created = IoTSensor.objects.get_or_create(
        device_id=f'IOT-{i:04d}',
        defaults=dict(name=f'Sensor-{i:02d}', location=loc, sensor_type=stype)
    )
    if created:
        print(f"  Created sensor: {s.name} ({stype})")

# ── Device Controller ─────────────────────────────────────────────
ctrl, created = DeviceController.objects.get_or_create(name='Main Controller Enugu')
if created:
    ctrl.managed_cameras.set(Camera.objects.all())
    ctrl.managed_sensors.set(IoTSensor.objects.all())
    print(f"  Created controller: {ctrl.name}")

# ── Initial mock readings (3 per location) ────────────────────────
print("Seeding initial traffic readings...")
for loc in created_locs:
    if loc.readings.exists():
        continue
    for _ in range(5):
        data = SystemController.generate_mock_reading(loc)
        SystemController.store_raw_data(data)
print("Seed complete!")
print("\n--- Login credentials ---")
print("Admin:       admin     / admin123")
print("Officer:     officer1  / officer123")
print("Road User:   roaduser1 / user123")
