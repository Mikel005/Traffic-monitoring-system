"""
Management command: seed_data
Creates Enugu road locations and an admin user, then generates
48 hours of historical mock traffic readings for each location.

Usage:
    python manage.py seed_data
    python manage.py seed_data --hours 72
"""
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import timedelta
import random, math


LOCATIONS = [
    {'name': 'Okpara Avenue',           'segment_id': 'enu_001', 'lat': 6.4355, 'lon': 7.4920, 'road': 'Okpara Avenue',             'limit': 50},
    {'name': 'Agbani Road',             'segment_id': 'enu_002', 'lat': 6.4020, 'lon': 7.5020, 'road': 'Agbani Road',               'limit': 50},
    {'name': 'Independence Layout',     'segment_id': 'enu_003', 'lat': 6.4460, 'lon': 7.5250, 'road': 'Bisalla Road',              'limit': 60},
    {'name': 'Ogui Road',               'segment_id': 'enu_004', 'lat': 6.4360, 'lon': 7.5020, 'road': 'Ogui Road',                 'limit': 50},
    {'name': 'Abakpa Nike Road',        'segment_id': 'enu_005', 'lat': 6.4710, 'lon': 7.5220, 'road': 'Abakpa Nike Road',          'limit': 50},
    {'name': 'Emene Old Road',          'segment_id': 'enu_006', 'lat': 6.4770, 'lon': 7.5680, 'road': 'Emene Old Road',            'limit': 80},
    {'name': 'Enugu-Onitsha Expressway','segment_id': 'enu_007', 'lat': 6.4390, 'lon': 7.4720, 'road': 'Enugu-Onitsha Expressway',  'limit': 100},
    {'name': 'Zik Avenue',              'segment_id': 'enu_008', 'lat': 6.4250, 'lon': 7.4930, 'road': 'Zik Avenue',                'limit': 50},
]

User = get_user_model()


def _sim(hour, is_weekend):
    if is_weekend:
        base = 20 + 15 * math.sin(math.pi * (hour - 10) / 12)
    else:
        am   = 60 * math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        pm   = 70 * math.exp(-0.5 * ((hour - 17) / 1.5) ** 2)
        base = max(am, pm) + 10
    return max(0.0, min(100.0, base + random.gauss(0, 5)))


class Command(BaseCommand):
    help = 'Seed Enugu locations, admin user, and historical mock readings'

    def add_arguments(self, parser):
        parser.add_argument('--hours', type=int, default=48,
                            help='Hours of historical data to generate')

    def handle(self, *args, **options):
        from apps.traffic.models import Location, TrafficReading

        # ── Create locations ──────────────────────────────────────
        for loc_data in LOCATIONS:
            loc, created = Location.objects.get_or_create(
                segment_id = loc_data['segment_id'],
                defaults   = {
                    'name':        loc_data['name'],
                    'latitude':    loc_data['lat'],
                    'longitude':   loc_data['lon'],
                    'road_name':   loc_data['road'],
                    'speed_limit': loc_data['limit'],
                    'city':        'Enugu',
                }
            )
            status = 'Created' if created else 'Exists'
            self.stdout.write(f'  {status}: {loc.name}')

        self.stdout.write(self.style.SUCCESS(f'{len(LOCATIONS)} locations ready'))

        # ── Create admin user ─────────────────────────────────────
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser(
                username='admin', email='admin@traffic.ng', password='admin123'
            )
            self.stdout.write(self.style.SUCCESS('Admin user created (admin / admin123)'))
        else:
            self.stdout.write('   Admin user already exists')

        # ── Generate historical readings ──────────────────────────
        hours = options['hours']
        self.stdout.write(f'Generating {hours}h of mock readings...')

        locations = Location.objects.filter(is_active=True)
        now       = timezone.now()
        count     = 0

        for loc in locations:
            for minutes_ago in range(hours * 60, 0, -5):   # every 5 minutes
                ts       = now - timedelta(minutes=minutes_ago)
                idx      = _sim(ts.hour, ts.weekday() >= 5)
                speed    = max(5.0, 60 - idx * 0.55 + random.gauss(0, 3))
                vehicles = max(0, int(idx * 1.2 + random.gauss(0, 8)))

                TrafficReading.objects.create(
                    location         = loc,
                    timestamp        = ts,
                    vehicle_count    = vehicles,
                    car_count        = int(vehicles * 0.65),
                    truck_count      = int(vehicles * 0.10),
                    motorcycle_count = int(vehicles * 0.20),
                    bus_count        = int(vehicles * 0.05),
                    avg_speed        = round(speed, 1),
                    congestion_index = round(idx, 2),
                    congestion_level = TrafficReading.index_to_level(idx),
                    rainfall_mm      = 0.0,
                    source           = 'mock',
                )
                count += 1

        self.stdout.write(self.style.SUCCESS(
            f'Generated {count} readings across {locations.count()} locations'
        ))

        # ── Setup Celery Beat periodic tasks ──────────────────────
        try:
            from django_celery_beat.models import PeriodicTask, IntervalSchedule
            every_min, _ = IntervalSchedule.objects.get_or_create(
                every=1, period=IntervalSchedule.MINUTES
            )
            PeriodicTask.objects.get_or_create(
                name='Fetch Traffic Data Every Minute',
                defaults={
                    'interval': every_min,
                    'task':     'traffic.fetch_traffic_data',
                }
            )
            self.stdout.write(self.style.SUCCESS('Celery Beat task registered'))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'   Celery Beat setup skipped: {e}'))

        self.stdout.write(self.style.SUCCESS('\nSeed complete! Run the server with:'))
        self.stdout.write('   python manage.py runserver')
