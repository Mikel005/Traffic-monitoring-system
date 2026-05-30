"""
Tests for the traffic app.
Run: pytest apps/traffic/tests.py -v
"""
import pytest
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone

from apps.traffic.models import Location, TrafficReading, CongestionLevel
from apps.traffic.views import generate_mock_reading

User = get_user_model()


@pytest.fixture
def admin_user(db):
    return User.objects.create_superuser(
        username='testadmin', password='testpass123', email='test@traffic.ng'
    )


@pytest.fixture
def location(db):
    return Location.objects.create(
        name='Test Bridge', segment_id='test_001',
        latitude=6.4541, longitude=3.3947, city='Enugu',
    )


@pytest.fixture
def auth_client(admin_user):
    client = Client()
    client.login(username='testadmin', password='testpass123')
    return client


class TestLocation(TestCase):
    def test_str(self):
        loc = Location(name='Enugu Bridge', city='Enugu')
        assert str(loc) == 'Enugu Bridge (Enugu)'

    def test_index_to_level(self):
        assert TrafficReading.index_to_level(10)  == CongestionLevel.FREE_FLOW
        assert TrafficReading.index_to_level(35)  == CongestionLevel.MODERATE
        assert TrafficReading.index_to_level(60)  == CongestionLevel.HEAVY
        assert TrafficReading.index_to_level(80)  == CongestionLevel.GRIDLOCK

    def test_index_boundaries(self):
        assert TrafficReading.index_to_level(0)   == CongestionLevel.FREE_FLOW
        assert TrafficReading.index_to_level(25)  == CongestionLevel.FREE_FLOW
        assert TrafficReading.index_to_level(26)  == CongestionLevel.MODERATE
        assert TrafficReading.index_to_level(75)  == CongestionLevel.HEAVY
        assert TrafficReading.index_to_level(76)  == CongestionLevel.GRIDLOCK
        assert TrafficReading.index_to_level(100) == CongestionLevel.GRIDLOCK


@pytest.mark.django_db
def test_generate_mock_reading(location):
    data = generate_mock_reading(location)
    assert data['location'] == location
    assert 0 <= data['congestion_index'] <= 100
    assert data['avg_speed'] >= 0
    assert data['vehicle_count'] >= 0
    assert data['source'] == 'mock'


@pytest.mark.django_db
def test_mock_reading_all_hours(location):
    from apps.traffic.views import _simulate_congestion_index
    for hour in range(24):
        idx = _simulate_congestion_index(hour, is_weekend=False)
        assert 0 <= idx <= 100, f"Out-of-range at hour {hour}: {idx}"


@pytest.mark.django_db
def test_dashboard_redirects_anonymous(client):
    resp = client.get(reverse('traffic:dashboard'))
    assert resp.status_code == 302
    assert '/accounts/login/' in resp['Location']


@pytest.mark.django_db
def test_dashboard_loads(auth_client):
    resp = auth_client.get(reverse('traffic:dashboard'))
    assert resp.status_code == 200
    assert b'TrafficIQ' in resp.content


@pytest.mark.django_db
def test_map_view_loads(auth_client):
    resp = auth_client.get(reverse('traffic:map'))
    assert resp.status_code == 200


@pytest.mark.django_db
def test_api_live_returns_json(auth_client, location):
    TrafficReading.objects.create(
        location=location, timestamp=timezone.now(),
        vehicle_count=30, avg_speed=45.0,
        congestion_index=40.0, congestion_level='moderate',
    )
    resp = auth_client.get(reverse('traffic:api_live'))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.django_db
def test_api_history_returns_json(auth_client, location):
    resp = auth_client.get(
        reverse('traffic:api_history'),
        {'location_id': location.id, 'hours': 1}
    )
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.django_db
def test_location_detail(auth_client, location):
    resp = auth_client.get(reverse('traffic:location_detail', args=[location.id]))
    assert resp.status_code == 200


@pytest.mark.django_db
def test_alert_created_for_gridlock(location):
    from apps.alerts.tasks import check_and_fire_alerts
    from apps.alerts.models import Alert
    check_and_fire_alerts(
        location_id=location.id, location_name=location.name,
        congestion_index=82.0, congestion_level='gridlock',
    )
    assert Alert.objects.filter(location=location, level='gridlock').exists()


@pytest.mark.django_db
def test_no_alert_for_free_flow(location):
    from apps.alerts.tasks import check_and_fire_alerts
    from apps.alerts.models import Alert
    check_and_fire_alerts(
        location_id=location.id, location_name=location.name,
        congestion_index=15.0, congestion_level='free_flow',
    )
    assert not Alert.objects.filter(location=location).exists()
