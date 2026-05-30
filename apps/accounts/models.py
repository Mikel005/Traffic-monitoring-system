"""
User model implementing the UML hierarchy:
    User ◄── TrafficAdministrator
         ◄── TrafficOfficer
         ◄── RoadUser
"""
from django.contrib.auth.models import AbstractUser
from django.db import models


# ── Role constants ────────────────────────────────────────────────
ROLE_ADMIN     = 'admin'
ROLE_OFFICER   = 'officer'
ROLE_ROAD_USER = 'road_user'

ROLE_CHOICES = [
    (ROLE_ADMIN,     'Traffic Administrator'),
    (ROLE_OFFICER,   'Traffic Officer'),
    (ROLE_ROAD_USER, 'Road User'),
]


class User(AbstractUser):
    """
    Base User from UML: + login()
    Extended with a role field that determines which proxy model applies.
    """
    role  = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_ROAD_USER)
    phone = models.CharField(max_length=30, blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    class Meta:
        verbose_name = 'User'

    def __str__(self):
        return f"{self.get_full_name() or self.username} ({self.get_role_display()})"

    # UML: + login() — delegated to Django's authenticate/login
    def login(self, request, password):
        from django.contrib.auth import authenticate, login
        user = authenticate(request, username=self.username, password=password)
        if user:
            login(request, user)
        return user

    @property
    def is_admin(self):
        return self.role == ROLE_ADMIN

    @property
    def is_officer(self):
        return self.role == ROLE_OFFICER

    @property
    def is_road_user(self):
        return self.role == ROLE_ROAD_USER

    @property
    def role_icon(self):
        return {'admin': '🛡️', 'officer': '👮', 'road_user': '🚗'}.get(self.role, '👤')


# ── Proxy managers ────────────────────────────────────────────────
class AdminManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(role=ROLE_ADMIN)


class OfficerManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(role=ROLE_OFFICER)


class RoadUserManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(role=ROLE_ROAD_USER)


class TrafficAdministrator(User):
    """UML: TrafficAdministrator + configureCameras() + manageUsers()"""
    objects = AdminManager()

    class Meta:
        proxy        = True
        verbose_name = 'Traffic Administrator'

    def configure_cameras(self):
        """Returns queryset of all cameras available to manage."""
        from apps.devices.models import Camera
        return Camera.objects.all()

    def manage_users(self):
        """Returns all non-admin users for management."""
        return User.objects.exclude(role=ROLE_ADMIN)


class TrafficOfficer(User):
    """UML: TrafficOfficer + receiveAlerts() + generateReports()"""
    objects = OfficerManager()

    class Meta:
        proxy        = True
        verbose_name = 'Traffic Officer'

    def receive_alerts(self):
        """Returns unresolved alerts assigned to this officer."""
        from apps.alerts.models import Alert
        return Alert.objects.filter(is_resolved=False).order_by('-timestamp')

    def generate_reports(self, location=None, days=7):
        """Delegates to DatabaseManager.save_report()."""
        from apps.traffic.services import DatabaseManager
        return DatabaseManager.get_historical_data(location=location, days=days)


class RoadUser(User):
    """UML: RoadUser + getAlternativeRoutes()"""
    objects = RoadUserManager()

    class Meta:
        proxy        = True
        verbose_name = 'Road User'

    def get_alternative_routes(self, location):
        """Delegates to TrafficProcessor.calculate_alternative_routes()."""
        from apps.traffic.services import TrafficProcessor
        return TrafficProcessor.calculate_alternative_routes(location)
