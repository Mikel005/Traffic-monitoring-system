"""Context processor — injects unresolved alert count into every template."""
from apps.alerts.models import Alert


def alert_count(request):
    if request.user.is_authenticated:
        count = Alert.objects.filter(is_resolved=False).count()
        return {'alert_count': count}
    return {'alert_count': 0}
