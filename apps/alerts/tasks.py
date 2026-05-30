from celery import shared_task
from loguru import logger

THRESHOLDS = {'heavy': 50, 'gridlock': 75}


@shared_task(name='alerts.check_and_fire_alerts')
def check_and_fire_alerts(location_id, location_name, congestion_index, congestion_level):
    """Create an Alert record if congestion is Heavy or Gridlock."""
    if congestion_level not in ('heavy', 'gridlock'):
        return

    from apps.alerts.models import Alert
    from apps.traffic.models import Location

    emoji   = '🔴' if congestion_level == 'gridlock' else '🟠'
    label   = 'GRIDLOCK' if congestion_level == 'gridlock' else 'Heavy Traffic'
    message = (f"{emoji} {label} at {location_name}. "
               f"Congestion index: {congestion_index:.1f}/100")

    Alert.objects.create(
        location_id      = location_id,
        level            = congestion_level,
        message          = message,
        congestion_index = congestion_index,
    )
    logger.warning(f"🚨 Alert: {message}")
