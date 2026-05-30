"""
Background scheduler — feeds mock traffic data every 30 seconds.
Uses APScheduler with Django integration (no Redis/Celery needed).
"""
import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django_apscheduler.jobstores import DjangoJobStore
from django_apscheduler.models import DjangoJobExecution

logger = logging.getLogger(__name__)

_scheduler = None


def feed_mock_data():
    """Job: generate and store mock readings for all active locations."""
    try:
        from apps.traffic.models import Location
        from apps.traffic.services import SystemController
        locations = list(Location.objects.filter(is_active=True))
        for location in locations:
            data = SystemController.generate_mock_reading(location)
            SystemController.store_raw_data(data)
        logger.debug(f"[scheduler] Fed mock data for {len(locations)} locations")
    except Exception as exc:
        logger.warning(f"[scheduler] feed_mock_data error: {exc}")


def update_predictions():
    """Job: refresh predictions for all active locations."""
    try:
        from apps.traffic.models import Location
        from apps.traffic.services import SystemController
        for location in Location.objects.filter(is_active=True):
            SystemController.update_predictive_focus(location)
    except Exception as exc:
        logger.warning(f"[scheduler] update_predictions error: {exc}")


def _start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_jobstore(DjangoJobStore(), 'default')

    _scheduler.add_job(
        feed_mock_data,
        trigger  = IntervalTrigger(seconds=30),
        id       = 'feed_mock_data',
        name     = 'Feed mock traffic data',
        replace_existing = True,
    )
    _scheduler.add_job(
        update_predictions,
        trigger  = IntervalTrigger(minutes=2),
        id       = 'update_predictions',
        name     = 'Update congestion predictions',
        replace_existing = True,
    )
    _scheduler.start()
    logger.info("[scheduler] Background scheduler started.")


def start():
    """Start the background scheduler. Called from AppConfig.ready()."""
    threading.Thread(target=_start_scheduler, daemon=True).start()
