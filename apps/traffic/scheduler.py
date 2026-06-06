"""
Background scheduler — feeds mock traffic data every 60 seconds.
Uses APScheduler with MemoryJobStore (NOT DjangoJobStore) so that
APScheduler does NOT wrap each job in transaction.atomic() to persist
DjangoJobExecution records. That wrapping was the source of long-held
SQLite write locks that caused "database is locked" errors.
"""
import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

_scheduler = None


def _with_db(fn, *args, **kwargs):
    """
    Run fn, then close the thread-local Django connection immediately.
    APScheduler reuses threads, so connections accumulate without explicit
    cleanup; closing after each job prevents stale open transactions.
    """
    try:
        return fn(*args, **kwargs)
    finally:
        try:
            from django.db import connection
            connection.close()
        except Exception:
            pass


def feed_mock_data():
    """Job: generate and store mock readings for all active locations."""
    def _run():
        from apps.traffic.models import Location
        from apps.traffic.services import SystemController
        locations = list(Location.objects.filter(is_active=True))
        for location in locations:
            data = SystemController.generate_mock_reading(location)
            SystemController.store_raw_data(data)
        logger.debug(f"[scheduler] Fed mock data for {len(locations)} locations")

    try:
        _with_db(_run)
    except Exception as exc:
        logger.warning(f"[scheduler] feed_mock_data error: {exc}")


def update_predictions():
    """Job: refresh predictions for all active locations."""
    def _run():
        from apps.traffic.models import Location
        from apps.traffic.services import SystemController
        for location in Location.objects.filter(is_active=True):
            SystemController.update_predictive_focus(location)

    try:
        _with_db(_run)
    except Exception as exc:
        logger.warning(f"[scheduler] update_predictions error: {exc}")


def _start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    # MemoryJobStore: APScheduler keeps job metadata in memory only.
    # This eliminates the DjangoJobExecution writes that previously
    # wrapped each job in transaction.atomic() and held the write lock.
    _scheduler = BackgroundScheduler(
        jobstores={'default': {'type': 'memory'}},
    )

    _scheduler.add_job(
        feed_mock_data,
        trigger          = IntervalTrigger(seconds=60),   # was 30s; halved write rate
        id               = 'feed_mock_data',
        name             = 'Feed mock traffic data',
        replace_existing = True,
        misfire_grace_time = 30,
    )
    _scheduler.add_job(
        update_predictions,
        trigger          = IntervalTrigger(minutes=3),    # was 2min
        id               = 'update_predictions',
        name             = 'Update congestion predictions',
        replace_existing = True,
        misfire_grace_time = 60,
    )
    _scheduler.start()
    logger.info("[scheduler] Background scheduler started (MemoryJobStore).")


def start():
    """Start the background scheduler. Called from AppConfig.ready()."""
    threading.Thread(target=_start_scheduler, daemon=True).start()
