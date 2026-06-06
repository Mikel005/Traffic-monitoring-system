from django.apps import AppConfig


class TrafficConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.traffic'
    verbose_name = 'Traffic'

    def ready(self):
        import os
        from django.db.backends.signals import connection_created
        from django.dispatch import receiver

        @receiver(connection_created)
        def configure_sqlite(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                cursor = connection.cursor()
                cursor.execute('PRAGMA journal_mode=WAL;')
                cursor.execute('PRAGMA synchronous=NORMAL;')
                # busy_timeout must be >= Django OPTIONS timeout (30s) so SQLite's
                # C-level retry fires first, giving the smoothest backoff behaviour.
                cursor.execute('PRAGMA busy_timeout=35000;')
                cursor.execute('PRAGMA cache_size=-32000;')     # 32 MB page cache

        # Only start scheduler in the main process (not during migrations/tests)
        if os.environ.get('RUN_MAIN') == 'true' or not os.environ.get('DJANGO_SETTINGS_MODULE'):
            try:
                from apps.traffic.scheduler import start
                start()
            except Exception:
                pass   # fail silently if DB not ready yet
