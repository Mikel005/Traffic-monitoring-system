"""
ASGI config — standard Django ASGI (no Channels/Redis required).
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'traffic_project.settings')
application = get_asgi_application()
