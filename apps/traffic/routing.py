from django.urls import re_path
from .consumers import TrafficConsumer

websocket_urlpatterns = [
    re_path(r'ws/traffic/$', TrafficConsumer.as_asgi()),
]
