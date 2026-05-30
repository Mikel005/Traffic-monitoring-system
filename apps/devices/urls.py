from django.urls import path
from . import views

app_name = 'devices'

urlpatterns = [
    path('',              views.device_list,     name='list'),
    path('camera/add/',   views.add_camera,      name='add_camera'),
    path('sensor/add/',   views.add_sensor,      name='add_sensor'),
    path('api/poll/',     views.api_poll_devices, name='poll'),
]
