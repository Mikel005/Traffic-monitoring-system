from django.urls import path
from . import views

app_name = 'alerts'

urlpatterns = [
    path('',              views.alert_list,   name='list'),
    path('<int:pk>/resolve/', views.resolve_alert, name='resolve'),
    path('resolve-all/',  views.resolve_all,  name='resolve_all'),
    path('api/',          views.api_alerts,   name='api'),
]
