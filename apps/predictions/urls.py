from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('',        views.predictions_view, name='index'),
    path('routes/', views.routes_view,      name='routes'),
]
