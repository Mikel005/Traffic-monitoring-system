from django.urls import path
from . import views

app_name = 'traffic'

urlpatterns = [
    path('',                views.dashboard,         name='dashboard'),
    path('map/',            views.map_view,           name='map'),
    path('location/<int:pk>/', views.location_detail, name='location_detail'),
    path('reports/',        views.reports_view,       name='reports'),
    path('corridor/',       views.corridor_view,      name='corridor'),

    # AJAX endpoints
    path('api/live/',       views.api_live,           name='api_live'),
    path('api/location/<int:pk>/', views.api_location_data, name='api_location_data'),
    path('api/nearest/',    views.api_nearest_location, name='api_nearest_location'),
    path('api/corridor/',   views.api_corridor_predictions, name='api_corridor_predictions'),
]
