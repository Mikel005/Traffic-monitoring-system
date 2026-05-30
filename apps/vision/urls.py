from django.urls import path
from . import views
app_name = 'vision'
urlpatterns = [
    path('', views.vision_status, name='status'),
    path('upload/', views.upload_video_view, name='upload_video'),
    path('inference/<int:video_id>/', views.video_inference, name='video_inference'),
    path('stream_feed/<int:video_id>/', views.stream_inference_feed, name='stream_inference_feed'),
    path('api/stats/<int:video_id>/', views.inference_stats_api, name='inference_stats_api'),
]
