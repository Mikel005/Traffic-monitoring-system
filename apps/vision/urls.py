from django.urls import path
from . import views

app_name = 'vision'

urlpatterns = [
    # ── Video ──────────────────────────────────────────────────────
    path('',                              views.vision_status,        name='status'),
    path('upload/',                       views.upload_video_view,    name='upload_video'),
    path('inference/<int:video_id>/',     views.video_inference,      name='video_inference'),
    path('stream_feed/<int:video_id>/',   views.stream_inference_feed,name='stream_inference_feed'),
    path('api/stats/<int:video_id>/',     views.inference_stats_api,  name='inference_stats_api'),

    # ── Count sessions & CSV ──────────────────────────────────────
    path('sessions/',                        views.session_list,         name='sessions'),
    path('sessions/<int:session_id>/csv/',   views.download_csv,         name='download_csv'),

    # ── Camera management ──────────────────────────────────────────
    path('cameras/',                      views.camera_list,          name='cameras'),
    path('cameras/add/',                  views.camera_create,        name='camera_create'),
    path('cameras/<int:camera_id>/edit/', views.camera_edit,          name='camera_edit'),
    path('cameras/<int:camera_id>/delete/',views.camera_delete,       name='camera_delete'),
    path('cameras/<int:camera_id>/live/', views.camera_stream_view,   name='camera_live'),
    path('cameras/<int:camera_id>/feed/', views.camera_stream_feed,   name='camera_feed'),
]
