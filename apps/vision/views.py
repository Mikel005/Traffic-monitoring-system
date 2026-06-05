import threading
import sys
import os

from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from loguru import logger

from apps.accounts.permissions import admin_required
from .models import UploadedVideo, Camera
from .forms import VideoUploadForm, CameraForm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.src.vehicle_detector import VehicleDetector


# ── Background inference ──────────────────────────────────────────────

def _run_inference_thread(video_id: int):
    from django.db import close_old_connections, connection as db_connection
    close_old_connections()
    try:
        video    = UploadedVideo.objects.get(pk=video_id)
        detector = VehicleDetector(model_path=str(settings.YOLO_WEIGHTS_PATH))
        detector.process_video(video.video_file.path, video)
    except Exception as exc:
        logger.error(f"Background inference failed for video {video_id}: {exc}")
        try:
            UploadedVideo.objects.filter(pk=video_id).update(status='failed')
        except Exception:
            pass
    finally:
        db_connection.close()


# ── Video views ───────────────────────────────────────────────────────

@login_required
def vision_status(request):
    from apps.traffic.models import TrafficReading

    videos = UploadedVideo.objects.select_related('location', 'camera').order_by('-uploaded_at')[:30]
    vision_readings = (
        TrafficReading.objects
        .filter(source='vision')
        .select_related('location')
        .order_by('-timestamp')[:20]
    )
    cameras = Camera.objects.select_related('location').order_by('name')

    return render(request, 'vision/status.html', {
        'videos':          videos,
        'vision_readings': vision_readings,
        'cameras':         cameras,
        'stats': {
            'total':      UploadedVideo.objects.count(),
            'completed':  UploadedVideo.objects.filter(status='completed').count(),
            'processing': UploadedVideo.objects.filter(status='processing').count(),
            'failed':     UploadedVideo.objects.filter(status='failed').count(),
        },
        'page': 'vision',
    })


@admin_required
def upload_video_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video        = form.save()
            video.status = 'processing'
            video.save(update_fields=['status'])
            threading.Thread(
                target=_run_inference_thread,
                args=(video.id,),
                daemon=True,
            ).start()
            messages.success(request, "Video uploaded — analysis is now running.")
            return redirect('vision:video_inference', video_id=video.id)
    else:
        form = VideoUploadForm()
    return render(request, 'vision/upload.html', {'form': form, 'page': 'upload'})


@login_required
def video_inference(request, video_id):
    video = get_object_or_404(UploadedVideo, pk=video_id)
    return render(request, 'vision/inference.html', {'video': video, 'page': 'vision'})


@login_required
def stream_inference_feed(request, video_id):
    video    = get_object_or_404(UploadedVideo, pk=video_id)
    detector = VehicleDetector(model_path=str(settings.YOLO_WEIGHTS_PATH))
    return StreamingHttpResponse(
        detector.stream_inference(video.video_file.path, video),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )


@login_required
def inference_stats_api(request, video_id):
    video = get_object_or_404(UploadedVideo, pk=video_id)

    # Derive which pipeline stage is active from status
    stage_map = {
        'pending':    0,
        'processing': 2,   # YOLO running
        'completed':  4,
        'failed':     -1,
    }
    active_stage = stage_map.get(video.status, 0)

    return JsonResponse({
        'status':                     video.status,
        'vehicle_count':              video.vehicle_count,
        'average_speed':              round(video.average_speed, 1),
        'predicted_congestion_level': video.predicted_congestion_level or 'ANALYZING',
        'car_count':                  video.car_count,
        'truck_count':                video.truck_count,
        'motorcycle_count':           video.motorcycle_count,
        'bus_count':                  video.bus_count,
        'queue_length':               video.queue_length,
        'active_stage':               active_stage,
        'congestion_index':           min(video.vehicle_count * 2, 100),
    })


# ── Camera views ──────────────────────────────────────────────────────

@admin_required
def camera_list(request):
    cameras = Camera.objects.select_related('location').order_by('name')
    return render(request, 'vision/cameras.html', {
        'cameras': cameras,
        'page':    'vision',
    })


@admin_required
def camera_create(request):
    if request.method == 'POST':
        form = CameraForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Camera added successfully.")
            return redirect('vision:cameras')
    else:
        form = CameraForm()
    return render(request, 'vision/camera_form.html', {
        'form':  form,
        'title': 'Add Camera',
        'page':  'vision',
    })


@admin_required
def camera_edit(request, camera_id):
    camera = get_object_or_404(Camera, pk=camera_id)
    if request.method == 'POST':
        form = CameraForm(request.POST, instance=camera)
        if form.is_valid():
            form.save()
            messages.success(request, "Camera updated.")
            return redirect('vision:cameras')
    else:
        form = CameraForm(instance=camera)
    return render(request, 'vision/camera_form.html', {
        'form':   form,
        'camera': camera,
        'title':  'Edit Camera',
        'page':   'vision',
    })


@admin_required
def camera_delete(request, camera_id):
    camera = get_object_or_404(Camera, pk=camera_id)
    if request.method == 'POST':
        camera.delete()
        messages.success(request, "Camera removed.")
    return redirect('vision:cameras')


@login_required
def camera_stream_view(request, camera_id):
    camera = get_object_or_404(Camera, pk=camera_id, is_active=True)
    return render(request, 'vision/camera_stream.html', {
        'camera': camera,
        'page':   'vision',
    })


@login_required
def camera_stream_feed(request, camera_id):
    camera   = get_object_or_404(Camera, pk=camera_id, is_active=True)
    detector = VehicleDetector(model_path=str(settings.YOLO_WEIGHTS_PATH))

    # Build a transient UploadedVideo-like proxy so _sync_to_traffic can use location
    class _CameraProxy:
        def __init__(self, loc):
            self.location = loc
            self.vehicle_count              = 0
            self.average_speed              = 0.0
            self.predicted_congestion_level = ''
            self.car_count = self.truck_count = self.motorcycle_count = self.bus_count = 0
            self.status    = 'processing'

        def save(self, update_fields=None):
            pass  # live stream — no DB row to update

    proxy = _CameraProxy(camera.location)
    source = int(camera.stream_url) if camera.stream_url.isdigit() else camera.stream_url

    return StreamingHttpResponse(
        detector.stream_inference(source, proxy),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )
