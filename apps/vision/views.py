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
from .models import UploadedVideo
from .forms import VideoUploadForm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.src.vehicle_detector import VehicleDetector


# ── Background inference ──────────────────────────────────────────────

def _run_inference_thread(video_id: int):
    """
    Runs in a daemon thread after video upload.
    Processes the full video with YOLO and writes results back to the DB.
    Always closes the thread-local DB connection on exit to avoid leaks.
    """
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


# ── Views ─────────────────────────────────────────────────────────────

@login_required
def vision_status(request):
    """
    Main vision page — uploaded videos, processing state, and the traffic
    readings that were synced back from vision analysis results.
    """
    from apps.traffic.models import TrafficReading

    videos = UploadedVideo.objects.select_related('location').order_by('-uploaded_at')[:30]
    vision_readings = (
        TrafficReading.objects
        .filter(source='vision')
        .select_related('location')
        .order_by('-timestamp')[:20]
    )

    return render(request, 'vision/status.html', {
        'videos':          videos,
        'vision_readings': vision_readings,
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
    """Upload a traffic video; inference runs in a background thread."""
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

            messages.success(request, "Video uploaded — analysis running in background.")
            return redirect('vision:status')
    else:
        form = VideoUploadForm()

    return render(request, 'vision/upload.html', {'form': form, 'page': 'upload'})


@login_required
def video_inference(request, video_id):
    """Live MJPEG viewing page for a specific uploaded video."""
    video = get_object_or_404(UploadedVideo, pk=video_id)
    return render(request, 'vision/inference.html', {'video': video, 'page': 'vision'})


@login_required
def stream_inference_feed(request, video_id):
    """MJPEG stream endpoint — runs YOLO on the video file for live browser display."""
    video    = get_object_or_404(UploadedVideo, pk=video_id)
    detector = VehicleDetector(model_path=str(settings.YOLO_WEIGHTS_PATH))
    return StreamingHttpResponse(
        detector.stream_inference(video.video_file.path, video),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )


@login_required
def inference_stats_api(request, video_id):
    """JSON polling endpoint — returns current processing stats for a video."""
    video = get_object_or_404(UploadedVideo, pk=video_id)
    return JsonResponse({
        'status':                     video.status,
        'vehicle_count':              video.vehicle_count,
        'average_speed':              round(video.average_speed, 1),
        'predicted_congestion_level': video.predicted_congestion_level or 'ANALYZING',
    })
