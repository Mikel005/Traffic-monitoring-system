from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse

from apps.devices.models import Camera, IoTSensor, DeviceController
from apps.traffic.models import Location


@login_required
def device_list(request):
    if not request.user.is_admin:
        from django.contrib import messages
        from django.shortcuts import redirect
        messages.error(request, 'Access restricted to administrators.')
        return redirect('traffic:dashboard')
    cameras      = Camera.objects.select_related('location').all()
    sensors      = IoTSensor.objects.select_related('location').all()
    controllers  = DeviceController.objects.prefetch_related(
        'managed_cameras', 'managed_sensors'
    ).all()
    locations    = Location.objects.filter(is_active=True)
    return render(request, 'devices/index.html', {
        'cameras':     cameras,
        'sensors':     sensors,
        'controllers': controllers,
        'locations':   locations,
    })


@login_required
def add_camera(request):
    if not request.user.is_admin:
        from django.shortcuts import redirect
        return redirect('traffic:dashboard')
    if request.method == 'POST':
        loc  = get_object_or_404(Location, pk=request.POST.get('location'))
        Camera.objects.create(
            name       = request.POST.get('name'),
            location   = loc,
            stream_url = request.POST.get('stream_url', ''),
            resolution = request.POST.get('resolution', '1080p'),
            direction  = request.POST.get('direction', ''),
        )
        from django.contrib import messages
        messages.success(request, 'Camera added successfully.')
    from django.shortcuts import redirect
    return redirect('devices:list')


@login_required
def add_sensor(request):
    if not request.user.is_admin:
        from django.shortcuts import redirect
        return redirect('traffic:dashboard')
    if request.method == 'POST':
        loc = get_object_or_404(Location, pk=request.POST.get('location'))
        import uuid
        IoTSensor.objects.create(
            name        = request.POST.get('name'),
            location    = loc,
            sensor_type = request.POST.get('sensor_type', 'volume'),
            device_id   = request.POST.get('device_id') or str(uuid.uuid4())[:8].upper(),
        )
        from django.contrib import messages
        messages.success(request, 'IoT Sensor added successfully.')
    from django.shortcuts import redirect
    return redirect('devices:list')


@login_required
def api_poll_devices(request):
    """AJAX: poll all active device controllers and return results."""
    results = []
    for ctrl in DeviceController.objects.filter(is_active=True):
        results.append({
            'controller': ctrl.name,
            'data':       ctrl.poll_all_devices(),
        })
    return JsonResponse({'results': results})
