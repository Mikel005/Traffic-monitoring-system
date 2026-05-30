import json
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from django.db.models import Avg, Count

from apps.traffic.models import Location, TrafficReading, CongestionLevel
from apps.traffic.services import SystemController, TrafficProcessor, DatabaseManager
from apps.alerts.models import Alert
from apps.predictions.models import Prediction
import math


@login_required
def dashboard(request):
    """Main dashboard — role-adaptive widgets."""
    locations   = Location.objects.filter(is_active=True).prefetch_related('readings')
    recent_alerts = Alert.objects.filter(is_resolved=False).order_by('-timestamp')[:5]
    total_vehicles = 0
    avg_congestion = 0
    gridlock_count = 0

    loc_data = []
    for loc in locations:
        r = loc.latest_reading
        if r:
            total_vehicles += r.vehicle_count
            avg_congestion  += r.congestion_index
            if r.congestion_level == CongestionLevel.GRIDLOCK:
                gridlock_count += 1
            loc_data.append({
                'id':          loc.pk,
                'name':        loc.name,
                'lat':         loc.latitude,
                'lng':         loc.longitude,
                'congestion':  r.congestion_index,
                'level':       r.congestion_level,
                'color':       r.level_color,
                'speed':       r.avg_speed,
                'vehicles':    r.vehicle_count,
            })

    n = len(loc_data)
    avg_congestion = round(avg_congestion / n, 1) if n else 0

    # 24-hour chart data
    chart_labels, chart_data = _get_chart_data()

    return render(request, 'traffic/dashboard.html', {
        'locations':      locations,
        'loc_json':       json.dumps(loc_data),
        'recent_alerts':  recent_alerts,
        'total_vehicles': total_vehicles,
        'avg_congestion': avg_congestion,
        'gridlock_count': gridlock_count,
        'active_locations': n,
        'chart_labels':   json.dumps(chart_labels),
        'chart_data':     json.dumps(chart_data),
    })


def _get_chart_data():
    """Build last-24-hours average congestion chart data."""
    now    = timezone.now()
    labels = []
    data   = []
    for h in range(23, -1, -1):
        t_start = now - timedelta(hours=h + 1)
        t_end   = now - timedelta(hours=h)
        agg = TrafficReading.objects.filter(
            timestamp__gte=t_start, timestamp__lt=t_end
        ).aggregate(avg=Avg('congestion_index'))
        labels.append(t_end.strftime('%H:%M'))
        data.append(round(agg['avg'] or 0, 1))
    return labels, data


@login_required
def map_view(request):
    locations = Location.objects.filter(is_active=True)
    loc_data  = []
    for loc in locations:
        r = loc.latest_reading
        loc_data.append({
            'id':       loc.pk,
            'name':     loc.name,
            'lat':      loc.latitude,
            'lng':      loc.longitude,
            'congestion': r.congestion_index if r else 0,
            'level':    r.congestion_level   if r else 'free_flow',
            'color':    r.level_color        if r else '#22c55e',
            'speed':    r.avg_speed          if r else 0,
            'vehicles': r.vehicle_count      if r else 0,
        })
    return render(request, 'traffic/map.html', {
        'locations': locations,
        'loc_json':  json.dumps(loc_data),
    })


@login_required
def location_detail(request, pk):
    location = get_object_or_404(Location, pk=pk)
    readings = location.readings.order_by('-timestamp')[:48]
    analysis = TrafficProcessor.analyze_traffic_data(location, hours=24)
    prediction = location.predictions.order_by('-timestamp').first()
    routes = TrafficProcessor.calculate_alternative_routes(location)
    return render(request, 'traffic/location_detail.html', {
        'location':   location,
        'readings':   readings,
        'analysis':   analysis,
        'prediction': prediction,
        'routes':     routes,
    })


@login_required
def reports_view(request):
    if not (request.user.is_admin or request.user.is_officer):
        from django.shortcuts import redirect
        return redirect('traffic:dashboard')
    from apps.traffic.models import Report
    reports = Report.objects.select_related('location', 'generated_by').all()[:50]
    locations = Location.objects.filter(is_active=True)
    if request.method == 'POST':
        loc_id    = request.POST.get('location')
        rep_type  = request.POST.get('report_type', 'daily')
        days      = int(request.POST.get('days', 7))
        loc       = get_object_or_404(Location, pk=loc_id) if loc_id else None
        end       = timezone.now()
        start     = end - timedelta(days=days)
        DatabaseManager.save_report(
            title        = f"{rep_type.title()} Report — {loc or 'All Locations'}",
            report_type  = rep_type,
            location     = loc,
            period_start = start,
            period_end   = end,
            generated_by = request.user,
        )
        from django.contrib import messages
        messages.success(request, 'Report generated successfully.')
    return render(request, 'traffic/reports.html', {
        'reports':   reports,
        'locations': locations,
    })

@login_required
def corridor_view(request):
    locations = Location.objects.filter(is_active=True)
    return render(request, 'traffic/corridor.html', {
        'locations': locations,
    })


# ── AJAX API endpoints ────────────────────────────────────────────
@login_required
def api_live(request):
    """AJAX: live snapshot of all locations."""
    locs = Location.objects.filter(is_active=True)
    data = []
    for loc in locs:
        r = loc.latest_reading
        data.append({
            'id':          loc.pk,
            'name':        loc.name,
            'lat':         loc.latitude,
            'lng':         loc.longitude,
            'congestion':  round(r.congestion_index, 1) if r else 0,
            'level':       r.congestion_level            if r else 'free_flow',
            'color':       r.level_color                 if r else '#22c55e',
            'speed':       round(r.avg_speed, 1)         if r else 0,
            'vehicles':    r.vehicle_count               if r else 0,
            'timestamp':   r.timestamp.isoformat()       if r else None,
        })
    unresolved_alerts = Alert.objects.filter(is_resolved=False).count()
    labels, chart = _get_chart_data()
    return JsonResponse({
        'locations':    data,
        'alert_count':  unresolved_alerts,
        'chart_labels': labels,
        'chart_data':   chart,
    })


@login_required
def api_location_data(request, pk):
    """AJAX: recent readings for one location (for sparkline)."""
    loc      = get_object_or_404(Location, pk=pk)
    readings = loc.readings.order_by('-timestamp')[:20]
    return JsonResponse({
        'labels': [r.timestamp.strftime('%H:%M') for r in reversed(list(readings))],
        'congestion': [round(r.congestion_index, 1) for r in reversed(list(readings))],
        'speed': [round(r.avg_speed, 1) for r in reversed(list(readings))],
    })

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

@login_required
def api_nearest_location(request):
    """AJAX: Find closest location to given lat/lng."""
    try:
        lat = float(request.GET.get('lat', 0))
        lng = float(request.GET.get('lng', 0))
    except ValueError:
        return JsonResponse({'error': 'Invalid coordinates'}, status=400)
        
    locations = Location.objects.filter(is_active=True)
    nearest = None
    min_dist = float('inf')
    
    for loc in locations:
        dist = _haversine(lat, lng, loc.latitude, loc.longitude)
        if dist < min_dist:
            min_dist = dist
            nearest = loc
            
    if nearest:
        return JsonResponse({'id': nearest.pk, 'name': nearest.name, 'distance_km': round(min_dist, 2)})
    return JsonResponse({'error': 'No active locations found'}, status=404)

@login_required
def api_corridor_predictions(request):
    """AJAX: Calculate corridor between two locations and return predictions."""
    start_id = request.GET.get('start')
    end_id = request.GET.get('end')
    
    if not start_id or not end_id:
        return JsonResponse({'error': 'Missing start or end location'}, status=400)
        
    start_loc = get_object_or_404(Location, pk=start_id)
    end_loc = get_object_or_404(Location, pk=end_id)
    
    # Bounding box heuristic
    min_lat = min(start_loc.latitude, end_loc.latitude) - 0.02
    max_lat = max(start_loc.latitude, end_loc.latitude) + 0.02
    min_lng = min(start_loc.longitude, end_loc.longitude) - 0.02
    max_lng = max(start_loc.longitude, end_loc.longitude) + 0.02
    
    corridor_locs = Location.objects.filter(
        is_active=True,
        latitude__gte=min_lat, latitude__lte=max_lat,
        longitude__gte=min_lng, longitude__lte=max_lng
    )
    
    loc_list = list(corridor_locs)
    loc_list.sort(key=lambda l: _haversine(start_loc.latitude, start_loc.longitude, l.latitude, l.longitude))
    
    if start_loc in loc_list:
        loc_list.remove(start_loc)
    if end_loc in loc_list:
        loc_list.remove(end_loc)
        
    # Ordered path
    final_path = [start_loc] + loc_list + [end_loc]
    
    if len(final_path) > 5:
        step = len(loc_list) / 3.0
        final_path = [start_loc] + [loc_list[int(i*step)] for i in range(3)] + [end_loc]
    
    from apps.predictions.ml import CongestionPredictor
    
    results = []
    total_dist = 0
    for i, loc in enumerate(final_path):
        pred = CongestionPredictor.predict(loc)
        results.append({
            'id': loc.pk,
            'name': loc.name,
            'prediction': {
                'm15': TrafficReading.index_to_level(pred.get('minutes_15', 0)),
                'm30': TrafficReading.index_to_level(pred.get('minutes_30', 0)),
                'm60': TrafficReading.index_to_level(pred.get('minutes_60', 0)),
            }
        })
        if i < len(final_path) - 1:
            total_dist += _haversine(final_path[i].latitude, final_path[i].longitude, 
                                     final_path[i+1].latitude, final_path[i+1].longitude)
                                     
    # Avg speed 30km/h in city
    est_time_mins = int((max(total_dist, 1) / 30.0) * 60)
    
    return JsonResponse({
        'locations': results,
        'distance_km': round(total_dist, 1),
        'est_time_mins': est_time_mins
    })
