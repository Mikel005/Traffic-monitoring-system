from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404

from apps.predictions.models import Prediction
from apps.predictions.ml import CongestionPredictor
from apps.traffic.models import Location
from apps.traffic.services import TrafficProcessor


@login_required
def routes_view(request):
    """RoadUser.getAlternativeRoutes() — shows congested locations + alternative routes."""
    locations = Location.objects.filter(is_active=True)
    congested = []
    for loc in locations:
        r = loc.latest_reading
        if r and r.congestion_index > 50:
            routes = TrafficProcessor.calculate_alternative_routes(loc)
            congested.append({
                'location': loc,
                'reading':  r,
                'routes':   routes,
            })
    return render(request, 'predictions/routes.html', {
        'congested': congested,
        'locations': locations,
    })


@login_required
def predictions_view(request):
    """Shows prediction charts for each location."""
    locations = Location.objects.filter(is_active=True)
    pred_data = []
    for loc in locations:
        pred = CongestionPredictor.predict(loc)
        latest_pred = loc.predictions.order_by('-timestamp').first()
        pred_data.append({
            'location':    loc,
            'prediction':  pred,
            'stored_pred': latest_pred,
        })
    return render(request, 'predictions/index.html', {'pred_data': pred_data})
