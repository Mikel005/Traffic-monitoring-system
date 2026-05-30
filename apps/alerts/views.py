from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone

from apps.alerts.models import Alert
from apps.alerts.services import NotificationService


@login_required
def alert_list(request):
    alerts     = Alert.objects.select_related('location').order_by('-timestamp')[:100]
    unresolved = Alert.objects.filter(is_resolved=False).count()
    return render(request, 'alerts/index.html', {
        'alerts':     alerts,
        'unresolved': unresolved,
    })


@login_required
def resolve_alert(request, pk):
    if not (request.user.is_admin or request.user.is_officer):
        messages.error(request, 'Permission denied.')
        return redirect('alerts:list')
    alert = get_object_or_404(Alert, pk=pk)
    alert.resolve(user=request.user)
    messages.success(request, f'Alert #{pk} resolved.')
    return redirect('alerts:list')


@login_required
def resolve_all(request):
    if not (request.user.is_admin or request.user.is_officer):
        messages.error(request, 'Permission denied.')
        return redirect('alerts:list')
    Alert.objects.filter(is_resolved=False).update(
        is_resolved=True,
        resolved_at=timezone.now(),
    )
    messages.success(request, 'All alerts resolved.')
    return redirect('alerts:list')


@login_required
def api_alerts(request):
    """AJAX: latest 10 unresolved alerts."""
    alerts = Alert.objects.filter(is_resolved=False).order_by('-timestamp')[:10]
    data = [{
        'id':       a.pk,
        'location': a.location.name,
        'level':    a.level,
        'severity': a.severity,
        'message':  a.message,
        'color':    a.level_color,
        'time':     a.timestamp.strftime('%H:%M'),
    } for a in alerts]
    return JsonResponse({'alerts': data, 'count': len(data)})
