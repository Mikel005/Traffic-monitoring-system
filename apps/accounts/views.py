from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_http_methods

from apps.accounts.models import User, ROLE_CHOICES, ROLE_ADMIN, ROLE_OFFICER, ROLE_ROAD_USER


@require_http_methods(['GET', 'POST'])
def login_view(request):
    if request.user.is_authenticated:
        return redirect('traffic:dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect(request.GET.get('next', '/'))
        messages.error(request, 'Invalid username or password.')

    return render(request, 'accounts/login.html')


def logout_view(request):
    logout(request)
    return redirect('accounts:login')


@login_required
def profile_view(request):
    return render(request, 'accounts/profile.html', {'user': request.user})


@login_required
def manage_users_view(request):
    """TrafficAdministrator.manageUsers()"""
    if not request.user.is_admin:
        messages.error(request, 'Access denied.')
        return redirect('traffic:dashboard')
    users = User.objects.all().order_by('role', 'username')
    return render(request, 'accounts/manage_users.html', {'users': users, 'roles': ROLE_CHOICES})


@login_required
@require_http_methods(['POST'])
def change_user_role_view(request, user_id):
    if not request.user.is_admin:
        messages.error(request, 'Access denied.')
        return redirect('traffic:dashboard')
    target = get_object_or_404(User, pk=user_id)
    new_role = request.POST.get('role')
    if new_role in dict(ROLE_CHOICES):
        target.role = new_role
        target.save(update_fields=['role'])
        messages.success(request, f"Role updated for {target.username}")
    return redirect('accounts:manage_users')
