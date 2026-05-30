from django.contrib.auth.decorators import user_passes_test

def is_admin(user):
    return user.is_authenticated and user.is_superuser

def is_officer(user):
    return user.is_authenticated and (user.is_operator or user.is_superuser)

def is_road_user(user):
    return user.is_authenticated

def admin_required(function=None):
    """Decorator for views that checks that the user is an Administrator."""
    actual_decorator = user_passes_test(
        is_admin,
        login_url='/accounts/login/'
    )
    if function:
        return actual_decorator(function)
    return actual_decorator

def officer_required(function=None):
    """Decorator for views that checks that the user is an Officer or Admin."""
    actual_decorator = user_passes_test(
        is_officer,
        login_url='/accounts/login/'
    )
    if function:
        return actual_decorator(function)
    return actual_decorator

def road_user_required(function=None):
    """Decorator for views that checks that the user is authenticated."""
    actual_decorator = user_passes_test(
        is_road_user,
        login_url='/accounts/login/'
    )
    if function:
        return actual_decorator(function)
    return actual_decorator
