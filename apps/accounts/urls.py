from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/',        views.login_view,          name='login'),
    path('logout/',       views.logout_view,          name='logout'),
    path('profile/',      views.profile_view,         name='profile'),
    path('users/',        views.manage_users_view,    name='manage_users'),
    path('users/<int:user_id>/role/', views.change_user_role_view, name='change_role'),
]
