from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from apps.accounts.models import User, TrafficAdministrator, TrafficOfficer, RoadUser


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display  = ('username', 'email', 'role', 'is_active', 'date_joined')
    list_filter   = ('role', 'is_active', 'is_staff')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    fieldsets     = UserAdmin.fieldsets + (
        ('Traffic System', {'fields': ('role', 'phone')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Traffic System', {'fields': ('role', 'phone')}),
    )


@admin.register(TrafficAdministrator)
class TrafficAdministratorAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'is_active')


@admin.register(TrafficOfficer)
class TrafficOfficerAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'is_active')


@admin.register(RoadUser)
class RoadUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'email', 'is_active')
