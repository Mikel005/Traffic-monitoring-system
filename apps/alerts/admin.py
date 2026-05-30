from django.contrib import admin
from .models import Alert

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display  = ['location', 'level', 'congestion_index', 'timestamp', 'is_resolved']
    list_filter   = ['level', 'is_resolved']
    date_hierarchy = 'timestamp'
    actions       = ['resolve_selected']

    @admin.action(description='Mark selected alerts as resolved')
    def resolve_selected(self, request, queryset):
        queryset.update(is_resolved=True)
