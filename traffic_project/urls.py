from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/',       admin.site.urls),
    path('accounts/',    include('apps.accounts.urls',   namespace='accounts')),
    path('',             include('apps.traffic.urls',    namespace='traffic')),
    path('alerts/',      include('apps.alerts.urls',     namespace='alerts')),
    path('devices/',     include('apps.devices.urls',    namespace='devices')),
    path('predictions/', include('apps.predictions.urls',namespace='predictions')),
    path('vision/',      include('apps.vision.urls',     namespace='vision')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
