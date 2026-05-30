import os
import django
from django.conf import settings
from django.template import loader, TemplateDoesNotExist

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'traffic_project.settings')
django.setup()

templates_to_check = [
    'base.html',
    'traffic/dashboard.html',
    'traffic/location_detail.html',
    'traffic/map.html',
    'alerts/alerts.html',
    'predictions/predictions.html',
    'accounts/login.html'
]

print("Checking templates for syntax errors...")
any_errors = False
for t_path in templates_to_check:
    try:
        loader.get_template(t_path)
        print(f"OK: {t_path}")
    except Exception as e:
        print(f"FAIL: {t_path} -> {e}")
        any_errors = True

if not any_errors:
    print("All templates are syntactically valid.")
else:
    print("Some templates have errors.")
