"""
Django Settings — Real-Time Intelligent Traffic Monitoring & Congestion Prediction
Runs with zero external services: SQLite DB, in-memory cache, APScheduler for background jobs.
"""
import environ
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Environment ───────────────────────────────────────────────────
env = environ.Env(
    DEBUG=(bool, True),
    USE_MOCK_DATA=(bool, True),
    MOCK_LOCATIONS=(int, 8),
)
environ.Env.read_env(BASE_DIR / '.env')

# ── Core ──────────────────────────────────────────────────────────
SECRET_KEY    = env('SECRET_KEY')
DEBUG         = env('DEBUG')
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=['localhost', '127.0.0.1', '*'])

CSRF_TRUSTED_ORIGINS = [
    'https://*.github.dev',
    'https://*.app.github.dev',
    'https://*.google.com',
    'https://*.idx.google.com',
    'http://localhost',
    'http://127.0.0.1',
]

# ── Applications ──────────────────────────────────────────────────
INSTALLED_APPS = [
    # Django built-ins
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',

    # Third-party
    'django_extensions',
    'django_apscheduler',

    # Project apps
    'apps.accounts',
    'apps.devices',
    'apps.traffic',
    'apps.alerts',
    'apps.predictions',
    'apps.vision',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF     = 'traffic_project.urls'
WSGI_APPLICATION = 'traffic_project.wsgi.application'

# ── Templates ─────────────────────────────────────────────────────
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'apps.alerts.context_processors.alert_count',
            ],
        },
    },
]

# ── Database ──────────────────────────────────────────────────────
DATABASES = {
    'default': env.db('DATABASE_URL', default=f'sqlite:///{BASE_DIR / "db.sqlite3"}')
}
if DATABASES['default']['ENGINE'] == 'django.db.backends.sqlite3':
    DATABASES['default']['OPTIONS'] = {
        'timeout': 30,
    }
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Cache (in-memory, no Redis) ───────────────────────────────────
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'traffic-cache',
    }
}

# ── Auth ──────────────────────────────────────────────────────────
AUTH_USER_MODEL     = 'accounts.User'
LOGIN_URL           = '/accounts/login/'
LOGIN_REDIRECT_URL  = '/'
LOGOUT_REDIRECT_URL = '/accounts/login/'

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ── Static & Media ────────────────────────────────────────────────
STATIC_URL       = '/static/'
STATIC_ROOT      = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ── Internationalisation ──────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'Africa/Lagos'
USE_I18N      = True
USE_TZ        = True

# ── APScheduler ───────────────────────────────────────────────────
APSCHEDULER_DATETIME_FORMAT = "N j, Y, f:s a"
APSCHEDULER_RUN_NOW_TIMEOUT = 25   # seconds

# ── App-specific ──────────────────────────────────────────────────
USE_MOCK_DATA   = env('USE_MOCK_DATA')
MOCK_LOCATIONS  = env('MOCK_LOCATIONS')
TOMTOM_API_KEY  = env('TOMTOM_API_KEY', default='')
OPENWEATHER_API_KEY = env('OPENWEATHER_API_KEY', default='')

# ── ML Model Paths ────────────────────────────────────────────────
LSTM_MODEL_PATH   = BASE_DIR / env('LSTM_MODEL_PATH',   default='ml/saved_models/lstm_best.onnx')
XGB_MODEL_PATH    = BASE_DIR / env('XGB_MODEL_PATH',    default='ml/saved_models/xgb_classifier.pkl')
YOLO_WEIGHTS_PATH = BASE_DIR / env('YOLO_WEIGHTS_PATH', default='ml/saved_models/yolov8n.pt')
