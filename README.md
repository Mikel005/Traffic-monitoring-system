# 🚦 Real-Time Intelligent Traffic Monitoring System
### Django + TimescaleDB + Celery + WebSocket

> Final Year Project — Computer Science  
> Godfrey Okoye University, Enugu  
> Supervisor: Engr. Dr. Kingsley Chibueze

---

## Overview

A full-stack Django web application that monitors road traffic in real time across Enugu, Nigeria. The system:

- **Fetches traffic data** from TomTom API (or realistic mock data)
- **Predicts congestion** 15, 30, and 60 minutes ahead using LSTM
- **Classifies congestion** — Free Flow / Moderate / Heavy / Gridlock via XGBoost
- **Live dashboard** with Django Templates + Bootstrap 5 + Chart.js
- **WebSocket alerts** via Django Channels when thresholds are breached

---

## Tech Stack

| Layer      | Technology                                        |
|------------|---------------------------------------------------|
| Framework  | Django 5.0 + Django Channels (WebSocket)          |
| Database   | PostgreSQL 15 + TimescaleDB (time-series)         |
| Task Queue | Celery + django-celery-beat + Redis               |
| ML Models  | LSTM (PyTorch/ONNX), XGBoost, YOLOv8             |
| Frontend   | Django Templates + Bootstrap 5 + Chart.js         |
| Deployment | Docker Compose + Daphne (ASGI)                    |

---

## Quick Start — Windows (Docker Desktop)

### Prerequisites
- Docker Desktop (WSL 2 backend, 4 GB RAM minimum)

### Steps
```cmd
git clone https://github.com/yourusername/traffic-monitoring-system.git
cd traffic-monitoring-system
copy .env.example .env
docker compose up -d
```

Open http://localhost:8000 — Login: `admin` / `admin123`

---

## Local Development (no Docker)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
python manage.py migrate
python manage.py seed_data
python manage.py runserver
```

Start Celery in a separate terminal:
```cmd
celery -A traffic_project worker --loglevel=info
celery -A traffic_project beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler
```

---

## Train ML Models

```cmd
pip install -r ml/requirements.txt
python ml/src/train_lstm.py
python ml/src/train_xgb.py
python ml/src/evaluate.py
```

---

## Useful Commands

```cmd
make run       # Start dev server
make migrate   # Apply DB migrations
make seed      # Seed locations + mock data
make celery    # Start Celery worker
make beat      # Start Celery Beat
make test      # Run pytest
make shell     # Django shell
```

---

## Configuration (.env)

| Variable           | Default    | Description                      |
|--------------------|------------|----------------------------------|
| `USE_MOCK_DATA`    | `True`     | Simulated data (no API key needed)|
| `TOMTOM_API_KEY`   | *(empty)*  | TomTom Traffic Flow API key      |
| `DATABASE_URL`     | postgres://| TimescaleDB connection           |
| `REDIS_URL`        | redis://   | Redis connection                 |
| `SECRET_KEY`       | change-this| Django secret key                |
| `DEBUG`            | `True`     | Set False in production          |
