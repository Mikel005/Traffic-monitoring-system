.PHONY: help run migrate seed celery beat test shell

help:
	@echo ""
	@echo "  Traffic Monitor — Django Commands"
	@echo "  ──────────────────────────────────"
	@echo "  make run      Start Django dev server (localhost:8000)"
	@echo "  make docker   Start all services with Docker"
	@echo "  make migrate  Apply database migrations"
	@echo "  make seed     Seed Enugu locations + mock data"
	@echo "  make celery   Start Celery worker"
	@echo "  make beat     Start Celery Beat scheduler"
	@echo "  make test     Run pytest"
	@echo "  make shell    Open Django shell"
	@echo ""

run:
	python manage.py runserver 0.0.0.0:8000

docker:
	docker compose up -d
	@echo "✅ App running at http://localhost:8000"

docker-build:
	docker compose build --no-cache

docker-down:
	docker compose down

migrate:
	python manage.py migrate

seed:
	python manage.py seed_data

celery:
	celery -A traffic_project worker --loglevel=info

beat:
	celery -A traffic_project beat --loglevel=info \
	       --scheduler django_celery_beat.schedulers:DatabaseScheduler

test:
	pytest apps/ -v

shell:
	python manage.py shell

createsuperuser:
	python manage.py createsuperuser

static:
	python manage.py collectstatic --noinput
