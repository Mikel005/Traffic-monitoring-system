# TrafficIQ: Technical Project Manual
**Real-Time Traffic Monitoring & Predictive Analytics System**

---

## 1. Project Overview
TrafficIQ is a comprehensive traffic management platform designed for the Enugu metropolitan area. It provides real-time monitoring of road segments, AI-driven congestion forecasting, and automated alerting via a robust distributed architecture.

---

## 2. Technology Stack

### Backend (The Brain)
- **Django 5.0**: The core web framework providing the ORM, URL routing, and authentication.
- **Django Channels**: Enables **WebSockets** for real-time data streaming without page refreshes.
- **Django REST Framework (DRF)**: Powers the JSON API endpoints used by the frontend and external services.

### Real-Time & Background Processing
- **Redis**: Acts as the message broker and caching layer.
- **Celery**: Manages heavy background tasks (like calculating analytics) to keep the UI fast.
- **Celery Beat**: A scheduler that triggers the "data fetch" task every 60 seconds to simulate/ingest live traffic.

### Database (Storage)
- **SQLite (Development)**: Portable and simple for initial setup.
- **PostgreSQL / TimescaleDB (Production)**: Optimized for time-series data, allowing fast queries across millions of historical traffic readings.

### Frontend (The Face)
- **Bootstrap 5 & Vanilla CSS**: Modern, responsive UI with custom "Glassmorphism" dark-theme aesthetics.
- **Chart.js**: Dynamic, interactive charts for historical trends and predictions.
- **Vanilla JavaScript (ES6)**: Lightweight client-side logic for WebSocket handling and UI updates.

### Machine Learning
- **LSTM (Long Short-Term Memory)**: A neural network used for time-series forecasting (predicting traffic 60 minutes out).
- **XGBoost**: A gradient boosting algorithm used for high-accuracy congestion classification.

---

## 3. Core Architecture & Workflow

### A. Data Ingestion & Simulation
1. **Trigger**: Every minute, **Celery Beat** sends a signal.
2. **Task**: The `fetch_traffic_data` task runs in the background.
3. **Fetching**: It either connects to the **TomTom Traffic API** or uses a sophisticated **Mathematical Simulation** (modeling rush hour peaks using Gaussian distributions).
4. **Storage**: Data is saved to the `TrafficReading` table.

### B. The Alerting Engine
1. Once a reading is saved, an **Alert Task** is fired.
2. If the `congestion_index` exceeds a threshold (e.g., > 80% for "Gridlock"), an **Alert** record is created.
3. The system immediately sends a **WebSocket Message** to all connected users.

### C. Live Updates (WebSockets)
1. When you open the Dashboard, your browser establishes a **WebSocket connection** (`ws://...`).
2. When new data arrives on the server, Django Channels broadcasts a JSON packet.
3. The browser receives this packet and updates the KPI cards and graphs **instantly** without the user clicking refresh.

---

## 4. Logical Components (The Code)

### `apps.traffic`
- **Models**: Defines `Location` (road segments) and `TrafficReading` (the time-series data).
- **Views**: Handles the Dashboard metrics computation and the Live Map data.

### `apps.predictions`
- Uses historical data to feed ML models.
- Provides the "Predictive Analysis" page where users see forecasted traffic levels for the next 15, 30, and 60 minutes.

### `apps.alerts`
- Manages the lifecycle of an alert (Active → Resolved).
- Includes a **Context Processor** to show the red alert count bubble on every page.

---

## 5. From Setup to Deployment

1. **Environment Config**: Uses `.env` files to store sensitive keys (Google Maps, Database URLs).
2. **Migrations**: Django ORM generates the database tables based on Python classes.
3. **Data Seeding**: A custom management command (`seed_data`) populates the system with 24 hours of "fake" history so developers can work without waiting for live data.

---

## 6. How to Export to PDF

To share this manual as a standard document:
1. **VS Code**: Install the extension "Markdown PDF" -> Right-click this file -> **Markdown PDF: Export (pdf)**.
2. **Chrome/Edge**: Open the Markdown file in a viewer -> Right-click -> **Print** -> **Save as PDF**.
3. **Pandoc**: Run `pandoc manual.md -o manual.pdf`.
