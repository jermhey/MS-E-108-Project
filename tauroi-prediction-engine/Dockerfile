# ─────────────────────────────────────────────────────────────────────────────
#  Tauroi Prediction Engine — Production Container
# ─────────────────────────────────────────────────────────────────────────────
#  Build:   docker build -t tauroi .
#  Run:     docker run --env-file .env tauroi
#  Loop:    docker run --env-file .env tauroi python main.py --loop --interval 900
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY main.py .
COPY run_backtest.py .

# Create the signal log directory
RUN mkdir -p signals

# Default: single run (GitHub Actions / cron compatible)
CMD ["python", "main.py"]
