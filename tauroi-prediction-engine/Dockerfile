# ─────────────────────────────────────────────────────────────────────────────
#  Tauroi Prediction Engine — Production Container
# ─────────────────────────────────────────────────────────────────────────────
#  Build:   docker build -t tauroi .
#  Run:     docker run --env-file .env tauroi
#  Scan:    docker run --env-file .env tauroi python main.py --scan --dry-run
#  Loop:    docker run --env-file .env tauroi python main.py --scan --loop --interval 3600
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
COPY tests/ tests/
COPY main.py .
COPY run_backtest.py .

# Create persistent directories
RUN mkdir -p signals cache

# Default: full market scan, single run (GitHub Actions / cron compatible)
CMD ["python", "main.py", "--scan", "--dry-run"]
