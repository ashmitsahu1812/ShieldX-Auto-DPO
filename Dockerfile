FROM python:3.11-slim

# Set environment variables for better logging and non-root execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user and give them permissions to /app
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install dependencies first for better caching
COPY --chown=user requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# OpenEnv's LocalDockerProvider maps host_port -> container:8000 by default.
EXPOSE 8000

# Use PORT so Hugging Face Spaces can override it (often PORT=7860),
# while local Docker runs default to 8000.
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
