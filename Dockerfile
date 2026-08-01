# Multi-stage production build
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

COPY --from=builder /install /usr/local

COPY . /app

# Change permissions
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
