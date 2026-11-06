FROM python:3.12-slim
RUN useradd --create-home appuser && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[ml,api]"
ENV HF_HOME=/app/hf-cache PYTHONPATH=/app/src:/app PORT=8000
# Bake both models into the image so the container starts without network access to Hugging Face
RUN python -c "from newsmood.models import load; load()" && chown -R appuser /app
COPY api ./api
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
