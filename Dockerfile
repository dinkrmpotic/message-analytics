FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./
RUN uv sync --no-dev

COPY models/ ./models/
COPY src/api/ ./src/api/

EXPOSE 8000

ENV MODEL_PATH=/app/models/spam_classifier.joblib

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
