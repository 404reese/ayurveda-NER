FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AYURNER_CACHE_DIR=/app/cache \
    WORKERS=2 \
    PORT=8000

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[server,fuzzy]"

# Optional extra lexicons (e.g. a generated nia.csv) live in ./lexicons.
ARG AYURNER_LEXICONS=""
ENV AYURNER_LEXICONS=${AYURNER_LEXICONS}
COPY lexicons ./lexicons

# Compile the index at build time so containers start instantly.
RUN python -m ayurner build-index

EXPOSE 8000
CMD ["sh", "-c", "uvicorn --factory ayurner.server.app:get_app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
