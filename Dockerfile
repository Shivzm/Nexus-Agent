FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config ./config
COPY src ./src
COPY tools ./tools
COPY api ./api

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
