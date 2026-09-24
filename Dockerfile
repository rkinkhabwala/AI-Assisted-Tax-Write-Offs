# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, for layer caching. The claude-agent-sdk wheel bundles the Claude
# Code CLI binary it drives, so no Node.js install is needed.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY migrations ./migrations
COPY prompts ./prompts
COPY data ./data
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# Apply pending migrations, then serve.
CMD ["sh", "-c", "python -m writeoff.db.migrate && uvicorn writeoff.api.app:app --host 0.0.0.0 --port 8000"]
