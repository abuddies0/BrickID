# https://docs.astral.sh/uv/guides/integration/fastapi/#deployment
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
WORKDIR /app

# Cache uv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

# Copy application into container
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8080", "--host", "0.0.0.0"]
