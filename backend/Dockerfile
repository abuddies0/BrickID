# https://docs.astral.sh/uv/guides/integration/fastapi/#deployment
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy application into container
COPY . /app

# Install dependencies
WORKDIR /app
RUN uv sync --frozen --no-cache

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8080", "--host", "0.0.0.0"]
