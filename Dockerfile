# --- Build stage: install dependencies with uv ---
FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files and install
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

# --- Runtime stage: slim image with only what we need ---
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) dbmate_arch='linux-amd64' ;; \
      arm64) dbmate_arch='linux-arm64' ;; \
      *) echo "unsupported architecture: $arch"; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/bin/dbmate "https://github.com/amacneil/dbmate/releases/latest/download/dbmate-${dbmate_arch}"; \
    chmod +x /usr/local/bin/dbmate

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv .venv/

# Copy .env so load_dotenv() finds it at /app/.env
COPY .env .env
COPY db/ db/

# Create data directory (Fly.io volume mounts over /opt/montferrand,
# but this ensures it exists for local docker run without a volume)
RUN mkdir -p /opt/montferrand

# Ensure the venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["montferrand", "serve", "--host", "0.0.0.0", "--port", "8080"]
