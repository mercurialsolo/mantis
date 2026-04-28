FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY . .

# Default: stdio for HUD platform. Override at runtime for external use:
#   docker run my-image hud dev env:env --port 8080
CMD ["uv", "run", "python", "-m", "hud", "dev", "env:env", "--stdio"]
