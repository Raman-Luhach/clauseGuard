# ClauseGuard web UI + REST API. Serves the packaged models (TF-IDF and ONNX): no PyTorch.
# Build after `make package` has written deploy/models (see docs/DEPLOY.md).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.11.29
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    CLAUSEGUARD_HOME=/app \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first (cached unless the lock file changes); serve + llm groups only.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-default-groups --group serve --group llm

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-default-groups --group serve --group llm

COPY deploy/models ./artifacts
RUN mkdir -p /app/logs && chown -R app:app /app/logs
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:7860/health').status != 200)"
CMD ["clauseguard", "serve", "--host", "0.0.0.0", "--port", "7860"]
