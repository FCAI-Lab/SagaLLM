# ── SafeSagaLLM Application Image ─────────────────────────────────────────────
# Includes: Python dependencies, OPA binary, TLC jar (Java)
# API keys are NOT baked in — pass via --env-file .env at runtime

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# ── OPA binary ────────────────────────────────────────────────────────────────
RUN curl -L -o /usr/local/bin/opa \
    https://openpolicyagent.org/downloads/v0.68.0/opa_linux_amd64_static && \
    chmod +x /usr/local/bin/opa

# ── TLC (tla2tools.jar) ───────────────────────────────────────────────────────
RUN mkdir -p /opt/tlc && \
    wget -q -O /opt/tlc/tla2tools.jar \
    https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

ENV TLC_JAR=/opt/tlc/tla2tools.jar

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir anthropic

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/       ./src/
COPY spec/      ./spec/
COPY experiments/ ./experiments/
COPY dataset/   ./dataset/

ENV PYTHONPATH=/app/src

# Default: drop into shell (override in docker-compose or docker run)
CMD ["bash"]
