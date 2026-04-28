# Mantis Holo3 server — cloud-portable container
#
# Runs the FastAPI workload server (mantis_agent.baseten_server:app) backed by
# Holo3-35B-A3B (llama.cpp GGUF) on a single CUDA GPU. Drop-in for AWS EKS,
# GKE, or any K8s/ECS environment with NVIDIA device plugin.
#
# Build:    docker build -f docker/server.Dockerfile -t mantis-holo3-server .
# Run:      docker run --gpus all -p 8000:8000 \
#             -e MANTIS_API_TOKEN=... -e ANTHROPIC_API_KEY=... \
#             -e PROXY_URL=... -e PROXY_USER=... -e PROXY_PASS=... \
#             -v /srv/mantis-data:/workspace/mantis-data \
#             -v /srv/models/holo3:/models/holo3 \
#             mantis-holo3-server
# Smoke:    curl -X POST http://<host>:8000/predict \
#             -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
#             -H "Content-Type: application/json" \
#             -d '{"detached":true,"micro":"plans/boattrader/extract_url_filtered_3listings.json","state_key":"smoke","resume_state":false,"max_cost":2,"max_time_minutes":20}'
#
# Holo3 GGUF weights (~34 GB) are NOT baked into the image; mount them from a
# pre-warmed volume at /models/holo3 (see deploy/aws/ and deploy/gke/ for the
# init-job pattern that downloads from HF on first run).

# Same base as the Baseten Truss to maximize layer reuse.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MANTIS_DATA_DIR=/workspace/mantis-data \
    MANTIS_LLAMA_PORT=18080 \
    MANTIS_MODEL=holo3 \
    PYTHONPATH=/app/src

# System deps: build chain for llama.cpp + Chrome + Xvfb + xdotool
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake curl wget gnupg ca-certificates \
        xvfb xdotool scrot \
        fonts-liberation fonts-noto-color-emoji \
        libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA — pinned to b8948 (commit 42401c72) for reproducibility
RUN git clone --depth 1 --branch b8948 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && ldconfig \
    && cd /opt/llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build build --target llama-server --config Release -j"$(nproc)"

# Python deps (server extras + llama-cpp client deps)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
        fastapi uvicorn pydantic requests pillow mss \
        openai anthropic huggingface-hub

# App source
WORKDIR /app
COPY src/  /app/src/
COPY plans/ /app/plans/
COPY tasks/ /app/tasks/

EXPOSE 8000

# Health checks (fast — does not warm the model)
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Holo3 weights expected at /models/holo3/Holo3-35B-A3B.Q8_0.gguf
# + /models/holo3/Holo3-35B-A3B.mmproj-f16.gguf (mount as a volume).
ENV MANTIS_HOLO3_MODEL_DIR=/models/holo3

CMD ["uvicorn", "mantis_agent.baseten_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300"]
