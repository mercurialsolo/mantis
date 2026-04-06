"""Quick test: verify Gemma4 inference on Modal A100 GPU."""

import modal
import subprocess
import time

app = modal.App("gemma4-gpu-test")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake")
    .run_commands(
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)",
    )
    .pip_install("huggingface-hub[cli]", "requests", "pillow")
)

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)


@app.function(gpu="A100", image=image, volumes={"/data": vol}, timeout=600)
def test_inference():
    import json
    import base64
    import requests
    from PIL import Image, ImageDraw
    from io import BytesIO
    from huggingface_hub import hf_hub_download

    # Download model if needed
    model_dir = "/data/models"
    model_path = f"{model_dir}/gemma-4-e4b-it-Q4_K_M.gguf"
    if not __import__("os").path.exists(model_path):
        print("Downloading Gemma4 GGUF...")
        __import__("os").makedirs(model_dir, exist_ok=True)
        hf_hub_download("ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-e4b-it-Q4_K_M.gguf", local_dir=model_dir)
        hf_hub_download("ggml-org/gemma-4-E4B-it-GGUF", "mmproj-gemma-4-e4b-it-f16.gguf", local_dir=model_dir)
        vol.commit()
    print(f"Model ready at {model_path}")

    # Start server
    print("Starting llama-server on A100...")
    proc = subprocess.Popen(
        ["/opt/llama.cpp/build/bin/llama-server",
         "-m", model_path, "--host", "0.0.0.0", "--port", "8080",
         "-ngl", "99", "-c", "4096"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Wait for ready
    for _ in range(60):
        try:
            r = requests.get("http://localhost:8080/v1/models", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        proc.terminate()
        return {"error": "Server failed to start"}

    print("Server ready!")

    # Create test image
    img = Image.new("RGB", (1920, 1080), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([800, 500, 1000, 540], fill=(0, 120, 212))
    draw.text((850, 510), "Submit", fill="white")
    buf = BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    tools = [{"type": "function", "function": {
        "name": "click", "description": "Click",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "integer"}, "y": {"type": "integer"}
        }, "required": ["x", "y"]}
    }}]

    payload = {
        "model": "gemma-4",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Click the Submit button. Screen 1920x1080."},
        ]}],
        "tools": tools,
        "max_tokens": 1024,
        "temperature": 0,
    }

    print("Running inference...")
    t0 = time.time()
    resp = requests.post("http://localhost:8080/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0

    data = resp.json()
    msg = data["choices"][0]["message"]

    proc.terminate()

    result = {
        "inference_time": f"{elapsed:.2f}s",
        "tool_calls": msg.get("tool_calls"),
        "thinking": (msg.get("reasoning_content") or "")[:200],
        "tokens": data.get("usage"),
    }
    print(f"\nResult: {json.dumps(result, indent=2)}")
    return result


@app.local_entrypoint()
def main():
    result = test_inference.remote()
    print(f"\n{'='*50}")
    print(f"  GPU Inference Test Result:")
    print(f"  Time: {result.get('inference_time', 'N/A')}")
    print(f"  Tool calls: {result.get('tool_calls', 'none')}")
    print(f"{'='*50}")
