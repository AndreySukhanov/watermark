#!/bin/bash
# Deploy Watermark Remover to RunPod GPU Pod
# Usage: curl -sSL <url> | bash   OR   bash deploy_runpod.sh
set -e

echo "=== Watermark Remover — RunPod Setup ==="

# 1. System deps
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg curl git > /dev/null 2>&1
echo "  ✓ ffmpeg, curl, git installed"

# 2. Clone or copy project
echo "[2/5] Setting up project..."
cd /workspace
if [ -d "watermark" ]; then
    echo "  → Project dir exists, pulling updates..."
    cd watermark && git pull 2>/dev/null || true
else
    echo "  → Enter your git repo URL (or press Enter to skip and upload manually):"
    read -r REPO_URL
    if [ -n "$REPO_URL" ]; then
        git clone "$REPO_URL" watermark
        cd watermark
    else
        mkdir -p watermark
        cd watermark
        echo "  → Upload project files to /workspace/watermark/ then re-run this script"
        exit 0
    fi
fi

# 3. Python deps
echo "[3/5] Installing Python dependencies..."
pip install -q -r requirements_web.txt 2>&1 | tail -5
echo "  ✓ Python packages installed"

# 4. Prepare ProPainter runtime
echo "[4/5] Preparing ProPainter runtime..."
bash scripts/setup_propainter_runtime.sh
echo "  ✓ ProPainter runtime ready"

# 5. Check GPU
echo "[5/5] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem // 1024**3} GB)')
else:
    print('  ⚠ No GPU detected — will run on CPU')
"

# 5. Start server
echo ""
echo "=== Starting server on port 8000 ==="
echo "Access via RunPod proxy URL or http://<pod-ip>:8000"
echo "Health check: curl http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "==================================="

cd /workspace/watermark
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
