#!/bin/bash
set -euo pipefail

PROPAINTER_DIR="${PROPAINTER_DIR:-/workspace/ProPainter}"
PROPAINTER_REPO_URL="${PROPAINTER_REPO_URL:-https://github.com/sczhou/ProPainter.git}"
PROPAINTER_REF="${PROPAINTER_REF:-main}"

echo "[propainter] Preparing runtime in ${PROPAINTER_DIR}"

if [ -d "${PROPAINTER_DIR}/.git" ]; then
  git -C "${PROPAINTER_DIR}" fetch origin "${PROPAINTER_REF}" --depth 1
  git -C "${PROPAINTER_DIR}" checkout "${PROPAINTER_REF}"
  git -C "${PROPAINTER_DIR}" pull --ff-only origin "${PROPAINTER_REF}"
else
  rm -rf "${PROPAINTER_DIR}"
  git clone --depth 1 --branch "${PROPAINTER_REF}" "${PROPAINTER_REPO_URL}" "${PROPAINTER_DIR}"
fi

python3 -m pip install -q \
  av \
  addict \
  einops \
  future \
  scipy \
  opencv-python-headless \
  matplotlib \
  scikit-image \
  imageio-ffmpeg \
  pyyaml \
  requests \
  timm \
  yapf

echo "[propainter] Runtime ready"
