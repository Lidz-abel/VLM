#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_PROXY="${NO_PROXY},localhost,127.0.0.1,0.0.0.0"
export no_proxy="${no_proxy},localhost,127.0.0.1,0.0.0.0"
conda run -n multimodal-vlm uvicorn backend.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"
