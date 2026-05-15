#!/bin/bash
set -e

conda create -n multimodal-vlm python=3.10 -y

conda run -n multimodal-vlm python -m pip install --upgrade pip
conda run -n multimodal-vlm python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda run -n multimodal-vlm python -m pip install -r requirements.txt
