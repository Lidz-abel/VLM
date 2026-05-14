#!/bin/bash
set -e

conda create -n multimodal-vlm python=3.10 -y
conda activate multimodal-vlm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
