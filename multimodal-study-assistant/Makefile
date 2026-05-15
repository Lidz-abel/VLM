SHELL := /bin/bash
ENV ?= multimodal-vlm
GPU ?= 0
API_PORT ?= 8000

.PHONY: setup demo demo-mock api api-mock test check clean

setup:
	bash scripts/setup_env.sh

demo:
	CUDA_VISIBLE_DEVICES=$(GPU) bash scripts/run_demo.sh

demo-mock:
	MOCK_VLM=1 bash scripts/run_demo.sh

api:
	CUDA_VISIBLE_DEVICES=$(GPU) API_PORT=$(API_PORT) bash scripts/run_api.sh

api-mock:
	MOCK_VLM=1 API_PORT=$(API_PORT) bash scripts/run_api.sh

test:
	conda run -n $(ENV) pytest -q

check:
	conda run -n $(ENV) python -m py_compile app.py src/*.py scripts/*.py eval/*.py backend/*.py

clean:
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__ eval/__pycache__ backend/__pycache__ tests/__pycache__ .pytest_cache
