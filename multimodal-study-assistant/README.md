# 基于 Qwen2.5-VL 的学习资料多模态问答系统

面向考试题截图、课件截图、论文图表、代码报错截图和 PDF 学习资料的多模态理解与问答系统。项目基于 `Qwen/Qwen2.5-VL-7B-Instruct` 实现图片/PDF 输入、结构化解析、页面级 RAG 检索问答、Markdown 笔记生成、benchmark 评测，并进一步封装为带 FastAPI 后端、异步任务和 doc_id 工作区的工程化服务原型。

## 项目状态

当前阶段已经完成：

- 单图 VLM 问答与结构化解析
- PDF 按页渲染为图片
- PDF 页面级 VLM 解析和 JSON 持久化
- 断点续跑、失败页记录、页码范围限制
- FAISS + sentence-transformers 页面级检索
- PDF RAG 检索问答
- Markdown 学习笔记生成
- Benchmark 评测和人工评分 CSV
- Mock 模式，无 GPU 时可测试完整工程链路
- 运行日志 `outputs/run_logs.jsonl`
- Gradio Demo
- FastAPI 后端
- 异步任务管理器
- doc_id 工作区文件组织
- 无 GPU pytest 测试

已完成一次真实 GPU 实验：

| 项目 | 结果 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| GPU | RTX 4090 单卡 |
| 样本 | 3 张操作系统考试题截图 |
| 图片类型识别准确率 | 100.00% |
| 平均响应时间 | 26.40s |
| 峰值显存 | 约 18.8GB |
| 结果文件 | `eval/results/qwen2_5_vl_7b_3_images.json` |

## 项目动机

传统文本 RAG 系统难以处理课件截图、论文图表、公式、代码截图和考试题图片等多模态学习资料。为解决这一问题，本项目基于视觉语言模型构建面向学习资料场景的多模态问答系统，支持图片/PDF 输入、结构化解析、页面级检索和 Markdown 笔记生成。

## 系统架构

```text
图片 / PDF
  │
  ├── 单图输入
  │     └── Qwen2.5-VL 结构化解析
  │
  └── PDF 输入
        ├── PyMuPDF 按页渲染为 PNG
        ├── Qwen2.5-VL 页面级解析
        ├── results.json / page_*.json
        ├── SentenceTransformer embedding
        ├── FAISS 页面检索
        └── RAG 问答 / Markdown 笔记
```

工程层：

```text
Gradio Demo
FastAPI Backend
Async Task Manager
doc_id Workspace
JSONL Run Logs
pytest Tests
```

## 目录结构

```text
multimodal-study-assistant/
├── app.py                       # Gradio Demo
├── backend/                     # FastAPI 后端
├── configs/model_config.yaml    # 模型、生成、显存和 RAG 配置
├── data/
│   ├── images/                  # 单图测试图片
│   ├── pdfs/                    # PDF 输入
│   ├── parsed_pages/            # PDF 页面图片和解析结果
│   └── vector_db/               # FAISS 索引
├── docs/                        # 项目报告、实验日志、Demo 用例
├── eval/                        # Benchmark 与评测脚本
├── outputs/                     # 运行日志、导出笔记和结果
├── scripts/                     # CLI 脚本
├── src/                         # 核心模块
├── tests/                       # 无 GPU 单元测试
├── workspace/                   # API doc_id 工作区
├── Makefile
└── requirements.txt
```

## 环境配置

推荐环境：

```text
Python 3.10
CUDA 12.1
GPU: 单卡 24GB，例如 RTX 4090
```

安装：

```bash
conda create -n multimodal-vlm python=3.10 -y
conda activate multimodal-vlm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

也可以直接执行：

```bash
bash scripts/setup_env.sh
```

默认配置：[configs/model_config.yaml](configs/model_config.yaml)

```yaml
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
generation:
  max_new_tokens: 1024
vision:
  max_pixels: 1003520
runtime:
  batch_size: 1
  min_free_gb: 18.0
  require_single_visible_gpu: true
```

## 快速开始

### Gradio Demo

真实模型模式：

```bash
GPU=0 make demo
```

无 GPU Mock 模式：

```bash
make demo-mock
```

默认地址：

```text
http://127.0.0.1:7860
```

Gradio 功能页：

- 单图问答
- PDF 解析
- PDF 检索
- Markdown 笔记
- 结果浏览

### FastAPI 服务

真实模型模式：

```bash
GPU=0 make api
```

无 GPU Mock 模式：

```bash
make api-mock
```

默认地址：

```text
http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

主要接口：

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| `GET` | `/health` | 服务和 GPU 状态 |
| `POST` | `/api/image/analyze` | 上传图片并解析 |
| `POST` | `/api/documents` | 上传 PDF，返回 `doc_id` |
| `GET` | `/api/documents/{doc_id}` | 查看文档元数据 |
| `POST` | `/api/documents/{doc_id}/parse` | 异步解析 PDF |
| `GET` | `/api/tasks/{task_id}` | 查询任务状态 |
| `GET` | `/api/documents/{doc_id}/results` | 获取解析结果 |
| `POST` | `/api/documents/{doc_id}/rag/build` | 异步构建 RAG 索引 |
| `POST` | `/api/documents/{doc_id}/rag/query` | 对文档提问 |
| `POST` | `/api/documents/{doc_id}/notes` | 生成 Markdown 笔记 |

## 使用示例

### 单图问答

启动 Gradio 后上传图片，问题示例：

```text
请解析这道操作系统题，给出题目类型、关键数据、解题思路和最终答案。
```

结构化输出包含：

```text
【图片类型】
【识别内容】
【核心知识点】
【分析过程】
【最终答案】
```

### PDF 渲染

只渲染 PDF 页面图片，不加载模型：

```bash
python scripts/render_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

### PDF 逐页解析

真实模型解析：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n multimodal-vlm python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

Mock 模式解析：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2_mock \
  --start-page 1 \
  --end-page 3 \
  --mock
```

解析已有页面图片：

```bash
python scripts/parse_pages.py \
  --images-dir data/parsed_pages/os_chapter2 \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

### 构建 RAG 索引

Embedding 默认使用 CPU，避免占用 GPU：

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2 \
  --device cpu
```

### 导出和失败页检查

导出页面解析 Markdown：

```bash
python scripts/export_results_md.py \
  --results data/parsed_pages/os_chapter2/results.json \
  --output outputs/os_chapter2_pages.md
```

列出失败页：

```bash
python scripts/list_failed_pages.py \
  --results data/parsed_pages/os_chapter2/results.json
```

## doc_id 工作区

FastAPI 上传的文档会按 `doc_id` 组织：

```text
workspace/documents/{doc_id}/
├── source.pdf
├── pages/
│   └── page_1.png
├── parsed/
│   ├── page_1.json
│   └── results.json
├── vector_db/
└── notes.md
```

这使得每份文档的源文件、页面图片、解析结果、索引和笔记互相隔离，方便工程化管理和后续扩展。

## Benchmark

运行真实模型评测：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n multimodal-vlm python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/qwen2_5_vl_7b_result.json
```

无 GPU 检查评测流程：

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/mock_result.json \
  --mock
```

评测会生成：

```text
eval/results/*.json
eval/results/*.manual_scores.csv
```

人工评分字段：

```text
ocr_score_0_3
concept_score_0_3
answer_score_0_3
hallucination_0_or_1
notes
```

评分标准：

| 分数 | 含义 |
| --- | --- |
| 0 | 完全错误 |
| 1 | 部分相关，但有明显错误 |
| 2 | 基本正确，但不完整 |
| 3 | 完全正确，表述清楚 |

## 真实实验结果

已完成 3 张操作系统考试题截图的真实模型验证：

| 样本 ID | 期望类型 | 预测类型 | 耗时 |
| --- | --- | --- | ---: |
| `os_scheduling_001` | `exam_question` | `exam_question` | 40.71s |
| `os_banker_001` | `exam_question` | `exam_question` | 18.95s |
| `os_process_thread_001` | `exam_question` | `exam_question` | 19.53s |

汇总：

```text
模型：Qwen/Qwen2.5-VL-7B-Instruct
样本数：3
图片类型识别准确率：100.00%
平均响应时间：26.40s
峰值显存：约 18.8GB
```

## 显存控制

24GB 单卡建议：

- `batch_size=1`
- `max_new_tokens=1024`
- `max_pixels=1280*28*28`
- PDF 逐页处理
- 不要同时开启多个 Gradio 请求
- 启动前设置 `CUDA_VISIBLE_DEVICES=单张卡编号`
- 默认要求可见 GPU 空闲显存不少于 18GB

监控命令：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
```

如果 OOM，依次降低 `max_pixels`、降低 `max_new_tokens`、切换 Qwen2.5-VL-3B、使用量化模型或关闭其他 GPU 进程。

## 运行日志

关键运行事件写入：

```text
outputs/run_logs.jsonl
```

日志包含事件类型、Mock 状态、可见 GPU、显存摘要、耗时、输入路径和错误信息，适合调试失败页、整理实验记录和复盘 benchmark。

## 测试

当前测试均不需要 GPU：

```bash
make test
```

覆盖内容：

- 轻量异步任务管理器
- `doc_id` 工作区布局
- PDF 结果 Markdown 导出和失败页提取
- FastAPI `/health`
- FastAPI Mock 图片解析

语法检查：

```bash
make check
```

## Token 需求

本项目默认使用本地模型 `Qwen/Qwen2.5-VL-7B-Instruct`，运行推理本身不需要 OpenAI API token，也不需要调用付费 AI API。

通常也不需要 Hugging Face token。只有以下情况可能需要：

- Hugging Face 下载限速严重，需要登录提升下载稳定性。
- 访问私有模型或 gated 模型。
- 服务器环境要求通过 Hugging Face CLI 登录缓存模型。

设置方式：

```bash
export HF_TOKEN=你的_huggingface_token
```

如果后续将 RAG 最终回答切换为 OpenAI、DeepSeek、通义千问等云端模型，才需要对应厂商的 API key。当前代码不依赖这些 token。

## 项目限制

- 当前 benchmark 样本量较小，需要继续扩充到 50-100 张以上。
- 页面级 RAG 依赖 VLM 的页面解析质量；若 OCR 或图表解释错误，后续检索问答会受影响。
- 当前 RAG 基于页面解析文本，不直接检索图片 patch。
- Mock 模式只用于验证工程链路，不代表真实模型效果。
- 当前异步任务管理器为内存实现，生产环境建议升级为 Redis/Celery 或其他持久化队列。

## 后续方向

- 扩充图片 benchmark 和 PDF QA 数据集。
- 做 Prompt 对比实验。
- 做 page-level / section-level / summary-only 等 RAG 粒度对比。
- 支持 Qwen2.5-VL-3B / 7B / 量化模型切换。
- 增加 Docker GPU 部署。
- 增加 GitHub Actions CI。
- 数据量达到 500-1000 条后，再考虑 LoRA 微调。

## 简历描述

基于 Qwen2.5-VL-7B-Instruct 构建了一个面向学习资料场景的多模态问答系统，支持考试题、课件截图、论文图表、代码截图和 PDF 文档的结构化解析、页面级检索问答与 Markdown 笔记生成。系统使用 PyMuPDF 将 PDF 转换为页面图像，调用视觉语言模型进行页面级理解，并结合 sentence-transformers 与 FAISS 构建检索增强问答模块。进一步实现了 benchmark 评测、人工评分 CSV、运行日志、Mock 推理模式、断点续跑和失败页恢复。工程层面基于 FastAPI 封装图片解析、PDF 异步解析、RAG 构建与查询接口，并设计 doc_id 工作区、轻量任务管理器、Makefile 和无 GPU 单元测试，提升系统的可维护性、可观测性和部署复现能力。
