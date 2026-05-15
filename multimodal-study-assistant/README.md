# 基于 Qwen2.5-VL 的学习资料多模态问答系统

本项目实现一个面向学习资料场景的多模态问答系统：用户可以上传图片或 PDF，系统调用 Qwen2.5-VL 识别其中的文字、公式、表格、图表、代码截图或考试题，并输出结构化解析；PDF 内容可以逐页解析、建立 FAISS 检索索引，并生成 Markdown 学习笔记。

## 项目动机

传统文本 RAG 系统难以处理课件截图、论文图表、公式、代码截图和考试题图片等多模态学习资料。为解决这一问题，本项目基于视觉语言模型构建了一个面向学习资料场景的多模态问答系统，支持图片/PDF 输入、结构化解析、页面级检索和 Markdown 笔记生成。

## 功能展示

- 单张图片上传与问答
- 考试题、论文图表、代码截图、课件截图和统计图表的结构化解析
- PDF 上传并逐页转图片
- 每页 VLM 解析结果保存为 JSON
- 基于 sentence-transformers + FAISS 的 PDF 页面检索
- Markdown 学习笔记生成
- 小规模 benchmark 评测脚本

## 技术路线

```text
图片 / PDF
→ PDF 按页渲染为 PNG
→ Qwen2.5-VL 页面级理解
→ 结构化 JSON 保存
→ SentenceTransformer embedding
→ FAISS 检索
→ 页面依据 + 问答 / 笔记生成
```

## 环境配置

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

默认模型配置在 `configs/model_config.yaml`：

```yaml
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
generation:
  max_new_tokens: 1024
vision:
  max_pixels: 1003520
runtime:
  batch_size: 1
```

## 快速开始

启动 Gradio：

```bash
bash scripts/run_demo.sh
```

默认只使用 `CUDA_VISIBLE_DEVICES=0` 暴露一张 GPU。需要换卡时：

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/run_demo.sh
```

如果当前没有空闲 GPU，可以先用 Mock 模式测试前端、PDF、RAG 和笔记链路：

```bash
MOCK_VLM=1 bash scripts/run_demo.sh
```

或者：

```bash
python app.py
```

打开页面后可以使用四个功能页：

- 单图问答
- PDF 解析
- PDF 检索
- Markdown 笔记
- 结果浏览

## FastAPI 后端

工程化后端位于 `backend/`，提供图片解析、PDF 异步解析、RAG 构建、RAG 查询、笔记生成和任务状态查询接口。

启动 API：

```bash
make api
```

无 GPU Mock 模式启动：

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

上传 PDF 示例：

```bash
curl -F "file=@data/pdfs/os_chapter2.pdf" \
  http://127.0.0.1:8000/api/documents
```

解析 PDF 示例：

```bash
curl -X POST http://127.0.0.1:8000/api/documents/$DOC_ID/parse \
  -H "Content-Type: application/json" \
  -d '{"start_page":1,"end_page":5,"resume":true,"mock":true}'
```

查询任务：

```bash
curl http://127.0.0.1:8000/api/tasks/$TASK_ID
```

## Doc ID 工作区

API 上传的文档会按 `doc_id` 组织到：

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

这样可以避免不同 PDF 的图片、解析结果、索引和笔记互相混淆。

## 工程命令

项目提供 `Makefile`：

```bash
make setup       # 创建/安装环境
make demo        # 启动 Gradio
make demo-mock   # Mock 模式启动 Gradio
make api         # 启动 FastAPI
make api-mock    # Mock 模式启动 FastAPI
make test        # 运行无 GPU 测试
make check       # Python 语法检查
make clean       # 清理缓存
```

## 单图问答示例

1. 上传一张题目截图、论文图表、课件截图或代码报错截图。
2. 输入问题：`请解释这张图片`。
3. 勾选结构化解析 Prompt。
4. 输出包含：

```text
【图片类型】
【识别内容】
【核心知识点】
【分析过程】
【最终答案】
```

## PDF 问答示例

只渲染 PDF 页面图片，不加载模型、不占 GPU：

```bash
python scripts/render_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

逐页解析 PDF。默认断点续跑，已有 `page_*.json` 会跳过：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

没有空闲 GPU 时，先用 Mock 模式验证链路：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2_mock \
  --start-page 1 \
  --end-page 3 \
  --mock
```

如果已经先渲染了页面图片，可以只解析已有图片：

```bash
python scripts/parse_pages.py \
  --images-dir data/parsed_pages/os_chapter2 \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

构建检索索引：

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2
```

Embedding 默认使用 CPU，避免在 GPU 忙时占卡；如需指定：

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2 \
  --device cpu
```

在 Gradio 的“PDF 检索”页填写索引目录后，可以先检索相关页面，也可以点击“检索并回答”生成带页码依据的回答。例如：

```text
这份课件中进程调度讲了什么？
第几页讲了银行家算法？
帮我总结这份 PDF 中所有关于死锁的内容。
```

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

Gradio 中也提供“结果浏览”页，可以查看指定页码的页面图片、解析内容，并导出整份页面解析 Markdown。

## Benchmark 结果

仓库提供 benchmark 模板：

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/qwen2_5_vl_7b_result.json
```

评测会同时生成一个人工评分 CSV，例如：

```text
eval/results/qwen2_5_vl_7b_result.manual_scores.csv
```

没有空闲 GPU 时也可以先检查评测输出格式：

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/mock_result.json \
  --mock
```

建议扩展到约 100 张图片：

| 类别 | 数量 |
| --- | ---: |
| 操作系统题目截图 | 30 |
| 数学 / 逻辑题目截图 | 20 |
| 论文图表截图 | 20 |
| 代码报错截图 | 20 |
| 课件截图 | 10 |

人工评分标准：

| 分数 | 含义 |
| --- | --- |
| 0 | 完全错误 |
| 1 | 部分相关，但有明显错误 |
| 2 | 基本正确，但不完整 |
| 3 | 完全正确，表述清楚 |

## 显存占用说明

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

系统会把关键运行事件写入：

```text
outputs/run_logs.jsonl
```

日志包含事件类型、Mock 状态、可见 GPU、显存摘要、耗时、输入路径和错误信息。它适合用于调试失败页、整理实验记录和复盘 benchmark。

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

## Token 需求

本项目默认使用本地模型 `Qwen/Qwen2.5-VL-7B-Instruct`，运行推理本身不需要 OpenAI API token，也不需要调用任何付费 AI API。

通常不需要 Hugging Face token；只有在以下情况才需要：

- Hugging Face 下载受限、需要登录或访问私有模型。
- 你把模型换成 gated/private repo。
- 网络环境要求通过 Hugging Face CLI 登录缓存模型。

如果需要 Hugging Face token，可以设置：

```bash
export HF_TOKEN=你的_huggingface_token
```

如果后续把 RAG 最终回答改成 OpenAI、DeepSeek、通义千问 API 等云端模型，才需要对应服务的 API key，例如 `OPENAI_API_KEY` 或厂商自己的 key。当前代码不依赖这些 token。

## 项目限制

- VLM 解析结果仍可能出现 OCR 错误或幻觉，需要 benchmark 和人工评分验证。
- 当前 RAG 基于页面文本解析结果，不直接检索图片 patch。
- PDF 问答质量依赖页面级 VLM 解析质量；若某页 OCR 或图表解释错误，RAG 也会受到影响。
- Mock 模式只验证工程链路，不代表真实模型效果。

## 后续改进方向

- 支持多轮对话和历史上下文管理。
- 支持生成复习题。
- 比较不同 Prompt 效果。
- 支持 Qwen2.5-VL-3B / 7B 切换和量化模型。
- 构建更完整 benchmark 并补充实验结果表格。
- 在课程资料上做 LoRA 微调。

## 简历描述

基于 Qwen2.5-VL 实现了一个面向学习资料场景的多模态问答系统，支持图片和 PDF 输入，能够对考试题、课件截图、论文图表和代码截图进行结构化解析、问答与 Markdown 笔记生成。系统使用 PyMuPDF 将 PDF 转换为页面图像，调用视觉语言模型进行页面级理解，并结合 FAISS 构建检索增强问答模块，实现对整份文档的多模态 RAG。进一步构建了包含操作系统题目、论文图表、代码截图的小规模 benchmark，从 OCR、知识点识别、答案正确性和幻觉率等维度评估模型效果。
