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

或者：

```bash
python app.py
```

打开页面后可以使用四个功能页：

- 单图问答
- PDF 解析
- PDF 检索
- Markdown 笔记

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

逐页解析 PDF：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2
```

构建检索索引：

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2
```

在 Gradio 的“PDF 检索”页填写索引目录后，可以先检索相关页面，也可以点击“检索并回答”生成带页码依据的回答。例如：

```text
这份课件中进程调度讲了什么？
第几页讲了银行家算法？
帮我总结这份 PDF 中所有关于死锁的内容。
```

## Benchmark 结果

仓库提供 benchmark 模板：

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/qwen2_5_vl_7b_result.json
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

监控命令：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
```

如果 OOM，依次降低 `max_pixels`、降低 `max_new_tokens`、切换 Qwen2.5-VL-3B、使用量化模型或关闭其他 GPU 进程。

## 项目限制

- VLM 解析结果仍可能出现 OCR 错误或幻觉，需要 benchmark 和人工评分验证。
- 当前 RAG 基于页面文本解析结果，不直接检索图片 patch。
- PDF 问答质量依赖页面级 VLM 解析质量；若某页 OCR 或图表解释错误，RAG 也会受到影响。

## 后续改进方向

- 支持多轮对话和历史上下文管理。
- 支持导出解析结果为 `.md`。
- 支持生成复习题。
- 比较不同 Prompt 效果。
- 支持 Qwen2.5-VL-3B / 7B 切换和量化模型。
- 构建更完整 benchmark 并补充实验结果表格。
- 在课程资料上做 LoRA 微调。

## 简历描述

基于 Qwen2.5-VL 实现了一个面向学习资料场景的多模态问答系统，支持图片和 PDF 输入，能够对考试题、课件截图、论文图表和代码截图进行结构化解析、问答与 Markdown 笔记生成。系统使用 PyMuPDF 将 PDF 转换为页面图像，调用视觉语言模型进行页面级理解，并结合 FAISS 构建检索增强问答模块，实现对整份文档的多模态 RAG。进一步构建了包含操作系统题目、论文图表、代码截图的小规模 benchmark，从 OCR、知识点识别、答案正确性和幻觉率等维度评估模型效果。
