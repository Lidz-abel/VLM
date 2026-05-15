# 基于 Qwen2.5-VL 的学习资料多模态问答系统项目报告

## 1. 项目背景

传统文本 RAG 系统难以处理课件截图、论文图表、公式、代码截图和考试题图片等多模态学习资料。本项目基于 Qwen2.5-VL 构建面向学习资料场景的多模态问答系统，支持图片/PDF 输入、结构化解析、页面级检索和 Markdown 笔记生成。

## 2. 系统功能

- 单张图片问答与结构化解析
- PDF 逐页转图片并调用 VLM 解析
- 保存页面级 JSON 解析结果
- 使用 sentence-transformers 和 FAISS 构建 PDF 页面检索
- 根据解析内容生成 Markdown 学习笔记
- 提供小规模 benchmark 评测入口
- 提供 FastAPI 后端接口
- 支持异步任务、任务状态查询和 doc_id 工作区文件组织
- 支持 Mock 模式和无 GPU 测试

## 3. 技术路线

1. 使用 Gradio 构建演示界面。
2. 使用 FastAPI 暴露图片解析、PDF 解析、RAG 和笔记生成接口。
3. 使用 ThreadPoolExecutor 管理 PDF 解析等长任务。
4. 使用 doc_id 工作区隔离每份文档的源文件、页面图片、解析结果、索引和笔记。
5. 使用 PyMuPDF 将 PDF 渲染为页面 PNG。
6. 使用 Qwen2.5-VL-7B-Instruct 完成图片理解与页面解析。
7. 使用 BAAI/bge-small-zh-v1.5 生成页面文本向量。
8. 使用 FAISS IndexFlatIP 进行归一化向量检索。
9. 将检索到的页面内容作为依据生成面向学习场景的回答或笔记。

## 4. 显存控制

- 单卡 24GB 下默认 batch_size=1。
- 默认 max_pixels=1280*28*28。
- PDF 必须逐页处理，避免一次输入多页大图。
- Gradio queue 默认并发限制为 1。

## 5. 实验设计

benchmark 建议包含约 100 张图片，覆盖操作系统题目、数学/逻辑题、论文图表、代码报错和课件截图。当前仓库提供 `eval/benchmark.json` 模板和 `eval/evaluate.py` 评测脚本。

## 6. 后续改进

- 扩充 benchmark 并补充人工评分。
- 比较 Qwen2.5-VL-3B、7B、量化模型的速度和效果。
- 加入多轮对话状态管理。
- 在特定课程资料上做 LoRA 微调。
- 将内存任务队列升级为 Redis/Celery，以支持多进程和持久化任务恢复。
- 增加 Docker GPU 部署和 CI。
