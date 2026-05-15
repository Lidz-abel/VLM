# Experiment Log

## 2026-05-15

- 初始化项目结构。
- 实现单图问答、结构化解析、PDF 逐页解析、FAISS 检索、Markdown 笔记生成和 benchmark 入口。
- 增加 Mock 模式、GPU 空闲显存检查、PDF 断点续跑、页码范围限制，以及 `render_pdf.py` / `parse_pages.py` 拆分脚本。
- 增加 JSONL 运行日志、PDF results.json 转 Markdown、失败页列表脚本、benchmark 人工评分 CSV，以及 Gradio 结果浏览页。
- 增加 FastAPI 后端、内存异步任务管理器、doc_id 工作区、Makefile 和无 GPU pytest 测试。

## 待记录

- 模型加载显存占用：
- 单图平均响应时间：
- PDF 解析速度：
- Benchmark 样本数：
- 图片类型识别准确率：
