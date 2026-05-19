# Experiment Log

## 2026-05-15

- 初始化项目结构。
- 实现单图问答、结构化解析、PDF 逐页解析、FAISS 检索、Markdown 笔记生成和 benchmark 入口。
- 增加 Mock 模式、GPU 空闲显存检查、PDF 断点续跑、页码范围限制，以及 `render_pdf.py` / `parse_pages.py` 拆分脚本。
- 增加 JSONL 运行日志、PDF results.json 转 Markdown、失败页列表脚本、benchmark 人工评分 CSV，以及 Gradio 结果浏览页。
- 增加 FastAPI 后端、内存异步任务管理器、doc_id 工作区、Makefile 和无 GPU pytest 测试。

## 待记录

- 模型加载显存占用：Qwen2.5-VL-7B-Instruct 在单张 RTX 4090 上测试时，推理期间 `allocated=15.45GB`，`reserved=17.86GB`，`nvidia-smi` 峰值约 `18.8GB`。
- 单图平均响应时间：3 张操作系统考试题截图平均 `26.40s`。
- PDF 解析速度：
- Benchmark 样本数：
- Benchmark 样本数：3
- 图片类型识别准确率：100.00%（3/3，均识别为 `exam_question`）

## 2026-05-19

- 下载并缓存 `Qwen/Qwen2.5-VL-7B-Instruct`，缓存目录约 16GB。
- 使用 GPU 0 完成 3 张真实图片 benchmark：
  - `os_scheduling_001`：40.71s
  - `os_banker_001`：18.95s
  - `os_process_thread_001`：19.53s
- 输出文件：
  - `eval/results/qwen2_5_vl_7b_3_images.json`
  - `eval/results/qwen2_5_vl_7b_3_images.manual_scores.csv`
