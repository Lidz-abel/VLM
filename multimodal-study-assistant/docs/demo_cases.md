# Demo Cases

## 单图问答

1. 上传考试题截图。
2. 问题填写：`请解答这道题，并说明涉及的知识点。`
3. 检查输出是否包含图片类型、识别内容、核心知识点、分析过程和最终答案。

## FastAPI

Mock 模式启动后端：

```bash
make api-mock
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

上传图片并使用 Mock 解析：

```bash
curl -X POST http://127.0.0.1:8000/api/image/analyze \
  -F "image=@data/images/example.png" \
  -F "question=请解释这张图片" \
  -F "structured=true" \
  -F "mock=true"
```

上传 PDF：

```bash
curl -F "file=@data/pdfs/os_chapter2.pdf" \
  http://127.0.0.1:8000/api/documents
```

拿到 `doc_id` 后异步解析：

```bash
curl -X POST http://127.0.0.1:8000/api/documents/$DOC_ID/parse \
  -H "Content-Type: application/json" \
  -d '{"start_page":1,"end_page":5,"resume":true,"mock":true}'
```

查询任务：

```bash
curl http://127.0.0.1:8000/api/tasks/$TASK_ID
```

## PDF 解析

GPU 忙时先只渲染页面：

```bash
python scripts/render_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

Mock 模式验证解析链路：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2_mock \
  --start-page 1 \
  --end-page 3 \
  --mock
```

真实模型逐页解析，默认断点续跑：

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

解析已有页面图片：

```bash
python scripts/parse_pages.py \
  --images-dir data/parsed_pages/os_chapter2 \
  --output data/parsed_pages/os_chapter2 \
  --start-page 1 \
  --end-page 5
```

## PDF RAG

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2 \
  --device cpu
```

## 结果浏览与导出

```bash
python scripts/export_results_md.py \
  --results data/parsed_pages/os_chapter2/results.json \
  --output outputs/os_chapter2_pages.md
```

```bash
python scripts/list_failed_pages.py \
  --results data/parsed_pages/os_chapter2/results.json
```

## Benchmark

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/qwen2_5_vl_7b_result.json
```

评测结束后会生成：

```text
eval/results/qwen2_5_vl_7b_result.manual_scores.csv
```

## 工程测试

```bash
make test
```
