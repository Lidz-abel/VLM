# Demo Cases

## 单图问答

1. 上传考试题截图。
2. 问题填写：`请解答这道题，并说明涉及的知识点。`
3. 检查输出是否包含图片类型、识别内容、核心知识点、分析过程和最终答案。

## PDF 解析

```bash
python scripts/parse_pdf.py \
  --pdf data/pdfs/os_chapter2.pdf \
  --output data/parsed_pages/os_chapter2
```

## PDF RAG

```bash
python scripts/build_rag_index.py \
  --parsed data/parsed_pages/os_chapter2/results.json \
  --output data/vector_db/os_chapter2
```

## Benchmark

```bash
python eval/evaluate.py \
  --benchmark eval/benchmark.json \
  --output eval/results/qwen2_5_vl_7b_result.json
```
