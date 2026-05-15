from __future__ import annotations

import os
from pathlib import Path

_LOCAL_NO_PROXY = "localhost,127.0.0.1,0.0.0.0"
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _current = os.environ.get(_proxy_key, "")
    os.environ[_proxy_key] = ",".join(part for part in [_current, _LOCAL_NO_PROXY] if part)

import gradio as gr

from src.note_generator import generate_note_from_records
from src.pdf_parser import parse_pdf
from src.prompts import build_rag_prompt
from src.rag import PdfRagIndex
from src.result_exporter import export_results_markdown, results_to_markdown
from src.run_logger import Timer, log_event
from src.utils import PROJECT_ROOT, get_gpu_memory_summary, load_yaml, read_json, safe_stem
from src.vlm_infer import get_default_inferencer


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CONFIG = load_yaml(CONFIG_PATH)


def _set_mock_mode(enabled: bool) -> None:
    if enabled:
        os.environ["MOCK_VLM"] = "1"
    else:
        os.environ.pop("MOCK_VLM", None)


def answer_single_image(image_path: str, question: str, structured: bool, model_name: str, mock_mode: bool) -> str:
    if not image_path:
        return "请先上传图片。"
    _set_mock_mode(mock_mode)
    timer = Timer()
    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    try:
        answer = inferencer.answer_image(image_path=image_path, question=question, structured=structured)
        log_event(
            "single_image_answered",
            {
                "image_path": image_path,
                "question": question,
                "structured": structured,
                "model": inferencer.model_name,
                "success": True,
                "elapsed_sec": round(timer.elapsed, 3),
            },
        )
    except Exception as exc:
        log_event(
            "single_image_answered",
            {
                "image_path": image_path,
                "question": question,
                "structured": structured,
                "model": model_name,
                "success": False,
                "error": str(exc),
                "elapsed_sec": round(timer.elapsed, 3),
            },
        )
        raise
    return f"{answer}\n\n---\n{get_gpu_memory_summary()}"


def parse_pdf_ui(
    pdf_path: str,
    model_name: str,
    start_page: int | None,
    end_page: int | None,
    resume: bool,
    overwrite_images: bool,
    mock_mode: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    if not pdf_path:
        return "请先上传 PDF。", ""

    _set_mock_mode(mock_mode)
    timer = Timer()
    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    output_dir = PROJECT_ROOT / "data" / "parsed_pages" / safe_stem(pdf_path)

    def update(done: int, total: int) -> None:
        progress(done / total, desc=f"解析 PDF：{done}/{total}")

    records = parse_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        inferencer=inferencer,
        start_page=int(start_page) if start_page else None,
        end_page=int(end_page) if end_page else None,
        resume=resume,
        overwrite_images=overwrite_images,
        progress_callback=update,
    )
    result_file = output_dir / "results.json"
    error_count = sum(1 for item in records if item.get("error"))
    log_event(
        "pdf_parsed",
        {
            "pdf_path": pdf_path,
            "output_dir": str(output_dir),
            "results_file": str(result_file),
            "pages": len(records),
            "errors": error_count,
            "start_page": start_page,
            "end_page": end_page,
            "resume": resume,
            "overwrite_images": overwrite_images,
            "model": inferencer.model_name,
            "elapsed_sec": round(timer.elapsed, 3),
        },
    )
    return f"完成解析/汇总 {len(records)} 页，失败 {error_count} 页。\n结果文件：{result_file}", str(result_file)


def build_rag_ui(results_file: str, embedding_model: str) -> tuple[str, str]:
    if not results_file:
        return "请先解析 PDF，或填写 results.json 路径。", ""
    index_dir = PROJECT_ROOT / "data" / "vector_db" / safe_stem(results_file)
    from src.rag import build_index_from_results

    rag_cfg = CONFIG.get("rag", {})
    timer = Timer()
    build_index_from_results(results_file, index_dir, embedding_model=embedding_model, device=rag_cfg.get("embedding_device", "cpu"))
    log_event(
        "rag_index_built",
        {
            "results_file": results_file,
            "index_dir": str(index_dir),
            "embedding_model": embedding_model,
            "device": rag_cfg.get("embedding_device", "cpu"),
            "elapsed_sec": round(timer.elapsed, 3),
        },
    )
    return f"索引已保存：{index_dir}", str(index_dir)


def search_pdf_ui(question: str, index_dir: str, top_k: int) -> str:
    if not question or not index_dir:
        return "请填写问题和索引目录。"
    index = PdfRagIndex.load(index_dir)
    results = index.search(question, top_k=int(top_k))
    if not results:
        return "没有检索到相关页面。"
    blocks = []
    for item in results:
        preview = item["content"][:600].replace("\n", "\n> ")
        blocks.append(f"### 第 {item['page']} 页  score={item['score']:.4f}\n> {preview}")
    log_event("pdf_searched", {"question": question, "index_dir": index_dir, "top_k": int(top_k), "pages": [item["page"] for item in results]})
    return "\n\n".join(blocks)


def answer_pdf_ui(question: str, index_dir: str, top_k: int, model_name: str, mock_mode: bool) -> str:
    if not question or not index_dir:
        return "请填写问题和索引目录。"
    _set_mock_mode(mock_mode)
    index = PdfRagIndex.load(index_dir)
    contexts = index.search(question, top_k=int(top_k))
    if not contexts:
        return "没有检索到相关页面，无法回答。"
    prompt = build_rag_prompt(question, contexts)
    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    timer = Timer()
    answer = inferencer.answer_text(prompt)
    sources = "，".join(f"第 {item['page']} 页" for item in contexts)
    log_event(
        "pdf_question_answered",
        {
            "question": question,
            "index_dir": index_dir,
            "top_k": int(top_k),
            "pages": [item["page"] for item in contexts],
            "model": inferencer.model_name,
            "elapsed_sec": round(timer.elapsed, 3),
        },
    )
    return f"{answer}\n\n---\n检索依据：{sources}\n\n{get_gpu_memory_summary()}"


def generate_note_ui(results_file: str) -> tuple[str, str]:
    if not results_file:
        return "请先提供 PDF 解析 results.json。", ""
    records = read_json(results_file)
    output_path = PROJECT_ROOT / "outputs" / f"{safe_stem(results_file)}_notes.md"
    timer = Timer()
    note = generate_note_from_records(records, output_path=output_path)
    log_event(
        "note_generated",
        {"results_file": results_file, "output_path": str(output_path), "pages": len(records), "elapsed_sec": round(timer.elapsed, 3)},
    )
    return note, str(output_path)


def browse_results_ui(results_file: str, page_number: int | None) -> tuple[str | None, str, str]:
    if not results_file:
        return None, "", "请填写 results.json 路径。"
    records = read_json(results_file)
    if not records:
        return None, "", "results.json 中没有页面记录。"

    page = int(page_number) if page_number else int(records[0].get("page", 1))
    by_page = {int(item.get("page", 0) or 0): item for item in records}
    record = by_page.get(page)
    if record is None:
        available = ", ".join(str(item.get("page")) for item in records)
        return None, "", f"没有第 {page} 页。可用页码：{available}"

    image_path = record.get("image_path") or None
    if image_path and not Path(image_path).exists():
        image_path = None
    markdown = results_to_markdown([record], title=f"第 {page} 页解析结果")
    status = f"已加载第 {page} 页，共 {len(records)} 页。"
    if record.get("error"):
        status += " 该页包含错误。"
    return image_path, markdown, status


def export_results_ui(results_file: str) -> tuple[str, str]:
    if not results_file:
        return "请填写 results.json 路径。", ""
    output_path = PROJECT_ROOT / "outputs" / f"{safe_stem(results_file)}_pages.md"
    export_results_markdown(results_file, output_path)
    log_event("results_markdown_exported", {"results_file": results_file, "output_path": str(output_path)})
    return f"已导出：{output_path}", str(output_path)


def build_app() -> gr.Blocks:
    default_model = CONFIG.get("model", {}).get("name", "Qwen/Qwen2.5-VL-7B-Instruct")
    default_embedding = CONFIG.get("rag", {}).get("embedding_model", "BAAI/bge-small-zh-v1.5")

    with gr.Blocks(title="Qwen2.5-VL 学习资料多模态问答系统") as demo:
        gr.Markdown("# 基于 Qwen2.5-VL 的学习资料多模态问答系统")

        with gr.Tab("单图问答"):
            with gr.Row():
                image = gr.Image(type="filepath", label="上传图片")
                with gr.Column():
                    model_name = gr.Textbox(value=default_model, label="模型名称")
                    question = gr.Textbox(value="请解释这张图片", label="用户问题")
                    structured = gr.Checkbox(value=True, label="使用结构化解析 Prompt")
                    image_mock = gr.Checkbox(value=False, label="Mock 模式（不加载 GPU 模型）")
                    submit = gr.Button("分析图片", variant="primary")
            image_output = gr.Markdown(label="回答")
            submit.click(answer_single_image, [image, question, structured, model_name, image_mock], image_output)

        with gr.Tab("PDF 解析"):
            pdf = gr.File(type="filepath", file_types=[".pdf"], label="上传 PDF")
            pdf_model = gr.Textbox(value=default_model, label="模型名称")
            with gr.Row():
                start_page = gr.Number(value=None, precision=0, label="起始页（可空）")
                end_page = gr.Number(value=None, precision=0, label="结束页（可空）")
            with gr.Row():
                resume = gr.Checkbox(value=True, label="断点续跑：跳过已有 page_*.json")
                overwrite_images = gr.Checkbox(value=False, label="重新渲染已有页面图片")
                pdf_mock = gr.Checkbox(value=False, label="Mock 模式（不加载 GPU 模型）")
            parse_button = gr.Button("逐页解析 PDF", variant="primary")
            parse_status = gr.Textbox(label="解析状态")
            results_file = gr.Textbox(label="results.json 路径")
            parse_button.click(
                parse_pdf_ui,
                [pdf, pdf_model, start_page, end_page, resume, overwrite_images, pdf_mock],
                [parse_status, results_file],
            )

        with gr.Tab("PDF 检索"):
            rag_results_file = gr.Textbox(label="PDF 解析 results.json 路径")
            embedding_model = gr.Textbox(value=default_embedding, label="Embedding 模型")
            build_button = gr.Button("构建 FAISS 索引")
            build_status = gr.Textbox(label="索引状态")
            index_dir = gr.Textbox(label="索引目录")
            build_button.click(build_rag_ui, [rag_results_file, embedding_model], [build_status, index_dir])

            pdf_question = gr.Textbox(value="这份资料主要讲了什么？", label="针对 PDF 提问")
            rag_model_name = gr.Textbox(value=default_model, label="回答模型名称")
            rag_mock = gr.Checkbox(value=False, label="Mock 模式（不加载 GPU 模型）")
            top_k = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Top-K")
            with gr.Row():
                search_button = gr.Button("检索相关页面")
                answer_button = gr.Button("检索并回答", variant="primary")
            search_output = gr.Markdown(label="检索结果")
            search_button.click(search_pdf_ui, [pdf_question, index_dir, top_k], search_output)
            answer_button.click(answer_pdf_ui, [pdf_question, index_dir, top_k, rag_model_name, rag_mock], search_output)

        with gr.Tab("Markdown 笔记"):
            note_results_file = gr.Textbox(label="PDF 解析 results.json 路径")
            note_button = gr.Button("生成 Markdown 笔记", variant="primary")
            note_output = gr.Markdown(label="笔记")
            note_path = gr.Textbox(label="保存路径")
            note_button.click(generate_note_ui, note_results_file, [note_output, note_path])

        with gr.Tab("结果浏览"):
            browse_results_file = gr.Textbox(label="PDF 解析 results.json 路径")
            browse_page = gr.Number(value=1, precision=0, label="页码")
            with gr.Row():
                browse_button = gr.Button("查看页面", variant="primary")
                export_button = gr.Button("导出页面解析 Markdown")
            browse_status = gr.Textbox(label="状态")
            browse_image = gr.Image(type="filepath", label="页面图片")
            browse_markdown = gr.Markdown(label="页面解析")
            export_path = gr.Textbox(label="导出路径")
            browse_button.click(
                browse_results_ui,
                [browse_results_file, browse_page],
                [browse_image, browse_markdown, browse_status],
            )
            export_button.click(export_results_ui, browse_results_file, [browse_status, export_path])

    return demo


if __name__ == "__main__":
    runtime = CONFIG.get("runtime", {})
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=runtime.get("server_name", "0.0.0.0"),
        server_port=int(runtime.get("server_port", 7860)),
        share=bool(runtime.get("share_gradio", False)),
    )
