from __future__ import annotations

import gradio as gr

from src.note_generator import generate_note_from_records
from src.pdf_parser import parse_pdf
from src.prompts import build_rag_prompt
from src.rag import PdfRagIndex
from src.utils import PROJECT_ROOT, get_gpu_memory_summary, load_yaml, safe_stem
from src.vlm_infer import get_default_inferencer


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CONFIG = load_yaml(CONFIG_PATH)


def answer_single_image(image_path: str, question: str, structured: bool, model_name: str) -> str:
    if not image_path:
        return "请先上传图片。"
    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    answer = inferencer.answer_image(image_path=image_path, question=question, structured=structured)
    return f"{answer}\n\n---\n{get_gpu_memory_summary()}"


def parse_pdf_ui(pdf_path: str, model_name: str, progress=gr.Progress()) -> tuple[str, str]:
    if not pdf_path:
        return "请先上传 PDF。", ""

    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    output_dir = PROJECT_ROOT / "data" / "parsed_pages" / safe_stem(pdf_path)

    def update(done: int, total: int) -> None:
        progress(done / total, desc=f"解析 PDF：{done}/{total}")

    records = parse_pdf(pdf_path=pdf_path, output_dir=output_dir, inferencer=inferencer, progress_callback=update)
    result_file = output_dir / "results.json"
    return f"完成解析 {len(records)} 页。\n结果文件：{result_file}", str(result_file)


def build_rag_ui(results_file: str, embedding_model: str) -> tuple[str, str]:
    if not results_file:
        return "请先解析 PDF，或填写 results.json 路径。", ""
    index_dir = PROJECT_ROOT / "data" / "vector_db" / safe_stem(results_file)
    from src.rag import build_index_from_results

    build_index_from_results(results_file, index_dir, embedding_model=embedding_model)
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
    return "\n\n".join(blocks)


def answer_pdf_ui(question: str, index_dir: str, top_k: int, model_name: str) -> str:
    if not question or not index_dir:
        return "请填写问题和索引目录。"
    index = PdfRagIndex.load(index_dir)
    contexts = index.search(question, top_k=int(top_k))
    if not contexts:
        return "没有检索到相关页面，无法回答。"
    prompt = build_rag_prompt(question, contexts)
    inferencer = get_default_inferencer(CONFIG_PATH, model_name.strip() or None)
    answer = inferencer.answer_text(prompt)
    sources = "，".join(f"第 {item['page']} 页" for item in contexts)
    return f"{answer}\n\n---\n检索依据：{sources}\n\n{get_gpu_memory_summary()}"


def generate_note_ui(results_file: str) -> tuple[str, str]:
    if not results_file:
        return "请先提供 PDF 解析 results.json。", ""
    from src.utils import read_json

    records = read_json(results_file)
    output_path = PROJECT_ROOT / "outputs" / f"{safe_stem(results_file)}_notes.md"
    note = generate_note_from_records(records, output_path=output_path)
    return note, str(output_path)


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
                    submit = gr.Button("分析图片", variant="primary")
            image_output = gr.Markdown(label="回答")
            submit.click(answer_single_image, [image, question, structured, model_name], image_output)

        with gr.Tab("PDF 解析"):
            pdf = gr.File(type="filepath", file_types=[".pdf"], label="上传 PDF")
            pdf_model = gr.Textbox(value=default_model, label="模型名称")
            parse_button = gr.Button("逐页解析 PDF", variant="primary")
            parse_status = gr.Textbox(label="解析状态")
            results_file = gr.Textbox(label="results.json 路径")
            parse_button.click(parse_pdf_ui, [pdf, pdf_model], [parse_status, results_file])

        with gr.Tab("PDF 检索"):
            rag_results_file = gr.Textbox(label="PDF 解析 results.json 路径")
            embedding_model = gr.Textbox(value=default_embedding, label="Embedding 模型")
            build_button = gr.Button("构建 FAISS 索引")
            build_status = gr.Textbox(label="索引状态")
            index_dir = gr.Textbox(label="索引目录")
            build_button.click(build_rag_ui, [rag_results_file, embedding_model], [build_status, index_dir])

            pdf_question = gr.Textbox(value="这份资料主要讲了什么？", label="针对 PDF 提问")
            rag_model_name = gr.Textbox(value=default_model, label="回答模型名称")
            top_k = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Top-K")
            with gr.Row():
                search_button = gr.Button("检索相关页面")
                answer_button = gr.Button("检索并回答", variant="primary")
            search_output = gr.Markdown(label="检索结果")
            search_button.click(search_pdf_ui, [pdf_question, index_dir, top_k], search_output)
            answer_button.click(answer_pdf_ui, [pdf_question, index_dir, top_k, rag_model_name], search_output)

        with gr.Tab("Markdown 笔记"):
            note_results_file = gr.Textbox(label="PDF 解析 results.json 路径")
            note_button = gr.Button("生成 Markdown 笔记", variant="primary")
            note_output = gr.Markdown(label="笔记")
            note_path = gr.Textbox(label="保存路径")
            note_button.click(generate_note_ui, note_results_file, [note_output, note_path])

    return demo


if __name__ == "__main__":
    runtime = CONFIG.get("runtime", {})
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=runtime.get("server_name", "0.0.0.0"),
        server_port=int(runtime.get("server_port", 7860)),
        share=bool(runtime.get("share_gradio", False)),
    )
