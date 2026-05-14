from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.structured_parser import parse_image_analysis
from src.utils import get_gpu_memory_summary, read_json, write_json
from src.vlm_infer import VLMInferencer


def normalize_type(text: str) -> str:
    allowed = ["exam_question", "paper_figure", "code_screenshot", "slide", "chart", "other"]
    for item in allowed:
        if item in text:
            return item
    match = re.search(r"(exam_question|paper_figure|code_screenshot|slide|chart|other)", text)
    return match.group(1) if match else "unknown"


def evaluate(benchmark_path: str, output_path: str, config_path: str, model_name: str | None = None) -> dict:
    benchmark = read_json(benchmark_path)
    inferencer = VLMInferencer(config_path=config_path, model_name=model_name)
    samples = []
    type_correct = 0
    total_latency = 0.0

    for item in benchmark:
        start = time.time()
        raw_answer = inferencer.answer_image(item["image"], item.get("question", "请解析这张图片"), structured=True)
        latency = time.time() - start
        parsed = parse_image_analysis(raw_answer)
        pred_type = normalize_type(parsed["image_type"])
        is_type_correct = pred_type == item.get("type")
        type_correct += int(is_type_correct)
        total_latency += latency
        samples.append(
            {
                "id": item.get("id"),
                "image": item.get("image"),
                "expected_type": item.get("type"),
                "pred_type": pred_type,
                "type_correct": is_type_correct,
                "latency_sec": latency,
                "gpu_memory": get_gpu_memory_summary(),
                "raw_answer": raw_answer,
                "manual_scores": {
                    "ocr": None,
                    "key_concepts": None,
                    "answer_correctness": None,
                    "hallucination": None,
                },
            }
        )

    summary = {
        "model": model_name or inferencer.model_name,
        "num_samples": len(benchmark),
        "type_accuracy": type_correct / len(benchmark) if benchmark else 0,
        "avg_latency_sec": total_latency / len(benchmark) if benchmark else 0,
        "samples": samples,
    }
    write_json(summary, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on a small multimodal benchmark.")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", required=True, help="Output result JSON path.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model config path.")
    parser.add_argument("--model", default=None, help="Override model name.")
    args = parser.parse_args()
    result = evaluate(args.benchmark, args.output, args.config, args.model)
    print(f"Model: {result['model']}")
    print(f"Samples: {result['num_samples']}")
    print(f"Type accuracy: {result['type_accuracy']:.2%}")
    print(f"Average latency: {result['avg_latency_sec']:.2f}s")
    print(f"Saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
