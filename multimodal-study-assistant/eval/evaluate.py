from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.structured_parser import parse_image_analysis
from src.run_logger import log_event
from src.utils import get_gpu_memory_summary, read_json, write_json
from src.vlm_infer import MockVLMInferencer, VLMInferencer


def normalize_type(text: str) -> str:
    allowed = ["exam_question", "paper_figure", "code_screenshot", "slide", "chart", "other"]
    for item in allowed:
        if item in text:
            return item
    match = re.search(r"(exam_question|paper_figure|code_screenshot|slide|chart|other)", text)
    return match.group(1) if match else "unknown"


def evaluate(
    benchmark_path: str,
    output_path: str,
    config_path: str,
    model_name: str | None = None,
    mock: bool = False,
) -> dict:
    benchmark = read_json(benchmark_path)
    if mock:
        os.environ["MOCK_VLM"] = "1"
    inferencer = (
        MockVLMInferencer(config_path=config_path, model_name=model_name)
        if mock
        else VLMInferencer(config_path=config_path, model_name=model_name)
    )
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
        log_event(
            "benchmark_sample_evaluated",
            {
                "id": item.get("id"),
                "image": item.get("image"),
                "expected_type": item.get("type"),
                "pred_type": pred_type,
                "type_correct": is_type_correct,
                "latency_sec": round(latency, 3),
                "model": inferencer.model_name,
            },
        )

    summary = {
        "model": model_name or inferencer.model_name,
        "num_samples": len(benchmark),
        "type_accuracy": type_correct / len(benchmark) if benchmark else 0,
        "avg_latency_sec": total_latency / len(benchmark) if benchmark else 0,
        "samples": samples,
    }
    write_json(summary, output_path)
    log_event(
        "benchmark_evaluated",
        {
            "benchmark": benchmark_path,
            "output": output_path,
            "model": summary["model"],
            "num_samples": summary["num_samples"],
            "type_accuracy": summary["type_accuracy"],
            "avg_latency_sec": summary["avg_latency_sec"],
        },
    )
    return summary


def write_manual_score_csv(result: dict, csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "image",
        "expected_type",
        "pred_type",
        "type_correct",
        "latency_sec",
        "ocr_score_0_3",
        "concept_score_0_3",
        "answer_score_0_3",
        "hallucination_0_or_1",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in result.get("samples", []):
            writer.writerow(
                {
                    "id": sample.get("id", ""),
                    "image": sample.get("image", ""),
                    "expected_type": sample.get("expected_type", ""),
                    "pred_type": sample.get("pred_type", ""),
                    "type_correct": sample.get("type_correct", ""),
                    "latency_sec": round(float(sample.get("latency_sec", 0)), 3),
                    "ocr_score_0_3": "",
                    "concept_score_0_3": "",
                    "answer_score_0_3": "",
                    "hallucination_0_or_1": "",
                    "notes": "",
                }
            )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on a small multimodal benchmark.")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON path.")
    parser.add_argument("--output", required=True, help="Output result JSON path.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model config path.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--score-csv", default=None, help="Output CSV template for manual scoring.")
    parser.add_argument("--mock", action="store_true", help="Use MOCK_VLM=1 and do not load GPU model.")
    args = parser.parse_args()
    result = evaluate(args.benchmark, args.output, args.config, args.model, mock=args.mock)
    score_csv = args.score_csv or str(Path(args.output).with_suffix(".manual_scores.csv"))
    write_manual_score_csv(result, score_csv)
    print(f"Model: {result['model']}")
    print(f"Samples: {result['num_samples']}")
    print(f"Type accuracy: {result['type_accuracy']:.2%}")
    print(f"Average latency: {result['avg_latency_sec']:.2f}s")
    print(f"Saved to: {Path(args.output)}")
    print(f"Manual score CSV: {score_csv}")


if __name__ == "__main__":
    main()
