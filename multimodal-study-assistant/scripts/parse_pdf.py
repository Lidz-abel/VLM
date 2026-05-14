from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_parser import parse_pdf
from src.vlm_infer import VLMInferencer


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF page by page with Qwen2.5-VL.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file.")
    parser.add_argument("--output", required=True, help="Output directory for rendered pages and JSON results.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model config path.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--zoom", type=float, default=2.0, help="PyMuPDF render zoom.")
    args = parser.parse_args()

    inferencer = VLMInferencer(config_path=args.config, model_name=args.model)
    records = parse_pdf(args.pdf, args.output, inferencer=inferencer, zoom=args.zoom)
    print(f"Parsed {len(records)} pages.")
    print(f"Results: {Path(args.output) / 'results.json'}")


if __name__ == "__main__":
    main()
