from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_parser import parse_page_images
from src.vlm_infer import MockVLMInferencer, VLMInferencer


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse existing page_*.png images with VLM.")
    parser.add_argument("--images-dir", required=True, help="Directory containing page_*.png images.")
    parser.add_argument("--output", required=True, help="Output directory for page JSON files and results.json.")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model config path.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--start-page", type=int, default=None, help="First page image to parse, 1-indexed.")
    parser.add_argument("--end-page", type=int, default=None, help="Last page image to parse, 1-indexed.")
    parser.add_argument("--no-resume", action="store_true", help="Re-parse pages even if page_*.json exists.")
    parser.add_argument("--mock", action="store_true", help="Use MOCK_VLM=1 and do not load GPU model.")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_VLM"] = "1"

    images = sorted(Path(args.images_dir).glob("page_*.png"), key=lambda path: int(path.stem.split("_")[-1]))
    if args.start_page is not None:
        images = [path for path in images if int(path.stem.split("_")[-1]) >= args.start_page]
    if args.end_page is not None:
        images = [path for path in images if int(path.stem.split("_")[-1]) <= args.end_page]

    inferencer = (
        MockVLMInferencer(config_path=args.config, model_name=args.model)
        if args.mock
        else VLMInferencer(config_path=args.config, model_name=args.model)
    )
    records = parse_page_images(images, args.output, inferencer=inferencer, resume=not args.no_resume)
    error_count = sum(1 for item in records if item.get("error"))
    print(f"Parsed/summarized {len(records)} pages.")
    print(f"Errors: {error_count}")
    print(f"Results: {Path(args.output) / 'results.json'}")


if __name__ == "__main__":
    main()
