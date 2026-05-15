from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf_parser import pdf_to_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDF pages to PNG without loading any VLM model.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file.")
    parser.add_argument("--output", required=True, help="Output directory for rendered page images.")
    parser.add_argument("--zoom", type=float, default=2.0, help="PyMuPDF render zoom.")
    parser.add_argument("--start-page", type=int, default=None, help="First page to render, 1-indexed.")
    parser.add_argument("--end-page", type=int, default=None, help="Last page to render, 1-indexed.")
    parser.add_argument("--overwrite", action="store_true", help="Render again even if page PNG already exists.")
    args = parser.parse_args()

    images = pdf_to_images(
        args.pdf,
        args.output,
        zoom=args.zoom,
        start_page=args.start_page,
        end_page=args.end_page,
        overwrite=args.overwrite,
    )
    print(f"Rendered/found {len(images)} page images.")
    print(f"Output: {Path(args.output)}")


if __name__ == "__main__":
    main()
