from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.result_exporter import failed_pages, failed_pages_markdown
from src.utils import read_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="List failed pages from parsed PDF results.json.")
    parser.add_argument("--results", required=True, help="Path to parsed results.json.")
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    records = read_json(args.results)
    failed = failed_pages(records)
    print(f"Failed pages: {len(failed)}")
    for item in failed:
        print(f"page={item['page']} image={item['image_path']} error={item['error']}")
    if args.output:
        write_text(failed_pages_markdown(args.results), args.output)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
