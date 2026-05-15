from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.result_exporter import export_results_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Export parsed PDF results.json to Markdown.")
    parser.add_argument("--results", required=True, help="Path to parsed results.json.")
    parser.add_argument("--output", default=None, help="Output Markdown path.")
    args = parser.parse_args()

    output = export_results_markdown(args.results, args.output)
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()
