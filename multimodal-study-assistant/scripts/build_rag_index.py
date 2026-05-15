from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import build_index_from_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from parsed PDF results.")
    parser.add_argument("--parsed", required=True, help="Path to data/parsed_pages/.../results.json.")
    parser.add_argument("--output", required=True, help="Output directory for FAISS index.")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5", help="SentenceTransformer model.")
    parser.add_argument("--device", default="cpu", help="Embedding device, default cpu to avoid occupying busy GPUs.")
    args = parser.parse_args()

    index = build_index_from_results(args.parsed, args.output, embedding_model=args.embedding_model, device=args.device)
    print(f"Indexed {len(index.docs)} pages.")
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    main()
