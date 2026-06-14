"""CLI entry point for ingest pipeline."""

import argparse
import sys
from pathlib import Path

from exchange_connectivity_hub.ingest.pipeline import (
    ingest_all_from_registry,
    ingest_single_pdf,
)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest exchange PDF documents")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest-single command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single PDF")
    ingest_parser.add_argument("--doc", required=True, help="PDF filename in data/raw/")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-ingest")

    # ingest-all command
    subparsers.add_parser("ingest-all", help="Ingest all registered PDFs")

    # reingest command (alias for ingest --force)
    reingest_parser = subparsers.add_parser("reingest", help="Re-ingest a single PDF")
    reingest_parser.add_argument("--doc", required=True, help="PDF filename in data/raw/")

    args = parser.parse_args()

    if args.command == "ingest":
        raw_dir = Path("data/raw")
        pdf_path = raw_dir / args.doc
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            return 1

        # Get exchange/doc_type from registry
        from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

        registry = DocRegistry(Path("data/doc_registry.json"))

        if args.doc not in registry.data:
            print(f"Error: {args.doc} not found in doc_registry.json")
            print("Available documents:", list(registry.data.keys()))
            return 1

        entry = registry.data[args.doc]
        result = ingest_single_pdf(
            pdf_path=pdf_path,
            exchange=entry["exchange"],
            doc_type=entry["doc_type"],
            source_url=entry.get("source_url"),
            force_reingest=args.force,
        )
        print(f"Status: {result['status']}")
        if result["status"] == "success":
            print(f"Chunks ingested: {result['chunks_ingested']}")
        else:
            print(f"Reason: {result.get('reason', 'N/A')}")
        return 0

    elif args.command == "ingest-all":
        results = ingest_all_from_registry()
        for r in results:
            print(f"{r['filename']}: {r['status']}")
            if r.get("reason"):
                print(f"  Reason: {r['reason']}")
        return 0

    elif args.command == "reingest":
        # Reuse ingest logic with force=True
        raw_dir = Path("data/raw")
        pdf_path = raw_dir / args.doc
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            return 1

        from exchange_connectivity_hub.ingest.doc_registry import DocRegistry

        registry = DocRegistry(Path("data/doc_registry.json"))

        if args.doc not in registry.data:
            print(f"Error: {args.doc} not found in doc_registry.json")
            return 1

        entry = registry.data[args.doc]
        result = ingest_single_pdf(
            pdf_path=pdf_path,
            exchange=entry["exchange"],
            doc_type=entry["doc_type"],
            source_url=entry.get("source_url"),
            force_reingest=True,
        )
        print(f"Status: {result['status']}")
        if result["status"] == "success":
            print(f"Chunks ingested: {result['chunks_ingested']}")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
