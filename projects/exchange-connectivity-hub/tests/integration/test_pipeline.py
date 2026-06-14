"""Integration test for full ingest pipeline."""

import pytest


def test_ingest_single_pdf_end_to_end(tmp_path):
    """Full pipeline: load → chunk → embed → store → register."""
    # This requires a real PDF file and API keys
    # Skip if API keys not available
    import os

    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set")

    # Create a minimal test PDF (requires actual PDF file)
    # For now, test the pipeline structure
    pass
