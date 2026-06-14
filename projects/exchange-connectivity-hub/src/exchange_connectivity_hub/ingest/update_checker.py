"""Document change detection via MD5 hash comparison."""

import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from exchange_connectivity_hub.ingest.doc_registry import DocRegistry
from exchange_connectivity_hub.ingest.loader import extract_text_for_hash


def compute_hash(text: str) -> str:
    """Compute MD5 hash of normalized text.

    Args:
        text: Text content to hash

    Returns:
        Hexadecimal MD5 hash string
    """
    normalized = extract_text_for_hash(text)
    return hashlib.md5(normalized.encode()).hexdigest()


def download_pdf(url: str, dest_path: Path) -> None:
    """Download PDF from URL to destination path.

    Args:
        url: Source URL
        dest_path: Destination file path
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest_path)


def check_for_updates(registry: DocRegistry) -> dict[str, dict[str, str]]:
    """Check all tracked documents for content changes.

    Downloads each document from its source URL, computes hash,
    and compares with stored hash.

    Args:
        registry: DocRegistry instance

    Returns:
        Dict mapping filename -> {"old_hash": str, "new_hash": str}
        for documents that have changed
    """
    from exchange_connectivity_hub.ingest.loader import load_pdf

    changes = {}

    for filename, entry in registry.data.items():
        source_url = entry.get("source_url")
        if not source_url:
            continue  # Skip docs without URLs

        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                download_pdf(source_url, tmp_path)

            # Load and extract text
            docs = load_pdf(
                pdf_path=tmp_path,
                exchange=entry["exchange"],
                doc_type=entry["doc_type"],
            )

            # Compute hash
            from exchange_connectivity_hub.ingest.loader import extract_full_text_from_docs

            full_text = extract_full_text_from_docs(docs)
            new_hash = compute_hash(full_text)
            old_hash = entry.get("current_hash")

            # Compare
            if old_hash and new_hash != old_hash:
                changes[filename] = {
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                }

            # Clean up temp file
            tmp_path.unlink()

        except Exception as e:
            # Log error but continue checking other docs
            print(f"Error checking {filename}: {e}")
            continue

    return changes
