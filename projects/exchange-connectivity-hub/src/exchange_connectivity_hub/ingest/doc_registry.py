"""Document registry for tracking version history of ingested PDFs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> dict[str, Any]:
    """Load registry from file, creating empty dict if not exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Create empty registry
    save_registry({}, path)
    return {}


def save_registry(data: dict[str, Any], path: Path) -> None:
    """Save registry to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class DocRegistry:
    """Manages document version history and hash tracking."""

    def __init__(self, registry_path: Path) -> None:
        """Initialize with registry file path."""
        self.path = registry_path
        self.data = load_registry(registry_path)

    def register_doc(
        self,
        *,
        filename: str,
        source_url: str | None,
        exchange: str,
        doc_type: str,
        content_hash: str,
        chunks_count: int,
    ) -> None:
        """Register or update a document in the registry."""
        now = datetime.now(timezone.utc).isoformat()

        if filename not in self.data:
            # New document
            self.data[filename] = {
                "source_url": source_url,
                "exchange": exchange,
                "doc_type": doc_type,
                "current_hash": content_hash,
                "version_history": [
                    {
                        "hash": content_hash,
                        "ingested_at": now,
                        "chunks_count": chunks_count,
                    }
                ],
            }
        else:
            # Update existing document
            entry = self.data[filename]
            entry["current_hash"] = content_hash
            entry["version_history"].append(
                {
                    "hash": content_hash,
                    "ingested_at": now,
                    "chunks_count": chunks_count,
                }
            )

        self.save()

    def check_hash_changed(self, filename: str, new_hash: str) -> bool:
        """Check if document hash has changed.

        Returns True if:
        - Document not in registry (will register as new)
        - Hash differs from current_hash
        """
        if filename not in self.data:
            return True  # New doc
        return self.data[filename]["current_hash"] != new_hash

    def get_source_url(self, filename: str) -> str | None:
        """Get source URL for a document."""
        if filename not in self.data:
            return None
        return self.data[filename]["source_url"]

    def get_current_hash(self, filename: str) -> str | None:
        """Get current hash for a document."""
        if filename not in self.data:
            return None
        return self.data[filename]["current_hash"]

    def save(self) -> None:
        """Save registry to disk."""
        save_registry(self.data, self.path)

    def get_all_filenames(self) -> list[str]:
        """Get list of all tracked filenames."""
        return list(self.data.keys())
