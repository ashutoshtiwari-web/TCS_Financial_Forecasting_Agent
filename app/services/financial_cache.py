"""On-disk cache for deterministic/LLM financial extraction results.

PDF extraction and LLM enrichment are expensive. This cache keys entries by source
file path hash and invalidates automatically when file content SHA256 changes.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _cache_dir() -> Path:
    """Directory for financial extraction cache files; created on demand."""
    path = settings.data_path / "cache" / "financial"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    """Hash file bytes to detect content changes (not just filename/path changes)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(doc_path: Path) -> Path:
    """Stable cache filename derived from resolved path; human-readable prefix for debugging."""
    resolved = str(doc_path.resolve())
    key = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_path.stem)[:80]
    return _cache_dir() / f"{safe}__{key}.json"


def load_cached_extraction(doc_path: Path) -> dict[str, Any] | None:
    """Return cached extraction result when present and content hash still matches."""
    cache_path = _cache_path(doc_path)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        # Invalidate if the underlying PDF changed.
        if payload.get("content_sha256") != _file_sha256(doc_path):
            logger.info("Financial cache stale for %s", doc_path.name)
            return None
        result = payload.get("result")
        if isinstance(result, dict):
            logger.info("Financial cache hit for %s", doc_path.name)
            return result
    except Exception as e:
        # Cache is an optimization; corrupted cache should not fail the request.
        logger.warning("Failed reading financial cache for %s: %s", doc_path.name, e)
    return None


def save_cached_extraction(doc_path: Path, result: dict[str, Any]) -> None:
    """Atomically persist extraction result for future requests."""
    cache_path = _cache_path(doc_path)
    payload = {
        "source": doc_path.name,
        "path": str(doc_path.resolve()),
        "content_sha256": _file_sha256(doc_path),
        "mtime_ns": doc_path.stat().st_mtime_ns,
        "size": doc_path.stat().st_size,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    try:
        # Write-then-replace avoids partially written cache files on crashes.
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cache_path)
        logger.info("Saved financial cache for %s", doc_path.name)
    except Exception as e:
        logger.warning("Failed writing financial cache for %s: %s", doc_path.name, e)
