"""Helpers for parsing model output that should be JSON.

LLMs occasionally return markdown fences or extra prose. This loose parser keeps
the main agent/tool code small and consistent.
"""

import json
import re
from typing import Any


def parse_json_loose(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction.

    Returns:
        dict when a JSON object (or object-like payload) is found, else None.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    # Strip common markdown code fences: ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Fast path: valid JSON as-is.
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass

    # Fallback: extract the first top-level-looking object from surrounding text.
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            return None
    return None
