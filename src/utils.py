"""Small helpers shared across modules."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Slugs / ids
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9а-яё]+", flags=re.IGNORECASE)


def slugify(value: str, *, max_len: int = 60) -> str:
    """Normalise a string into a filesystem-safe slug.

    Keeps Cyrillic letters intact so Russian model names stay readable.
    """

    if not value:
        return "untitled"
    normalised = unicodedata.normalize("NFKC", value).strip().lower()
    slug = _SLUG_RE.sub("-", normalised).strip("-")
    return (slug or "untitled")[:max_len]


def new_id() -> str:
    """Short, sortable, unique id."""

    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, returning ``default`` if it does not exist or is empty."""

    if not path.exists() or path.stat().st_size == 0:
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to disk (write-then-rename)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """ISO-8601 timestamp in UTC, second-precision."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_duration(seconds: float) -> str:
    """Render a duration like ``1h 23m 04s``."""

    seconds = max(0, int(seconds))
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalise_lyrics(text: str) -> str:
    """Strip BOM/zero-width chars and collapse excessive blank lines."""

    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def truncate(text: str, max_chars: int = 80) -> str:
    """Single-line preview for list views."""

    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[: max_chars - 1] + "…"


def filter_songs(songs: Iterable[Dict[str, Any]], query: Optional[str]) -> List[Dict[str, Any]]:
    """Case-insensitive substring filter over title / text / genre."""

    items = list(songs)
    if not query:
        return items
    q = query.strip().lower()
    if not q:
        return items
    return [
        s for s in items
        if q in s.get("title", "").lower()
        or q in s.get("text", "").lower()
        or q in (s.get("genre") or "").lower()
    ]
