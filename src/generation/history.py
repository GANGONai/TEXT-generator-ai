"""Per-model history of generated lyrics."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import HISTORY_DIR
from ..utils import new_id, now_iso, read_json, write_json
from .generator import GenerationResult


class HistoryStore:
    """JSON-backed history of generation results, one file per model slug."""

    MAX_ITEMS = 200

    def __init__(self, root: Path = HISTORY_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_slug: str) -> Path:
        return self.root / f"{model_slug}.json"

    def list(self, model_slug: str) -> List[Dict[str, Any]]:
        return read_json(self._path(model_slug), default=[]) or []

    def add(self, model_slug: str, result: GenerationResult) -> Dict[str, Any]:
        entry = {
            "id": new_id(),
            "created_at": now_iso(),
            **asdict(result),
        }
        items = self.list(model_slug)
        items.insert(0, entry)
        if len(items) > self.MAX_ITEMS:
            items = items[: self.MAX_ITEMS]
        write_json(self._path(model_slug), items)
        return entry

    def clear(self, model_slug: str) -> None:
        write_json(self._path(model_slug), [])

    def get(self, model_slug: str, entry_id: str) -> Optional[Dict[str, Any]]:
        for item in self.list(model_slug):
            if item.get("id") == entry_id:
                return item
        return None
