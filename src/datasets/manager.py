"""Per-model song dataset (CRUD + search + bulk import).

The dataset for a model is stored as a single JSON file (``songs.json``)
that lives next to the model's weights. Each entry has an id, title,
text, optional genre, and creation timestamp.

Storing it as JSON keeps the implementation trivial and the dataset
portable — no SQLite, no external services.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..logger import get_logger
from ..utils import filter_songs, new_id, normalise_lyrics, now_iso, read_json, write_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Song:
    id: str
    title: str
    text: str
    genre: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Song":
        return cls(
            id=raw.get("id") or new_id(),
            title=raw.get("title") or "Без названия",
            text=normalise_lyrics(raw.get("text", "")),
            genre=raw.get("genre") or None,
            created_at=raw.get("created_at") or now_iso(),
            updated_at=raw.get("updated_at") or raw.get("created_at") or now_iso(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SongManager:
    """Per-model songs database backed by ``songs.json``."""

    def __init__(self, songs_path: Path) -> None:
        self.path = Path(songs_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            write_json(self.path, [])

    # -- low level ---------------------------------------------------------
    def _load(self) -> List[Song]:
        raw = read_json(self.path, default=[]) or []
        return [Song.from_dict(item) for item in raw]

    def _save(self, songs: Iterable[Song]) -> None:
        write_json(self.path, [s.to_dict() for s in songs])

    # -- queries -----------------------------------------------------------
    def list(self, query: Optional[str] = None) -> List[Song]:
        songs = self._load()
        if query:
            return [Song.from_dict(s) for s in filter_songs([s.to_dict() for s in songs], query)]
        return songs

    def get(self, song_id: str) -> Optional[Song]:
        for s in self._load():
            if s.id == song_id:
                return s
        return None

    def count(self) -> int:
        return len(self._load())

    def total_chars(self) -> int:
        return sum(len(s.text) for s in self._load())

    # -- mutations ---------------------------------------------------------
    def add(self, title: str, text: str, genre: Optional[str] = None) -> Song:
        text = normalise_lyrics(text)
        if not text.strip():
            raise ValueError("Текст песни не может быть пустым.")
        title = (title or "").strip() or "Без названия"
        song = Song(id=new_id(), title=title, text=text, genre=genre)
        songs = self._load()
        songs.append(song)
        self._save(songs)
        log.info("Added song %r (%d chars) to %s", title, len(text), self.path)
        return song

    def update(
        self,
        song_id: str,
        *,
        title: Optional[str] = None,
        text: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> Song:
        songs = self._load()
        for i, s in enumerate(songs):
            if s.id == song_id:
                if title is not None:
                    s.title = title.strip() or s.title
                if text is not None:
                    s.text = normalise_lyrics(text)
                if genre is not None:
                    s.genre = genre or None
                s.updated_at = now_iso()
                songs[i] = s
                self._save(songs)
                return s
        raise KeyError(f"Song {song_id!r} not found.")

    def delete(self, song_id: str) -> bool:
        songs = self._load()
        remaining = [s for s in songs if s.id != song_id]
        if len(remaining) == len(songs):
            return False
        self._save(remaining)
        return True

    def clear(self) -> None:
        self._save([])

    # -- bulk import -------------------------------------------------------
    def import_texts(self, texts: List[Dict[str, str]]) -> int:
        """Append a list of ``{title, text, genre?}`` dicts."""

        songs = self._load()
        added = 0
        for item in texts:
            text = normalise_lyrics(item.get("text", ""))
            if not text:
                continue
            songs.append(
                Song(
                    id=new_id(),
                    title=(item.get("title") or "Без названия").strip(),
                    text=text,
                    genre=item.get("genre") or None,
                )
            )
            added += 1
        self._save(songs)
        return added

    def import_from_txt(self, raw: str, default_title: str = "Песня") -> int:
        """Split raw text on blank lines and import each block as a song."""

        blocks = [b.strip() for b in raw.replace("\r\n", "\n").split("\n\n\n") if b.strip()]
        if len(blocks) <= 1:
            # No triple-newline separator — treat the whole file as one song.
            return self.import_texts([{"title": default_title, "text": raw}])
        items = [
            {"title": f"{default_title} #{i + 1}", "text": block}
            for i, block in enumerate(blocks)
        ]
        return self.import_texts(items)

    def import_from_csv(self, raw: str) -> int:
        """Import a CSV with at least a ``text`` column (``title``/``genre`` optional)."""

        reader = csv.DictReader(io.StringIO(raw))
        if reader.fieldnames is None or "text" not in [
            (f or "").lower() for f in reader.fieldnames
        ]:
            raise ValueError("CSV must contain a 'text' column.")
        # Normalise header casing.
        rows = []
        for row in reader:
            norm = {(k or "").lower(): (v or "") for k, v in row.items()}
            rows.append(
                {
                    "title": norm.get("title") or "Без названия",
                    "text": norm.get("text", ""),
                    "genre": norm.get("genre") or None,
                }
            )
        return self.import_texts(rows)
