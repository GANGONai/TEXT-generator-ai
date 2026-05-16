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
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..logger import get_logger
from ..utils import filter_songs, new_id, normalise_lyrics, now_iso, read_json, write_json

# Recognised header lines at the top of a TXT song block. Case-insensitive,
# tolerates leading/trailing whitespace, accepts both Russian and English
# field names. The third group lets ``Название:`` sit on its own line *or*
# carry the value inline (``Название: Ночной город``).
_TXT_HEADER_RE = re.compile(
    r"^\s*(?P<field>название|title|жанр|genre|текст|lyrics)\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
_TXT_TITLE_FIELDS = {"название", "title"}
_TXT_GENRE_FIELDS = {"жанр", "genre"}
_TXT_BODY_FIELDS = {"текст", "lyrics"}


def _blocks_have_headers(blocks: List[str]) -> bool:
    """Return True if any block looks like a structured song (has a
    ``Название:`` / ``Title:`` / ``Жанр:`` / ``Genre:`` / ``Текст:`` /
    ``Lyrics:`` header on one of its first three non-empty lines).
    """

    for block in blocks:
        head_lines = [ln for ln in block.splitlines()[:5] if ln.strip()][:3]
        for line in head_lines:
            if _TXT_HEADER_RE.match(line):
                return True
    return False


def _parse_txt_block(block: str, *, fallback_title: str) -> Dict[str, Any]:
    """Parse a single block from a structured TXT file.

    Recognises ``Название:`` / ``Title:`` / ``Жанр:`` / ``Genre:`` / ``Текст:``
    / ``Lyrics:`` headers at the top of the block. Everything after the
    headers (or after a ``Текст:`` line) is treated as the song body.
    """

    title: Optional[str] = None
    genre: Optional[str] = None
    body_lines: List[str] = []
    consuming_body = False

    for line in block.splitlines():
        if consuming_body:
            body_lines.append(line)
            continue
        match = _TXT_HEADER_RE.match(line)
        if match:
            field_name = match.group("field").lower()
            value = match.group("value").strip()
            if field_name in _TXT_TITLE_FIELDS:
                if value:
                    title = value
                # No inline value → next non-header line(s) still belong to
                # other headers; keep scanning.
                continue
            if field_name in _TXT_GENRE_FIELDS:
                if value:
                    genre = value
                continue
            if field_name in _TXT_BODY_FIELDS:
                # Body starts here. Inline value (rare) joins the body.
                if value:
                    body_lines.append(value)
                consuming_body = True
                continue
        # First non-header line ⇒ body starts here (no ``Текст:`` marker).
        body_lines.append(line)
        consuming_body = True

    body = "\n".join(body_lines).strip()
    return {
        "title": (title or fallback_title).strip(),
        "text": body,
        "genre": genre,
    }

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
        """Import one or many songs from a TXT file.

        Three supported layouts (auto-detected):

        1) **Structured** — recommended for several songs with title & genre.
           Each song is a block separated from the next by **two blank lines**
           (i.e. three newlines). Within a block, the first lines may be any
           combination of ``Название: …`` / ``Title: …``, ``Жанр: …`` /
           ``Genre: …``, and ``Текст:`` / ``Lyrics:``. Anything after a
           ``Текст:`` line (or after the last header line) is the song body.

           Example::

               Название: Ночной город
               Жанр: рок
               Текст:
               Я иду по улице ночной
               Город спит, но не со мной

               Свети, моя звезда


               Название: Морская
               Жанр: шансон
               Море, море, мир бездонный

        2) **Plain multi-song** — blocks separated by two blank lines but no
           header markers. Each block becomes a song titled
           ``<default_title> #1``, ``#2``, …

        3) **Single song** — no two-blank-line separator anywhere. The whole
           file is imported as one song titled ``default_title``.
        """

        text_norm = raw.replace("\r\n", "\n").replace("\r", "\n")
        # 3+ consecutive newlines split songs; 2 newlines are a verse break.
        blocks = [b for b in re.split(r"\n{3,}", text_norm) if b.strip()]
        if not blocks:
            return 0

        if _blocks_have_headers(blocks):
            items = [
                _parse_txt_block(block, fallback_title=f"{default_title} #{i + 1}")
                for i, block in enumerate(blocks)
            ]
            return self.import_texts(items)

        # No headers anywhere — fall back to the historical behaviour.
        if len(blocks) == 1:
            return self.import_texts([{"title": default_title, "text": text_norm}])
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
