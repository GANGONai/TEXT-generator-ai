"""High-level lifecycle manager for user-trained song-lyrics models.

Every user model lives under ``data/models/<slug>/`` and consists of:

* ``config.json`` — metadata (name, base model, status, training stats…)
* ``weights/`` — HF-style folder with weights + tokenizer (only after training)
* ``songs.json`` — dataset of uploaded lyrics for this model

The :class:`ModelManager` is the *single source of truth* for the rest of
the application. UI code, training and generation never touch the filesystem
directly — they all go through this class.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BASE_MODELS, DEFAULT_BASE_MODEL, MODELS_DIR
from ..logger import get_logger
from ..utils import new_id, now_iso, read_json, slugify, write_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_STATUSES = {"untrained", "training", "trained", "error"}


@dataclass
class ModelMeta:
    """Serialisable metadata for one user model."""

    id: str
    name: str
    slug: str
    base_model: str
    created_at: str
    updated_at: str
    status: str = "untrained"
    last_error: Optional[str] = None
    training: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelMeta":
        return cls(
            id=raw["id"],
            name=raw["name"],
            slug=raw["slug"],
            base_model=raw["base_model"],
            created_at=raw["created_at"],
            updated_at=raw.get("updated_at", raw["created_at"]),
            status=raw.get("status", "untrained"),
            last_error=raw.get("last_error"),
            training=raw.get("training", {}) or {},
            stats=raw.get("stats", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages a directory of user models on disk."""

    def __init__(self, root: Path = MODELS_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------
    def model_dir(self, slug: str) -> Path:
        return self.root / slug

    def weights_dir(self, slug: str) -> Path:
        return self.model_dir(slug) / "weights"

    def songs_path(self, slug: str) -> Path:
        return self.model_dir(slug) / "songs.json"

    def config_path(self, slug: str) -> Path:
        return self.model_dir(slug) / "config.json"

    # -- listing ------------------------------------------------------------
    def list_models(self) -> List[ModelMeta]:
        """Return all models sorted by ``updated_at`` desc."""

        items: List[ModelMeta] = []
        for child in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not child.is_dir():
                continue
            raw = read_json(child / "config.json")
            if not raw:
                continue
            try:
                items.append(ModelMeta.from_dict(raw))
            except KeyError as exc:
                log.warning("Skipping malformed model %s: %s", child.name, exc)
        items.sort(key=lambda m: m.updated_at, reverse=True)
        return items

    def list_names(self) -> List[str]:
        return [m.name for m in self.list_models()]

    def exists(self, name_or_slug: str) -> bool:
        return self.model_dir(self._coerce_slug(name_or_slug)).exists()

    # -- CRUD ---------------------------------------------------------------
    def create(self, name: str, base_model: str = DEFAULT_BASE_MODEL) -> ModelMeta:
        """Create a new (empty, untrained) model."""

        if not name or not name.strip():
            raise ValueError("Model name cannot be empty.")
        if base_model not in BASE_MODELS:
            raise ValueError(
                f"Unknown base model '{base_model}'. "
                f"Choose one of: {', '.join(BASE_MODELS)}"
            )

        slug = self._unique_slug(name)
        meta = ModelMeta(
            id=new_id(),
            name=name.strip(),
            slug=slug,
            base_model=base_model,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self.model_dir(slug).mkdir(parents=True, exist_ok=True)
        write_json(self.config_path(slug), meta.to_dict())
        write_json(self.songs_path(slug), [])
        log.info("Created model %r (slug=%s, base=%s)", name, slug, base_model)
        return meta

    def delete(self, name_or_slug: str) -> bool:
        """Remove a model directory entirely. Returns True if it existed."""

        slug = self._coerce_slug(name_or_slug)
        path = self.model_dir(slug)
        if not path.exists():
            return False
        shutil.rmtree(path, ignore_errors=True)
        log.info("Deleted model %s", slug)
        return True

    # -- meta read/write ----------------------------------------------------
    def load_meta(self, name_or_slug: str) -> ModelMeta:
        slug = self._coerce_slug(name_or_slug)
        raw = read_json(self.config_path(slug))
        if not raw:
            raise FileNotFoundError(f"Model {slug!r} not found.")
        return ModelMeta.from_dict(raw)

    def save_meta(self, meta: ModelMeta) -> None:
        meta.updated_at = now_iso()
        write_json(self.config_path(meta.slug), meta.to_dict())

    def update_status(
        self,
        name_or_slug: str,
        status: str,
        *,
        error: Optional[str] = None,
        training: Optional[Dict[str, Any]] = None,
    ) -> ModelMeta:
        """Patch status + optional training/error info."""

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}")
        meta = self.load_meta(name_or_slug)
        meta.status = status
        meta.last_error = error
        if training is not None:
            meta.training = training
        self.save_meta(meta)
        return meta

    # -- helpers ------------------------------------------------------------
    def has_trained_weights(self, name_or_slug: str) -> bool:
        slug = self._coerce_slug(name_or_slug)
        weights = self.weights_dir(slug)
        return (weights / "config.json").exists() or any(
            weights.glob("*.bin")
        ) or any(weights.glob("*.safetensors"))

    def get_load_id(self, name_or_slug: str) -> str:
        """Return either a local path (if fine-tuned) or the HF base id."""

        meta = self.load_meta(name_or_slug)
        if self.has_trained_weights(meta.slug):
            return str(self.weights_dir(meta.slug))
        return BASE_MODELS[meta.base_model].hf_id

    # -- internal -----------------------------------------------------------
    def _coerce_slug(self, name_or_slug: str) -> str:
        """Accept either a slug or a human name and return the slug."""

        if not name_or_slug:
            raise ValueError("Empty model identifier.")
        if (self.root / name_or_slug).exists():
            return name_or_slug
        for meta in self.list_models():
            if meta.name == name_or_slug or meta.slug == name_or_slug:
                return meta.slug
        # Fall back to a freshly computed slug — useful for "does it exist?"
        return slugify(name_or_slug)

    def _unique_slug(self, name: str) -> str:
        base = slugify(name)
        slug = base
        i = 2
        while (self.root / slug).exists():
            slug = f"{base}-{i}"
            i += 1
        return slug
