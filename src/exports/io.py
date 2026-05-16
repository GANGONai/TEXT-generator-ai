"""ZIP-based import / export for user models.

Archive layout produced by :func:`export_model`::

    <slug>.zip
    ├── manifest.json          # version, slug, exported_at
    ├── config.json            # model metadata
    ├── songs.json             # the per-model dataset
    └── weights/               # HF weights + tokenizer (only when trained)
        ├── config.json
        ├── pytorch_model.bin  (or model.safetensors)
        ├── tokenizer.json
        └── ...

Both ``import_model`` and ``export_model`` are pure-Python and rely only
on the standard library, so they work in Colab without any extra deps.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import EXPORTS_DIR
from ..logger import get_logger
from ..models.manager import ModelManager, ModelMeta
from ..utils import now_iso, read_json, slugify, write_json

log = get_logger(__name__)

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(
    manager: ModelManager,
    name_or_slug: str,
    *,
    out_dir: Path = EXPORTS_DIR,
) -> Path:
    """Export a model directory to ``<out_dir>/<slug>.zip``."""

    meta = manager.load_meta(name_or_slug)
    src = manager.model_dir(meta.slug)
    if not src.exists():
        raise FileNotFoundError(f"Model directory missing: {src}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{meta.slug}.zip"
    if archive_path.exists():
        archive_path.unlink()

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "exported_at": now_iso(),
        "app": "song-lyrics-ai",
        "slug": meta.slug,
        "name": meta.name,
        "base_model": meta.base_model,
        "has_weights": manager.has_trained_weights(meta.slug),
    }

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in sorted(src.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            zf.write(path, arcname=str(rel))
    log.info("Exported model %s → %s", meta.slug, archive_path)
    return archive_path


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_model(
    manager: ModelManager,
    archive_path: Path,
    *,
    overwrite: bool = False,
    rename_to: Optional[str] = None,
) -> ModelMeta:
    """Import a previously exported archive."""

    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Файл не является ZIP-архивом.")

    with tempfile.TemporaryDirectory(prefix="lyrics-import-") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(archive_path, "r") as zf:
            _safe_extract(zf, tmp_dir)

        config_path = tmp_dir / "config.json"
        if not config_path.exists():
            raise ValueError("Архив не содержит config.json — несовместимый формат.")

        raw = read_json(config_path) or {}
        meta = ModelMeta.from_dict(raw)

        if rename_to:
            meta.name = rename_to.strip() or meta.name
            meta.slug = slugify(meta.name)

        target = manager.model_dir(meta.slug)
        if target.exists():
            if not overwrite:
                meta.slug = _unique_slug(manager, meta.slug)
                target = manager.model_dir(meta.slug)
            else:
                shutil.rmtree(target, ignore_errors=True)

        shutil.copytree(tmp_dir, target)

        # Re-write config with the (possibly) new slug/name and a fresh timestamp.
        meta.updated_at = now_iso()
        write_json(target / "config.json", meta.to_dict())
        if not (target / "songs.json").exists():
            write_json(target / "songs.json", [])

    log.info("Imported archive %s as model %s", archive_path, meta.slug)
    return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_slug(manager: ModelManager, slug: str) -> str:
    candidate = slug
    i = 2
    while manager.model_dir(candidate).exists():
        candidate = f"{slug}-{i}"
        i += 1
    return candidate


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip refusing path traversal."""

    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if dest not in target.parents and target != dest:
            raise ValueError(f"Опасный путь в архиве: {member.filename}")
    zf.extractall(dest)
