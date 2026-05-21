"""Top-level package for the ``fat_photo`` tool.

Convenience re-exports so callers can write::

    from fat_photo import fatten_photo, launch

instead of digging into ``fat_photo.src.pipeline`` / ``fat_photo.app``.
"""

from .src.pipeline import FattenResult, fatten_photo

try:
    # ``app`` pulls in gradio which is an optional UI dep. Make the re-export
    # best-effort so importing ``fat_photo`` from a script that doesn't need
    # the Gradio UI doesn't fail.
    from .app import build_ui, launch  # noqa: F401
except Exception:  # noqa: BLE001
    pass

__all__ = ["fatten_photo", "FattenResult", "launch", "build_ui"]
