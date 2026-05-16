"""Top-level entrypoint for Song Lyrics AI Studio.

Run locally::

    python app.py

Run in Google Colab — preferred (import directly, URL appears in the cell)::

    from app import launch
    launch()  # auto-detects Colab → enables share, disables browser

Or via the CLI (use ``-u`` so the URL is not buffered)::

    !python -u app.py --share --no-browser

Environment variables:

* ``LYRICS_AI_DATA``   – override the data directory (e.g. point at /content/drive/...).
* ``LYRICS_AI_PORT``   – override the local server port (default 7860).
* ``LYRICS_AI_HOST``   – override the bind host (default 0.0.0.0).
* ``LYRICS_AI_SHARE``  – set to ``1`` to force the public gradio.live tunnel.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

from src.config import APP_TITLE
from src.logger import get_logger
from src.ui import build_app

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def in_colab() -> bool:
    """Return ``True`` when running inside Google Colab."""

    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def in_notebook() -> bool:
    """Return ``True`` when running inside any Jupyter / IPython kernel."""

    try:
        from IPython import get_ipython  # type: ignore
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ in {
        "ZMQInteractiveShell",
        "Shell",          # Colab
        "GoogleColabShell",
    }


# ---------------------------------------------------------------------------
# Public API — called from notebooks
# ---------------------------------------------------------------------------

def launch(
    *,
    share: Optional[bool] = None,
    port: Optional[int] = None,
    host: Optional[str] = None,
    inbrowser: Optional[bool] = None,
    debug: bool = False,
    prevent_thread_lock: bool = False,
    **launch_kwargs: Any,
) -> Any:
    """Build and launch the Gradio app, with Colab-friendly defaults.

    When running inside Google Colab this automatically enables ``share=True``
    and disables ``inbrowser`` so the public URL is printed straight into the
    notebook cell output (no buffering, no opening a non-existent window).

    Returns the ``(app, local_url, share_url)`` tuple from ``gr.Blocks.launch``.
    """

    colab = in_colab()

    if share is None:
        share = colab or os.environ.get("LYRICS_AI_SHARE", "") == "1"
    if port is None:
        port = int(os.environ.get("LYRICS_AI_PORT", "7860"))
    if host is None:
        host = os.environ.get("LYRICS_AI_HOST", "0.0.0.0")
    if inbrowser is None:
        # Never try to open a browser inside a notebook / Colab kernel.
        inbrowser = not (colab or in_notebook())

    log.info(
        "Starting %s on %s:%s (share=%s, colab=%s)",
        APP_TITLE, host, port, share, colab,
    )

    app = build_app()
    return app.queue(max_size=16).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
        debug=debug,
        prevent_thread_lock=prevent_thread_lock,
        **launch_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("LYRICS_AI_SHARE", "") == "1",
        help="Expose a public *.gradio.live URL (auto-enabled in Colab).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LYRICS_AI_PORT", "7860")),
        help="Local server port.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("LYRICS_AI_HOST", "0.0.0.0"),
        help="Bind host.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser window.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose Gradio logs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    # Unbuffer stdout/stderr so the public Gradio URL prints immediately even
    # when this script is started as a child process from a Colab cell.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    args = _parse_args(argv)
    launch(
        share=args.share or in_colab(),
        port=args.port,
        host=args.host,
        inbrowser=not (args.no_browser or in_colab() or in_notebook()),
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
