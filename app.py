"""Top-level entrypoint for Song Lyrics AI Studio.

Run locally::

    python app.py

Run in Google Colab::

    !python app.py --share

Environment variables:

* ``LYRICS_AI_DATA``   – override the data directory (e.g. point at /content/drive/...).
* ``LYRICS_AI_PORT``   – override the local server port (default 7860).
* ``LYRICS_AI_SHARE``  – set to ``1`` to force the public gradio.live tunnel.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.config import APP_TITLE
from src.logger import get_logger
from src.ui import build_app

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("LYRICS_AI_SHARE", "") == "1",
        help="Expose a public *.gradio.live URL (use in Colab).",
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    log.info("Starting %s on %s:%s (share=%s)", APP_TITLE, args.host, args.port, args.share)

    app = build_app()
    app.queue(max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
