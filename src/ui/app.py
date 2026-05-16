"""Assemble the three-tab Gradio application."""

from __future__ import annotations

import gradio as gr

from ..config import APP_DESCRIPTION, APP_TITLE
from ..generation.generator import LyricsGenerator
from ..generation.history import HistoryStore
from ..models.manager import ModelManager
from ..training.trainer import LyricsTrainer
from .tab_generate import build_generate_tab
from .tab_models import build_models_tab
from .tab_upload import build_upload_tab
from .theme import CUSTOM_CSS


def build_app() -> gr.Blocks:
    """Return a fully-wired ``gr.Blocks`` app ready for ``.launch()``."""

    model_mgr = ModelManager()
    generator = LyricsGenerator(model_mgr)
    history_store = HistoryStore()
    trainer = LyricsTrainer(model_mgr)

    with gr.Blocks(
        title=APP_TITLE,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            f"<div id='app-header'>"
            f"<h1>🎶 {APP_TITLE}</h1>"
            f"<p>{APP_DESCRIPTION}</p>"
            f"</div>"
        )

        with gr.Tabs():
            with gr.Tab("🎵 Генератор"):
                build_generate_tab(model_mgr, generator, history_store)
            with gr.Tab("📝 Загрузка текстов"):
                build_upload_tab(model_mgr, trainer)
            with gr.Tab("🤖 Модели"):
                build_models_tab(model_mgr)

        gr.Markdown(
            "<center style='opacity:0.5;margin-top:1rem'>"
            "Song Lyrics AI Studio · Free & Open-Source · "
            "<a href='https://github.com/GANGONai/TEXT-generator-ai'>GitHub</a>"
            "</center>"
        )

    return app
