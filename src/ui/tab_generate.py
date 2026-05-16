"""Tab 1 — Song generator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ..config import DEFAULT_GENERATION_PARAMS, GENRES
from ..generation.generator import GenerationRequest, LyricsGenerator
from ..generation.history import HistoryStore
from ..models.manager import ModelManager


def build_generate_tab(
    models: ModelManager,
    generator: LyricsGenerator,
    history_store: HistoryStore,
) -> None:
    """Build Gradio components for the Generator tab (called inside ``gr.Tab``)."""

    def _model_choices() -> List[str]:
        return models.list_names()

    def _valid_value(current: Optional[str], choices: List[str]) -> Optional[str]:
        """Return ``current`` if it is still in ``choices``; otherwise the first
        choice (or ``None`` when the list is empty). This keeps Gradio 4 from
        rejecting a stale dropdown value after the model list changes."""

        if current and current in choices:
            return current
        return choices[0] if choices else None

    # ---- header info ----
    gr.Markdown("Выберите модель, задайте название / тему и нажмите **Генерировать**.")

    with gr.Row():
        with gr.Column(scale=2):
            initial = _model_choices()
            model_dd = gr.Dropdown(
                label="Модель",
                choices=initial,
                value=initial[0] if initial else None,
                interactive=True,
            )
            refresh_btn = gr.Button("🔄 Обновить список", size="sm")
            title_in = gr.Textbox(label="Название песни", placeholder="Оставьте пустым для авто")
            theme_in = gr.Textbox(label="Тема / описание", lines=2, placeholder="О чём должна быть песня?")
            genre_dd = gr.Dropdown(label="Жанр", choices=GENRES, value="Auto")
            lang_dd = gr.Dropdown(
                label="Язык",
                choices=[("Авто", "auto"), ("Русский", "ru"), ("English", "en")],
                value="auto",
            )

        with gr.Column(scale=1):
            gr.Markdown("#### Параметры генерации")
            max_tokens = gr.Slider(64, 1024, value=DEFAULT_GENERATION_PARAMS["max_new_tokens"], step=16, label="Макс. токенов")
            temperature = gr.Slider(0.1, 2.0, value=DEFAULT_GENERATION_PARAMS["temperature"], step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=DEFAULT_GENERATION_PARAMS["top_p"], step=0.05, label="Top-p")
            top_k = gr.Slider(1, 200, value=DEFAULT_GENERATION_PARAMS["top_k"], step=1, label="Top-k")
            rep_penalty = gr.Slider(1.0, 2.0, value=DEFAULT_GENERATION_PARAMS["repetition_penalty"], step=0.05, label="Repetition penalty")
            seed_in = gr.Number(label="Seed (пусто = случайный)", precision=0, value=None)

    gen_btn = gr.Button("🎵 Генерировать", variant="primary", size="lg")

    # ---- output area ----
    with gr.Column(elem_classes="result-card"):
        result_title = gr.Markdown("### *(результат появится здесь)*")
        result_text = gr.Textbox(label="Текст песни", lines=16, interactive=False, show_copy_button=True)
        with gr.Row():
            download_btn = gr.Button("⬇ Скачать TXT")
            download_file = gr.File(visible=False, label="Файл")

    # ---- history ----
    with gr.Accordion("📜 История генераций", open=False):
        hist_dd = gr.Dropdown(label="Запись", choices=[], interactive=True)
        hist_text = gr.Textbox(label="Текст", interactive=False, lines=10)
        hist_refresh = gr.Button("Обновить историю", size="sm")

    # ---- callbacks ----
    def refresh_models(current: Optional[str]) -> Dict[str, Any]:
        choices = _model_choices()
        return gr.update(choices=choices, value=_valid_value(current, choices))

    refresh_btn.click(fn=refresh_models, inputs=model_dd, outputs=model_dd)

    def do_generate(
        model_name: str,
        title: str,
        theme: str,
        genre: str,
        language: str,
        max_tok: int,
        temp: float,
        tp: float,
        tk: int,
        rep: float,
        seed: Optional[float],
    ) -> Tuple[str, str]:
        if not model_name:
            return "### Ошибка", "Сначала создайте модель на вкладке «Модели»."
        req = GenerationRequest(
            model=model_name,
            title=title,
            theme=theme,
            genre=genre if genre != "Auto" else None,
            language=language,
            max_new_tokens=int(max_tok),
            temperature=temp,
            top_p=tp,
            top_k=int(tk),
            repetition_penalty=rep,
            seed=int(seed) if seed is not None and seed == seed else None,
        )
        try:
            result = generator.generate(req)
        except Exception as exc:
            return "### Ошибка генерации", str(exc)
        # persist to history
        meta = models.load_meta(model_name)
        history_store.add(meta.slug, result)
        header = f"### {result.title}"
        return header, result.text

    gen_btn.click(
        fn=do_generate,
        inputs=[model_dd, title_in, theme_in, genre_dd, lang_dd, max_tokens, temperature, top_p, top_k, rep_penalty, seed_in],
        outputs=[result_title, result_text],
    )

    def download_txt(title_md: str, text: str) -> Optional[str]:
        if not text or text.startswith("Сначала"):
            return None
        title = title_md.replace("### ", "").strip() or "song"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", prefix=f"{title[:30]}_", mode="w", encoding="utf-8"
        )
        tmp.write(f"{title}\n\n{text}")
        tmp.close()
        return tmp.name

    download_btn.click(
        fn=download_txt,
        inputs=[result_title, result_text],
        outputs=download_file,
    ).then(fn=lambda: gr.update(visible=True), outputs=download_file)

    def refresh_history(model_name: str) -> Dict[str, Any]:
        if not model_name:
            return gr.update(choices=[])
        meta = models.load_meta(model_name)
        items = history_store.list(meta.slug)
        choices = [f"{it.get('created_at', '')} — {it.get('title', '?')}" for it in items]
        return gr.update(choices=choices)

    hist_refresh.click(fn=refresh_history, inputs=model_dd, outputs=hist_dd)

    def show_history_item(model_name: str, selection: str) -> str:
        if not model_name or not selection:
            return ""
        meta = models.load_meta(model_name)
        items = history_store.list(meta.slug)
        idx_map = {
            f"{it.get('created_at', '')} — {it.get('title', '?')}": it for it in items
        }
        entry = idx_map.get(selection, {})
        return entry.get("text", "")

    hist_dd.change(fn=show_history_item, inputs=[model_dd, hist_dd], outputs=hist_text)
