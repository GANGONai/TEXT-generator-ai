"""Tab 2 — Upload lyrics and manage per-model song datasets."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ..datasets.manager import SongManager
from ..models.manager import ModelManager
from ..training.trainer import LyricsTrainer


def build_upload_tab(models: ModelManager, trainer: LyricsTrainer) -> None:
    """Build Gradio components for the Upload tab (called inside ``gr.Tab``)."""

    def _model_choices() -> List[str]:
        return models.list_names()

    def _valid_value(current: Optional[str], choices: List[str]) -> Optional[str]:
        """Pick a still-valid dropdown value after the model list changes."""

        if current and current in choices:
            return current
        return choices[0] if choices else None

    def _song_choices(model_name: str) -> List[str]:
        if not model_name:
            return []
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        return [f"{s.id} | {s.title}" for s in sm.list()]

    def _song_count(model_name: str) -> str:
        if not model_name:
            return ""
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        cnt = sm.count()
        chars = sm.total_chars()
        return f"Песен: **{cnt}** | Символов: **{chars:,}**"

    # ---- layout ----
    with gr.Row():
        # LEFT: model + song list
        with gr.Column(scale=1):
            _initial = _model_choices()
            model_dd = gr.Dropdown(
                label="Модель",
                choices=_initial,
                value=_initial[0] if _initial else None,
                interactive=True,
            )
            refresh_btn = gr.Button("🔄 Обновить", size="sm")
            stats_md = gr.Markdown("")
            search_in = gr.Textbox(label="🔍 Поиск по песням", placeholder="Введите запрос…")
            song_list = gr.Dropdown(
                label="Песни",
                choices=[],
                interactive=True,
                elem_classes="song-list",
            )
            delete_song_btn = gr.Button("🗑 Удалить выбранную песню", variant="stop", size="sm")

        # RIGHT: edit / add
        with gr.Column(scale=2):
            song_title_in = gr.Textbox(label="Название песни", placeholder="Моя песня")
            song_genre_in = gr.Dropdown(label="Жанр (необязательно)", choices=[""] + [
                "Pop", "Rock", "Rap / Hip-Hop", "Folk", "Indie", "Metal",
                "R&B", "Country", "Electronic", "Шансон", "Авторская песня", "Романс"
            ], value="")
            song_text_in = gr.Textbox(label="Текст песни", lines=14, placeholder="Вставьте текст песни сюда…")

            with gr.Row():
                save_btn = gr.Button("💾 Сохранить песню", variant="primary")
                new_btn = gr.Button("📝 Новая песня")

            gr.Markdown("---")
            gr.Markdown("#### Массовый импорт")
            with gr.Row():
                file_upload = gr.File(label="Загрузить TXT / CSV", file_types=[".txt", ".csv"])
                import_btn = gr.Button("📥 Импортировать", size="sm")
            import_status = gr.Markdown("")

    # ---- training section ----
    gr.Markdown("---")
    gr.Markdown("### 🏋️ Обучение модели")
    gr.Markdown(
        "Обучение требует GPU. На бесплатном Colab включите его так: "
        "`Runtime → Change runtime type → Hardware accelerator → T4 GPU`, "
        "затем перезапустите ячейку с приложением."
    )
    with gr.Row():
        gpu_info_md = gr.Markdown(
            trainer.gpu_info().human_summary(),
            elem_classes="gpu-info",
        )
        gpu_check_btn = gr.Button("🔎 Проверить GPU", size="sm")
    with gr.Row():
        with gr.Column(scale=1):
            epochs_sl = gr.Slider(1, 20, value=3, step=1, label="Эпохи")
            lr_sl = gr.Slider(1e-6, 1e-3, value=5e-5, step=1e-6, label="Learning Rate")
            bs_sl = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
            block_sl = gr.Slider(64, 512, value=256, step=32, label="Block Size (tokens)")
        with gr.Column(scale=2):
            train_btn = gr.Button("🚀 Начать обучение", variant="primary", size="lg")
            cancel_btn = gr.Button("⛔ Остановить обучение", variant="stop", size="sm")
            training_status = gr.Textbox(
                label="Статус обучения",
                lines=6,
                interactive=False,
                elem_classes="progress-text",
            )
            training_progress = gr.Slider(
                0, 100, value=0, label="Прогресс (%)", interactive=False
            )
            poll_btn = gr.Button("🔄 Обновить статус", size="sm")

    def check_gpu() -> str:
        return trainer.gpu_info().human_summary()

    gpu_check_btn.click(fn=check_gpu, outputs=gpu_info_md)

    # ---- hidden state to track current editing song id ----
    current_song_id = gr.State(value=None)

    # ---- callbacks ----
    def refresh(model_name: str) -> Tuple[Any, Any, Any]:
        choices = _song_choices(model_name)
        stats = _song_count(model_name)
        model_choices = _model_choices()
        return (
            gr.update(choices=choices, value=None),
            stats,
            gr.update(choices=model_choices, value=_valid_value(model_name, model_choices)),
        )

    refresh_btn.click(fn=refresh, inputs=model_dd, outputs=[song_list, stats_md, model_dd])
    model_dd.change(fn=refresh, inputs=model_dd, outputs=[song_list, stats_md, model_dd])

    def search_songs(model_name: str, query: str) -> Any:
        if not model_name:
            return gr.update(choices=[])
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        results = sm.list(query=query)
        return gr.update(choices=[f"{s.id} | {s.title}" for s in results])

    search_in.change(fn=search_songs, inputs=[model_dd, search_in], outputs=song_list)

    def select_song(model_name: str, selection: str) -> Tuple[str, str, str, Optional[str]]:
        if not selection or not model_name:
            return "", "", "", None
        song_id = selection.split(" | ")[0].strip()
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        song = sm.get(song_id)
        if not song:
            return "", "", "", None
        return song.title, song.text, song.genre or "", song.id

    song_list.change(
        fn=select_song,
        inputs=[model_dd, song_list],
        outputs=[song_title_in, song_text_in, song_genre_in, current_song_id],
    )

    def save_song(
        model_name: str,
        title: str,
        text: str,
        genre: str,
        existing_id: Optional[str],
    ) -> Tuple[Any, str, Optional[str]]:
        if not model_name:
            return gr.update(), "Сначала выберите модель.", existing_id
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        try:
            if existing_id:
                sm.update(existing_id, title=title, text=text, genre=genre or None)
                msg = f"Песня «{title}» обновлена."
            else:
                song = sm.add(title=title, text=text, genre=genre or None)
                existing_id = song.id
                msg = f"Песня «{title}» сохранена."
        except (ValueError, KeyError) as exc:
            return gr.update(), str(exc), existing_id
        choices = [f"{s.id} | {s.title}" for s in sm.list()]
        return gr.update(choices=choices), msg, existing_id

    save_btn.click(
        fn=save_song,
        inputs=[model_dd, song_title_in, song_text_in, song_genre_in, current_song_id],
        outputs=[song_list, import_status, current_song_id],
    )

    def new_song() -> Tuple[str, str, str, None]:
        return "", "", "", None

    new_btn.click(fn=new_song, outputs=[song_title_in, song_text_in, song_genre_in, current_song_id])

    def delete_song(model_name: str, selection: str) -> Tuple[Any, str, str, str, None]:
        if not selection or not model_name:
            return gr.update(), "", "", "", None
        song_id = selection.split(" | ")[0].strip()
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        sm.delete(song_id)
        choices = [f"{s.id} | {s.title}" for s in sm.list()]
        return gr.update(choices=choices, value=None), "", "", "", None

    delete_song_btn.click(
        fn=delete_song,
        inputs=[model_dd, song_list],
        outputs=[song_list, song_title_in, song_text_in, song_genre_in, current_song_id],
    )

    def do_import(model_name: str, file_obj: Any) -> Tuple[Any, str]:
        if not model_name:
            return gr.update(), "Сначала выберите модель."
        if file_obj is None:
            return gr.update(), "Файл не загружен."
        meta = models.load_meta(model_name)
        sm = SongManager(models.songs_path(meta.slug))
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        raw = open(path, "r", encoding="utf-8", errors="replace").read()
        try:
            if path.endswith(".csv"):
                count = sm.import_from_csv(raw)
            else:
                count = sm.import_from_txt(raw)
        except Exception as exc:
            return gr.update(), f"Ошибка импорта: {exc}"
        choices = [f"{s.id} | {s.title}" for s in sm.list()]
        return gr.update(choices=choices), f"Импортировано песен: {count}"

    import_btn.click(fn=do_import, inputs=[model_dd, file_upload], outputs=[song_list, import_status])

    # ---- training ----
    def start_training(model_name: str, epochs: int, lr: float, bs: int, block: int) -> Tuple[str, float]:
        if not model_name:
            return "Сначала выберите модель.", 0
        try:
            state = trainer.start(
                model_name,
                params={
                    "epochs": int(epochs),
                    "learning_rate": float(lr),
                    "batch_size": int(bs),
                    "block_size": int(block),
                },
            )
            return state.human_summary(), state.percent
        except (RuntimeError, ValueError) as exc:
            return str(exc), 0

    train_btn.click(
        fn=start_training,
        inputs=[model_dd, epochs_sl, lr_sl, bs_sl, block_sl],
        outputs=[training_status, training_progress],
    )

    def poll_training(model_name: str) -> Tuple[str, float]:
        if not model_name:
            return "", 0
        meta = models.load_meta(model_name)
        state = trainer.get_state(meta.slug)
        return state.human_summary(), state.percent

    poll_btn.click(fn=poll_training, inputs=model_dd, outputs=[training_status, training_progress])

    def cancel_training(model_name: str) -> str:
        if not model_name:
            return ""
        meta = models.load_meta(model_name)
        trainer.cancel(meta.slug)
        return "Обучение останавливается…"

    cancel_btn.click(fn=cancel_training, inputs=model_dd, outputs=training_status)
