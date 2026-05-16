"""Tab 3 — Model management (CRUD, import, export)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ..config import BASE_MODELS
from ..exports.io import export_model, import_model
from ..models.manager import ModelManager


def build_models_tab(models: ModelManager) -> None:
    """Build Gradio components for the Models tab (called inside ``gr.Tab``)."""

    STATUS_ICONS = {
        "untrained": "⚪ не обучена",
        "training": "🟡 обучается",
        "trained": "🟢 обучена",
        "error": "🔴 ошибка",
    }

    def _table_data() -> List[List[str]]:
        rows = []
        for m in models.list_models():
            rows.append([
                m.name,
                m.base_model,
                STATUS_ICONS.get(m.status, m.status),
                m.updated_at[:19].replace("T", " "),
                m.slug,
            ])
        return rows or [["—", "—", "—", "—", "—"]]

    def _names() -> List[str]:
        return models.list_names()

    def _valid_value(current: Optional[str], choices: List[str]) -> Optional[str]:
        """Pick a still-valid dropdown value after the model list changes."""

        if current and current in choices:
            return current
        return choices[0] if choices else None

    # ---- layout ----
    gr.Markdown("Управление моделями: создание, удаление, импорт и экспорт.")

    model_table = gr.Dataframe(
        headers=["Название", "Базовая модель", "Статус", "Обновлено", "slug"],
        value=_table_data,
        interactive=False,
        wrap=True,
    )
    refresh_btn = gr.Button("🔄 Обновить таблицу", size="sm")

    gr.Markdown("---")

    with gr.Row():
        # CREATE
        with gr.Column():
            gr.Markdown("#### ➕ Создать модель")
            new_name = gr.Textbox(label="Название", placeholder="Моя модель")
            base_dd = gr.Dropdown(
                label="Базовая модель",
                choices=[(spec.label, key) for key, spec in BASE_MODELS.items()],
                value="distilgpt2",
            )
            create_btn = gr.Button("Создать", variant="primary")
            create_status = gr.Markdown("")

        # DELETE
        with gr.Column():
            gr.Markdown("#### 🗑 Удалить модель")
            _initial = _names()
            del_dd = gr.Dropdown(
                label="Модель",
                choices=_initial,
                value=_initial[0] if _initial else None,
                interactive=True,
            )
            del_btn = gr.Button("Удалить", variant="stop")
            del_status = gr.Markdown("")

    gr.Markdown("---")

    with gr.Row():
        # EXPORT
        with gr.Column():
            gr.Markdown("#### 📤 Экспорт модели (ZIP)")
            _initial = _names()
            exp_dd = gr.Dropdown(
                label="Модель",
                choices=_initial,
                value=_initial[0] if _initial else None,
                interactive=True,
            )
            exp_btn = gr.Button("Экспортировать")
            exp_file = gr.File(label="Архив", visible=False)
            exp_status = gr.Markdown("")

        # IMPORT
        with gr.Column():
            gr.Markdown("#### 📥 Импорт модели (ZIP)")
            imp_file = gr.File(label="Загрузить ZIP-архив", file_types=[".zip"])
            imp_rename = gr.Textbox(label="Переименовать (необязательно)", placeholder="Новое имя")
            imp_overwrite = gr.Checkbox(label="Перезаписать, если существует")
            imp_btn = gr.Button("Импортировать")
            imp_status = gr.Markdown("")

    # ---- callbacks ----
    def refresh_all(current_del: Optional[str], current_exp: Optional[str]) -> Tuple[Any, Any, Any]:
        names = _names()
        return (
            _table_data(),
            gr.update(choices=names, value=_valid_value(current_del, names)),
            gr.update(choices=names, value=_valid_value(current_exp, names)),
        )

    refresh_btn.click(
        fn=refresh_all,
        inputs=[del_dd, exp_dd],
        outputs=[model_table, del_dd, exp_dd],
    )

    def create_model(name: str, base: str, current_del: Optional[str], current_exp: Optional[str]) -> Tuple[Any, str, Any, Any]:
        if not name or not name.strip():
            return _table_data(), "Введите название модели.", gr.update(), gr.update()
        try:
            meta = models.create(name.strip(), base)
        except ValueError as exc:
            return _table_data(), str(exc), gr.update(), gr.update()
        names = _names()
        # Newly created model is now in the list — select it if there was no
        # previous selection.
        return (
            _table_data(),
            f"Модель «{meta.name}» создана.",
            gr.update(choices=names, value=_valid_value(current_del or meta.name, names)),
            gr.update(choices=names, value=_valid_value(current_exp or meta.name, names)),
        )

    create_btn.click(
        fn=create_model,
        inputs=[new_name, base_dd, del_dd, exp_dd],
        outputs=[model_table, create_status, del_dd, exp_dd],
    )

    def delete_model(name: str, current_exp: Optional[str]) -> Tuple[Any, str, Any, Any]:
        if not name:
            return _table_data(), "Выберите модель.", gr.update(), gr.update()
        models.delete(name)
        names = _names()
        return (
            _table_data(),
            f"Модель «{name}» удалена.",
            gr.update(choices=names, value=_valid_value(None, names)),
            gr.update(choices=names, value=_valid_value(current_exp if current_exp != name else None, names)),
        )

    del_btn.click(
        fn=delete_model,
        inputs=[del_dd, exp_dd],
        outputs=[model_table, del_status, del_dd, exp_dd],
    )

    def do_export(name: str) -> Tuple[Optional[str], str]:
        if not name:
            return None, "Выберите модель."
        try:
            path = export_model(models, name)
            return str(path), f"Экспорт готов: `{path.name}`"
        except Exception as exc:
            return None, f"Ошибка: {exc}"

    exp_btn.click(fn=do_export, inputs=exp_dd, outputs=[exp_file, exp_status]).then(
        fn=lambda p: gr.update(visible=p is not None), inputs=exp_file, outputs=exp_file
    )

    def do_import(
        file_obj: Any,
        rename: str,
        overwrite: bool,
        current_del: Optional[str],
        current_exp: Optional[str],
    ) -> Tuple[Any, str, Any, Any]:
        if file_obj is None:
            return _table_data(), "Загрузите ZIP-файл.", gr.update(), gr.update()
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        try:
            meta = import_model(
                models,
                path,
                overwrite=overwrite,
                rename_to=rename.strip() or None,
            )
        except Exception as exc:
            return _table_data(), f"Ошибка: {exc}", gr.update(), gr.update()
        names = _names()
        return (
            _table_data(),
            f"Модель «{meta.name}» импортирована.",
            gr.update(choices=names, value=_valid_value(current_del or meta.name, names)),
            gr.update(choices=names, value=_valid_value(current_exp or meta.name, names)),
        )

    imp_btn.click(
        fn=do_import,
        inputs=[imp_file, imp_rename, imp_overwrite, del_dd, exp_dd],
        outputs=[model_table, imp_status, del_dd, exp_dd],
    )
