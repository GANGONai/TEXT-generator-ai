"""Gradio app: upload a photo, pick "intensity", get a fattened version back.

Designed to run on a free Google Colab T4 GPU. On Colab the app is launched
with ``share=True`` so the user gets a public ``*.gradio.live`` URL.

The app has two modes:
- **Fast (default)**: geometric warp only — instant, no GPU required.
- **AI-улучшение**: also runs Stable Diffusion img2img on top of the warp
  to add realistic skin folds and remove warp artefacts. Requires ~1 min on
  a T4 GPU; the model is downloaded once (~2 GB) on first use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# When run from inside the repo, make ``fat_photo`` importable as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import gradio as gr  # noqa: E402

from fat_photo.src.pipeline import fatten_photo  # noqa: E402


DESCRIPTION = """
# Толстожоп 9000 — AI-приложение для добавления веса на фото

Загрузи фото человека и крути ползунок **«Сколько добавить кг»** —
приложение деформирует силуэт и опционально дорисовывает реалистичную
текстуру через Stable Diffusion.

- **Быстрый режим** — мгновенно, без GPU. Геометрическая деформация
  лица и тела по ключевым точкам MediaPipe.
- **AI-улучшение** — поверх деформации запускается Stable Diffusion
  img2img c промптом «obese, overweight…». Нужен GPU (Colab T4 — ок).
""".strip()


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _process(image, intensity, use_ai, ai_strength, ai_steps, seed):
    if image is None:
        return None, "Загрузи фото."
    seed_val = int(seed) if seed and int(seed) > 0 else None
    result = fatten_photo(
        image,
        intensity=float(intensity) / 100.0,
        use_ai_refine=bool(use_ai),
        ai_strength=float(ai_strength),
        ai_steps=int(ai_steps),
        seed=seed_val,
    )
    return result.image, result.message


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Толстожоп 9000") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    label="Исходное фото",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                )
                intensity = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Сколько добавить (0–100%)",
                )
                use_ai = gr.Checkbox(
                    value=False,
                    label="AI-улучшение (Stable Diffusion) — нужен GPU",
                )
                with gr.Accordion("Дополнительные параметры AI", open=False):
                    ai_strength = gr.Slider(
                        minimum=0.05,
                        maximum=0.7,
                        value=0.25,
                        step=0.05,
                        label="Сила диффузии (низкая = аккуратнее)",
                    )
                    ai_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Шаги диффузии",
                    )
                    seed = gr.Number(value=0, label="Seed (0 = случайный)", precision=0)

                btn = gr.Button("Сделать толстожопом 💪", variant="primary")

            with gr.Column(scale=1):
                out = gr.Image(label="Результат", type="pil")
                status = gr.Textbox(label="Статус", interactive=False, lines=2)

        btn.click(
            _process,
            inputs=[inp, intensity, use_ai, ai_strength, ai_steps, seed],
            outputs=[out, status],
        )

    return demo


def launch(share: bool | None = None, server_port: int | None = None) -> None:
    demo = build_ui()
    if share is None:
        share = _is_colab()
    kwargs: dict = {"share": share}
    if server_port is not None:
        kwargs["server_port"] = server_port
        kwargs["server_name"] = "0.0.0.0"
    demo.queue(max_size=8).launch(**kwargs)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Launch the fat_photo Gradio app.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Force public gradio.live URL (default: on in Colab).",
    )
    parser.add_argument("--port", type=int, default=None, help="Local server port.")
    args = parser.parse_args()

    share = True if args.share else (True if _is_colab() else False)
    launch(share=share, server_port=args.port)


if __name__ == "__main__":
    _main()


# Make ``python -c "from fat_photo.app import launch; launch()"`` work too.
__all__ = ["launch", "build_ui"]

# Allow ``HF_HOME`` redirection on Colab to ``/content`` so weights survive
# between cell reruns within the same session.
if _is_colab() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/content/.cache/huggingface"
