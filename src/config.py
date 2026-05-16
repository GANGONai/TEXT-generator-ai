"""Project-wide configuration: paths, defaults, base-model registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

# Root of the project (the directory that contains the `src/` package).
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# All runtime data lives under DATA_DIR.  It can be overridden with the
# LYRICS_AI_DATA env var so that users on Colab can point it at a Drive mount.
DATA_DIR: Path = Path(os.environ.get("LYRICS_AI_DATA", ROOT_DIR / "data"))

MODELS_DIR: Path = DATA_DIR / "models"        # one sub-folder per user model
HISTORY_DIR: Path = DATA_DIR / "history"      # generation history per model
EXPORTS_DIR: Path = DATA_DIR / "exports"      # generated zip archives
CACHE_DIR: Path = DATA_DIR / "cache"          # HF cache when running on Colab
LOGS_DIR: Path = DATA_DIR / "logs"            # rotating logs

for _p in (MODELS_DIR, HISTORY_DIR, EXPORTS_DIR, CACHE_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Base-model registry — only free / open Hugging Face checkpoints.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseModelSpec:
    """Metadata about a Hugging Face base model used for fine-tuning."""

    hf_id: str
    label: str
    language: str   # "en", "ru", or "multi"
    size_mb: int    # approximate disk footprint of weights
    notes: str = ""


BASE_MODELS: Dict[str, BaseModelSpec] = {
    "distilgpt2": BaseModelSpec(
        hf_id="distilgpt2",
        label="DistilGPT-2 (English, ~330 MB) — fastest",
        language="en",
        size_mb=330,
        notes="Recommended for Colab free tier.",
    ),
    "gpt2": BaseModelSpec(
        hf_id="gpt2",
        label="GPT-2 small (English, ~500 MB)",
        language="en",
        size_mb=500,
    ),
    "gpt2-medium": BaseModelSpec(
        hf_id="gpt2-medium",
        label="GPT-2 medium (English, ~1.5 GB) — needs GPU",
        language="en",
        size_mb=1500,
        notes="Requires GPU runtime in Colab.",
    ),
    "rugpt3-small": BaseModelSpec(
        hf_id="ai-forever/rugpt3small_based_on_gpt2",
        label="ruGPT-3 small (Russian, ~500 MB)",
        language="ru",
        size_mb=500,
        notes="Best baseline for Russian lyrics.",
    ),
    "tinyllama-1.1b": BaseModelSpec(
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        label="TinyLlama 1.1B (multilingual, ~2.2 GB) — needs GPU",
        language="multi",
        size_mb=2200,
        notes="Strong quality, needs GPU + fp16.",
    ),
}

DEFAULT_BASE_MODEL: str = "distilgpt2"


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------

DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 256,
    "temperature": 0.95,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "do_sample": True,
}


# ---------------------------------------------------------------------------
# Training defaults — conservative so it fits on Colab T4 (16 GB).
# ---------------------------------------------------------------------------

DEFAULT_TRAINING_PARAMS = {
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "block_size": 256,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "fp16": True,                       # auto-disabled on CPU
    "gradient_checkpointing": True,
    "save_total_limit": 1,
}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

GENRES = [
    "Auto",
    "Pop",
    "Rock",
    "Rap / Hip-Hop",
    "Folk",
    "Indie",
    "Metal",
    "R&B",
    "Country",
    "Electronic",
    "Шансон",
    "Авторская песня",
    "Романс",
]

APP_TITLE = "Song Lyrics AI Studio"
APP_DESCRIPTION = (
    "Fine-tune open-source language models on your own song lyrics and "
    "generate new songs in the same style. Free, local, and Colab-ready."
)
