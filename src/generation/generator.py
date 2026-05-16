"""Lyrics generator with a small LRU cache of loaded models."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..config import DEFAULT_GENERATION_PARAMS
from ..logger import get_logger
from ..models.manager import ModelManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """Single generation call parameters."""

    model: str                                  # user-model name or slug
    title: str = ""
    theme: str = ""
    genre: Optional[str] = None
    language: str = "auto"                      # "auto" | "ru" | "en"
    max_new_tokens: int = 256
    temperature: float = 0.95
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.15
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Output of a single generation call."""

    title: str
    text: str
    prompt: str
    model: str
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LyricsGenerator:
    """Loads HF models on demand (with LRU cache) and runs sampling."""

    _MAX_CACHED = 2  # protect VRAM — keep at most two models warm

    def __init__(self, model_manager: ModelManager) -> None:
        self.models = model_manager
        self._cache: "OrderedDict[str, Tuple[Any, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a single generation call and return the result."""

        tokenizer, model = self._load(request.model)
        prompt = self._build_prompt(request)
        text = self._sample(tokenizer, model, prompt, request)
        title = (request.title or "").strip() or self._guess_title(text) or "Без названия"
        return GenerationResult(
            title=title,
            text=self._postprocess(text, prompt),
            prompt=prompt,
            model=request.model,
            params={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "seed": request.seed,
                "genre": request.genre,
                "language": request.language,
            },
        )

    def evict(self, name_or_slug: Optional[str] = None) -> None:
        """Drop a single cached model or the whole cache."""

        with self._lock:
            if name_or_slug is None:
                self._cache.clear()
            else:
                key = self._cache_key_for(name_or_slug)
                self._cache.pop(key, None)
        self._gc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cache_key_for(self, name_or_slug: str) -> str:
        meta = self.models.load_meta(name_or_slug)
        return f"{meta.slug}:{self.models.get_load_id(meta.slug)}"

    def _load(self, name_or_slug: str) -> Tuple[Any, Any]:
        # Deferred imports — keep module import cheap.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        key = self._cache_key_for(name_or_slug)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        load_id = self.models.get_load_id(name_or_slug)
        log.info("Loading model %s from %s", name_or_slug, load_id)

        tokenizer = AutoTokenizer.from_pretrained(load_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(load_id, torch_dtype=dtype)
        except (OSError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(load_id)
        model.to(device)
        model.eval()

        with self._lock:
            self._cache[key] = (tokenizer, model)
            while len(self._cache) > self._MAX_CACHED:
                old_key, _ = self._cache.popitem(last=False)
                log.info("Evicting cached model %s", old_key)
        self._gc()
        return tokenizer, model

    @staticmethod
    def _build_prompt(req: GenerationRequest) -> str:
        """Turn structured request fields into a single text prompt."""

        lang = req.language
        if lang == "auto":
            lang = "ru" if _looks_russian(req.title + " " + req.theme) else "en"
        lines = []
        if lang == "ru":
            lines.append(f"Название: {req.title or '—'}")
            if req.genre and req.genre.lower() != "auto":
                lines.append(f"Жанр: {req.genre}")
            if req.theme:
                lines.append(f"Тема: {req.theme}")
            lines.append("Текст песни:")
        else:
            lines.append(f"Title: {req.title or '—'}")
            if req.genre and req.genre.lower() != "auto":
                lines.append(f"Genre: {req.genre}")
            if req.theme:
                lines.append(f"Theme: {req.theme}")
            lines.append("Lyrics:")
        return "\n".join(lines) + "\n"

    def _sample(self, tokenizer: Any, model: Any, prompt: str, req: GenerationRequest) -> str:
        import torch

        if req.seed is not None:
            torch.manual_seed(int(req.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(req.seed))

        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        gen_kwargs = {**DEFAULT_GENERATION_PARAMS}
        gen_kwargs.update(
            {
                "max_new_tokens": int(req.max_new_tokens),
                "temperature": float(req.temperature),
                "top_p": float(req.top_p),
                "top_k": int(req.top_k),
                "repetition_penalty": float(req.repetition_penalty),
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    @staticmethod
    def _postprocess(generated: str, prompt: str) -> str:
        text = generated
        if text.startswith(prompt):
            text = text[len(prompt):]
        # Strip trailing repetitions of the prompt scaffold if the model echoed it.
        for marker in ("Title:", "Название:"):
            idx = text.find(marker, 1)
            if idx > 0:
                text = text[:idx]
                break
        return text.strip()

    @staticmethod
    def _guess_title(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if 3 <= len(stripped) <= 60 and not stripped.lower().startswith(("title:", "lyrics:", "название:", "текст")):
                return stripped
        return ""

    @staticmethod
    def _gc() -> None:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def _looks_russian(text: str) -> bool:
    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    lat = sum("a" <= ch.lower() <= "z" for ch in text)
    return cyr > lat
