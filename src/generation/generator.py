"""Lyrics generator with a small LRU cache of loaded models."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    # ---- "based on user text" controls ------------------------------------
    seed_text: str = ""                         # user's draft / starting lyrics
    ai_addition_pct: int = 100                  # tokens to add ≈ len(seed) * pct/100
    rhyme_pct: int = 0                          # target rhyme density (soft prompt hint)


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

        # 0% AI addition + non-empty seed text → return user's text untouched.
        if request.seed_text.strip() and request.ai_addition_pct <= 0:
            seed_body = request.seed_text.strip()
            title = (
                (request.title or "").strip()
                or self._guess_title(seed_body)
                or "Без названия"
            )
            return GenerationResult(
                title=title,
                text=seed_body,
                prompt="",
                model=request.model,
                params={
                    "max_new_tokens": 0,
                    "ai_addition_pct": 0,
                    "rhyme_pct_target": int(request.rhyme_pct),
                    "rhyme_pct_actual": _rhyme_rate(seed_body),
                    "language": request.language,
                    "genre": request.genre,
                    "note": "AI ничего не добавил (ползунок «доля AI» = 0%).",
                },
            )

        tokenizer, model = self._load(request.model)
        header, full_prompt = self._build_prompt(request)
        sampled, effective_max = self._sample(tokenizer, model, full_prompt, request)
        body = self._postprocess(sampled, header)
        title = (request.title or "").strip() or self._guess_title(body) or "Без названия"
        return GenerationResult(
            title=title,
            text=body,
            prompt=header,
            model=request.model,
            params={
                "max_new_tokens": effective_max,
                "ai_addition_pct": int(request.ai_addition_pct),
                "rhyme_pct_target": int(request.rhyme_pct),
                "rhyme_pct_actual": _rhyme_rate(body),
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "seed": request.seed,
                "genre": request.genre,
                "language": request.language,
                "had_seed_text": bool(request.seed_text.strip()),
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
    def _build_prompt(req: GenerationRequest) -> Tuple[str, str]:
        """Turn the request into ``(header, full_prompt)``.

        ``header`` is the scaffold (Title / Theme / ...) we want to strip from
        the model output later. ``full_prompt`` is what we actually feed the
        model — header + the user's seed text (if any), so the model continues
        it instead of inventing from scratch.
        """

        seed = req.seed_text.strip()
        lang = req.language
        if lang == "auto":
            sample = " ".join([req.title or "", req.theme or "", seed])
            lang = "ru" if _looks_russian(sample) else "en"

        lines: List[str] = []
        if lang == "ru":
            lines.append(f"Название: {req.title or '—'}")
            if req.genre and req.genre.lower() != "auto":
                lines.append(f"Жанр: {req.genre}")
            if req.theme:
                lines.append(f"Тема: {req.theme}")
            if req.rhyme_pct > 0:
                lines.append(
                    f"Стиль: рифмуй окончания строк (~{req.rhyme_pct}% строк "
                    "должны рифмоваться попарно)."
                )
            lines.append("Текст песни:")
        else:
            lines.append(f"Title: {req.title or '—'}")
            if req.genre and req.genre.lower() != "auto":
                lines.append(f"Genre: {req.genre}")
            if req.theme:
                lines.append(f"Theme: {req.theme}")
            if req.rhyme_pct > 0:
                lines.append(
                    f"Style: rhyme line endings (~{req.rhyme_pct}% of lines "
                    "should rhyme in pairs)."
                )
            lines.append("Lyrics:")
        header = "\n".join(lines) + "\n"

        # Continuation mode: paste the user's draft after the header so the
        # model extends it instead of starting from scratch.
        full = header + (seed + "\n" if seed else "")
        return header, full

    def _sample(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        req: GenerationRequest,
    ) -> Tuple[str, int]:
        import torch

        if req.seed is not None:
            torch.manual_seed(int(req.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(req.seed))

        device = next(model.parameters()).device

        # Scale max_new_tokens by the user's seed-text length and AI-addition %.
        seed_token_count = 0
        if req.seed_text.strip():
            seed_token_count = len(
                tokenizer.encode(req.seed_text, add_special_tokens=False)
            )
        if seed_token_count and req.ai_addition_pct > 0:
            scaled = int(seed_token_count * req.ai_addition_pct / 100)
            effective_max = max(16, min(int(req.max_new_tokens), scaled))
        else:
            effective_max = int(req.max_new_tokens)

        # Truncate input from the start if it would not fit alongside the
        # requested generation length. We keep the END of the user's draft so
        # the model continues from the most recent context.
        ctx_limit = getattr(model.config, "max_position_embeddings", 1024) or 1024
        max_input_len = max(64, int(ctx_limit) - effective_max - 8)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        gen_kwargs = {**DEFAULT_GENERATION_PARAMS}
        gen_kwargs.update(
            {
                "max_new_tokens": effective_max,
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
        return tokenizer.decode(output[0], skip_special_tokens=True), effective_max

    @staticmethod
    def _postprocess(generated: str, header: str) -> str:
        """Strip the scaffold header from the model output. The user's seed
        text — if any — is *kept* because it was pasted *after* the header."""

        text = generated
        if text.startswith(header):
            text = text[len(header):]
        # Strip a second scaffold echo if the model decided to start a new song.
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
# Language + rhyme heuristics
# ---------------------------------------------------------------------------

def _looks_russian(text: str) -> bool:
    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    lat = sum("a" <= ch.lower() <= "z" for ch in text)
    return cyr > lat


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def _line_ending(line: str, n: int = 3) -> str:
    """Return the last ``n`` letters of the last word of a line, lower-cased.

    Empty if the line has no letter words.
    """

    words = _WORD_RE.findall(line)
    if not words:
        return ""
    last = words[-1].lower()
    return last[-n:] if len(last) >= n else last


def _rhyme_rate(text: str) -> float:
    """Estimate, as a percentage, how many lines in ``text`` rhyme with a
    neighbour within ±2 lines (covers AABB and ABAB schemes).

    Deliberately simple heuristic — last-3-letters matching of the final word.
    Good enough to surface a rough number to the user, not a competition-grade
    rhyme detector.
    """

    endings = [_line_ending(line) for line in text.splitlines()]
    endings = [e for e in endings if e]
    n = len(endings)
    if n < 2:
        return 0.0
    rhyming = 0
    for i, e in enumerate(endings):
        neighbours = []
        if i + 1 < n:
            neighbours.append(endings[i + 1])
        if i + 2 < n:
            neighbours.append(endings[i + 2])
        if i - 1 >= 0:
            neighbours.append(endings[i - 1])
        if i - 2 >= 0:
            neighbours.append(endings[i - 2])
        if any(neigh and neigh == e for neigh in neighbours):
            rhyming += 1
    return round(rhyming / n * 100, 1)
