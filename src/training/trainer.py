"""Fine-tuning loop for song-lyrics models.

The trainer is intentionally written from scratch (no HF ``Trainer``) so
that we can:

* surface per-step progress + ETA to the Gradio UI,
* recover from CUDA OOM by halving the batch size at runtime,
* fall back gracefully to CPU when no GPU is available.

It runs in a background thread so the UI stays responsive.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BASE_MODELS, DEFAULT_TRAINING_PARAMS
from ..datasets.manager import SongManager
from ..logger import get_logger
from ..models.manager import ModelManager
from ..utils import human_duration, now_iso
from .device import GPUInfo, get_gpu_info, require_gpu

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# State exposed to the UI
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Snapshot of training progress for one model.

    All attributes are JSON-serialisable so they can be persisted into the
    model's ``config.json`` between runs.
    """

    status: str = "idle"          # idle | preparing | running | done | error | cancelled
    message: str = ""
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    device: str = "cpu"
    fp16: bool = False
    batch_size: int = 0
    samples: int = 0
    error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def percent(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, 100.0 * self.step / self.total_steps)

    def human_summary(self) -> str:
        if self.status == "idle":
            return "Обучение ещё не запускалось."
        if self.status == "error":
            return f"Ошибка: {self.error or self.message}"
        if self.status in ("done",):
            return (
                f"Готово. Эпох: {self.total_epochs}, шагов: {self.step}, "
                f"финальный loss: {self.loss:.4f}, время: {human_duration(self.elapsed_seconds)}"
            )
        eta = human_duration(self.eta_seconds) if self.eta_seconds else "—"
        return (
            f"{self.status} • эпоха {self.epoch}/{self.total_epochs} • "
            f"шаг {self.step}/{self.total_steps} • loss {self.loss:.4f} • "
            f"ETA {eta} • устройство {self.device}{' (fp16)' if self.fp16 else ''}"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LyricsTrainer:
    """Fine-tune a base HF model on a user's song dataset, in a background thread."""

    # We keep a singleton-per-process state map so the UI can poll progress
    # for any model even if the trainer object goes out of scope.
    _states: Dict[str, TrainingState] = {}
    _threads: Dict[str, threading.Thread] = {}
    _stop_flags: Dict[str, threading.Event] = {}
    _lock = threading.Lock()

    def __init__(self, model_manager: ModelManager) -> None:
        self.models = model_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def get_state(cls, slug: str) -> TrainingState:
        with cls._lock:
            return cls._states.get(slug) or TrainingState()

    @classmethod
    def is_running(cls, slug: str) -> bool:
        thread = cls._threads.get(slug)
        return thread is not None and thread.is_alive()

    @classmethod
    def cancel(cls, slug: str) -> bool:
        flag = cls._stop_flags.get(slug)
        if flag is None:
            return False
        flag.set()
        return True

    @classmethod
    def gpu_info(cls) -> GPUInfo:
        """Convenience wrapper around :func:`get_gpu_info` for the UI."""

        return get_gpu_info()

    def start(
        self,
        model_name_or_slug: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> TrainingState:
        """Kick off training in a background thread.

        Raises ``RuntimeError`` immediately if no CUDA GPU is available on the
        host — we refuse to train on CPU because it is unusably slow.
        """

        # Fail fast: no GPU → no training. The error message guides the user
        # to enable a GPU runtime in Colab.
        info = require_gpu()

        meta = self.models.load_meta(model_name_or_slug)
        slug = meta.slug

        if self.is_running(slug):
            raise RuntimeError("Эта модель уже обучается.")

        songs = SongManager(self.models.songs_path(slug)).list()
        if not songs:
            raise ValueError("Нет текстов для обучения. Сначала добавьте песни.")

        merged_params = {**DEFAULT_TRAINING_PARAMS, **(params or {})}

        state = TrainingState(
            status="preparing",
            message=f"GPU OK: {info.name} — подготовка данных…",
            total_epochs=int(merged_params["epochs"]),
            samples=len(songs),
            batch_size=int(merged_params["batch_size"]),
            device="cuda",
            fp16=bool(merged_params.get("fp16", True)),
            started_at=now_iso(),
        )
        with self._lock:
            self._states[slug] = state
            self._stop_flags[slug] = threading.Event()

        self.models.update_status(slug, "training", training=state.to_dict())

        thread = threading.Thread(
            target=self._run,
            args=(slug, meta.base_model, merged_params),
            daemon=True,
            name=f"trainer-{slug}",
        )
        with self._lock:
            self._threads[slug] = thread
        thread.start()
        return state

    # ------------------------------------------------------------------
    # Internal training routine
    # ------------------------------------------------------------------
    def _run(self, slug: str, base_model_key: str, params: Dict[str, Any]) -> None:
        try:
            self._train(slug, base_model_key, params)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Training crashed for %s", slug)
            self._fail(slug, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def _train(self, slug: str, base_model_key: str, params: Dict[str, Any]) -> None:
        # Imports are deferred so importing the manager does not require torch.
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

        # Hard requirement: we already checked in start(), but re-check here
        # in case the thread was paused long enough for CUDA to disappear.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA исчезла между стартом и началом обучения. "
                "Перезапустите runtime и включите GPU."
            )

        device = "cuda"
        fp16 = bool(params.get("fp16", True))

        hf_id = BASE_MODELS[base_model_key].hf_id
        self._update_state(
            slug,
            status="preparing",
            message=f"Скачиваю токенизатор «{hf_id}»…",
            device=device,
            fp16=fp16,
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._update_state(
            slug,
            status="preparing",
            message=f"Скачиваю веса «{hf_id}» (~50–500 МБ, может занять минуту)…",
        )
        model = AutoModelForCausalLM.from_pretrained(hf_id)
        if params.get("gradient_checkpointing"):
            try:
                model.gradient_checkpointing_enable()
                model.config.use_cache = False
            except Exception:  # pragma: no cover - some models reject it
                pass
        self._update_state(slug, message="Переношу модель на GPU…")
        model.to(device)
        model.train()

        # --- build dataset -------------------------------------------------
        self._update_state(slug, message="Готовлю датасет (токенизация)…")
        songs = SongManager(self.models.songs_path(slug)).list()
        block_size = int(params["block_size"])
        eos = tokenizer.eos_token or ""
        joined = ("\n\n" + eos + "\n\n").join(
            f"Title: {s.title}\n{('Genre: ' + s.genre + chr(10)) if s.genre else ''}\n{s.text}"
            for s in songs
        )
        encoded = tokenizer(joined, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks: List[List[int]] = []
        for i in range(0, max(1, encoded.size(0) - 1), block_size):
            chunk = encoded[i : i + block_size]
            if chunk.numel() < 8:
                continue
            chunks.append(chunk.tolist())
        if not chunks:
            self._fail(slug, "Слишком мало текста для обучения (минимум ~8 токенов).")
            return

        class _Ds(Dataset):
            def __init__(self, items: List[List[int]]) -> None:
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
                ids = torch.tensor(self.items[idx], dtype=torch.long)
                return {"input_ids": ids, "labels": ids.clone()}

        def _collate(batch: List[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
            max_len = max(b["input_ids"].size(0) for b in batch)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            for i, b in enumerate(batch):
                n = b["input_ids"].size(0)
                input_ids[i, :n] = b["input_ids"]
                labels[i, :n] = b["labels"]
            return {"input_ids": input_ids.to(device), "labels": labels.to(device)}

        # --- training loop with OOM-aware batch sizing ---------------------
        batch_size = int(params["batch_size"])
        ds = _Ds(chunks)

        while batch_size >= 1:
            try:
                self._do_loop(
                    slug=slug,
                    model=model,
                    tokenizer=tokenizer,
                    ds=ds,
                    collate=_collate,
                    params=params,
                    batch_size=batch_size,
                    fp16=fp16,
                    device=device,
                )
                break
            except _OOM:
                batch_size = max(1, batch_size // 2)
                grad_accum = int(params.get("gradient_accumulation_steps", 1)) * 2
                params = {**params, "gradient_accumulation_steps": grad_accum}
                msg = f"OOM, понижаю batch_size до {batch_size}, grad_accum={grad_accum}"
                log.warning(msg)
                self._update_state(slug, message=msg, batch_size=batch_size)
                import gc
                gc.collect()
                if device == "cuda":
                    import torch as _t
                    _t.cuda.empty_cache()
        else:
            self._fail(slug, "Не хватает VRAM даже при batch_size=1.")
            return

        # --- save ---------------------------------------------------------
        weights_dir = self.models.weights_dir(slug)
        weights_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(weights_dir)
        tokenizer.save_pretrained(weights_dir)
        log.info("Saved fine-tuned weights to %s", weights_dir)

        final = self._update_state(
            slug,
            status="done",
            message="Обучение завершено.",
            finished_at=now_iso(),
        )
        self.models.update_status(slug, "trained", training=final.to_dict())

    # ------------------------------------------------------------------
    def _do_loop(
        self,
        *,
        slug: str,
        model: Any,
        tokenizer: Any,
        ds: Any,
        collate: Any,
        params: Dict[str, Any],
        batch_size: int,
        fp16: bool,
        device: str,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        stop_flag = self._stop_flags[slug]
        epochs = int(params["epochs"])
        lr = float(params["learning_rate"])
        warmup = int(params["warmup_steps"])
        grad_accum = max(1, int(params.get("gradient_accumulation_steps", 1)))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
        total_steps = steps_per_epoch * epochs

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(params["weight_decay"])
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
        )
        scaler = torch.cuda.amp.GradScaler() if fp16 else None

        self._update_state(
            slug,
            status="running",
            message="Старт обучения",
            total_steps=total_steps,
            batch_size=batch_size,
        )

        start = time.time()
        global_step = 0
        optimizer.zero_grad()
        for epoch in range(1, epochs + 1):
            for micro_step, batch in enumerate(loader):
                if stop_flag.is_set():
                    self._update_state(slug, status="cancelled", message="Обучение остановлено.")
                    return
                try:
                    if fp16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            out = model(**batch)
                            loss = out.loss / grad_accum
                        scaler.scale(loss).backward()
                    else:
                        out = model(**batch)
                        loss = out.loss / grad_accum
                        loss.backward()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise _OOM() from exc
                    raise

                if (micro_step + 1) % grad_accum == 0 or (micro_step + 1) == len(loader):
                    if fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    elapsed = time.time() - start
                    speed = global_step / max(elapsed, 1e-6)
                    eta = (total_steps - global_step) / max(speed, 1e-6)
                    raw_loss = float(loss.item() * grad_accum)

                    state = self._update_state(
                        slug,
                        epoch=epoch,
                        step=global_step,
                        total_steps=total_steps,
                        loss=raw_loss,
                        elapsed_seconds=elapsed,
                        eta_seconds=eta,
                        message=f"Эпоха {epoch}/{epochs} — шаг {global_step}/{total_steps}",
                    )
                    if global_step % 5 == 0:
                        state.history.append(
                            {"step": global_step, "epoch": epoch, "loss": raw_loss}
                        )

        return None

    # ------------------------------------------------------------------
    def _update_state(self, slug: str, **changes: Any) -> TrainingState:
        with self._lock:
            state = self._states.get(slug) or TrainingState()
            for key, value in changes.items():
                setattr(state, key, value)
            self._states[slug] = state
            snapshot = TrainingState(**asdict(state))
        try:
            self.models.update_status(
                slug,
                "training" if state.status not in ("done", "error", "cancelled") else
                ("trained" if state.status == "done" else "error" if state.status == "error" else "untrained"),
                error=state.error,
                training=state.to_dict(),
            )
        except FileNotFoundError:
            pass
        return snapshot

    def _fail(self, slug: str, message: str) -> None:
        log.error("Training failed for %s: %s", slug, message)
        self._update_state(slug, status="error", error=message, message=message, finished_at=now_iso())
        try:
            self.models.update_status(slug, "error", error=message, training=self.get_state(slug).to_dict())
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Internal sentinel exceptions
# ---------------------------------------------------------------------------

class _OOM(Exception):
    """Raised internally when CUDA reports out-of-memory."""
