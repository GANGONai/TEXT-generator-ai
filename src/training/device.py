"""GPU availability helpers.

Single source of truth for "do we have a CUDA GPU on this account?".

We import ``torch`` lazily so that simply importing the training package
(or running unit tests) does not pull torch into RAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GPUInfo:
    """Snapshot of GPU availability suitable for showing in the UI."""

    available: bool = False
    name: str = ""
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    cuda_version: str = ""
    torch_version: str = ""
    device_count: int = 0
    all_devices: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def human_summary(self) -> str:
        """Markdown-formatted summary for the Gradio UI."""

        if self.error:
            return (
                f"🔴 **GPU не определена** — {self.error}\n\n"
                "Откройте `Runtime → Change runtime type → T4 GPU` в Colab и "
                "перезапустите ячейку."
            )
        if not self.available:
            return (
                "🔴 **GPU недоступна на этой сессии Colab.**\n\n"
                f"PyTorch {self.torch_version} установлен, но CUDA не видна.\n\n"
                "Откройте `Runtime → Change runtime type → T4 GPU` и снова "
                "запустите ячейку с приложением. Тренинг на CPU отключён — "
                "он занял бы часы и подвесил бы ноутбук."
            )
        devices = ", ".join(self.all_devices) if self.all_devices else self.name
        return (
            f"🟢 **GPU доступна:** {devices}\n\n"
            f"VRAM: {self.free_memory_gb:.1f} / {self.total_memory_gb:.1f} GB свободно · "
            f"CUDA {self.cuda_version} · PyTorch {self.torch_version}"
        )


def get_gpu_info() -> GPUInfo:
    """Probe ``torch`` for CUDA availability and return a :class:`GPUInfo`.

    Never raises — any import / driver error is captured into ``info.error``
    so the UI can surface it gracefully.
    """

    try:
        import torch
    except ImportError as exc:
        return GPUInfo(error=f"torch не установлен: {exc}")

    info = GPUInfo(torch_version=torch.__version__)
    try:
        info.available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver-level breakage
        return GPUInfo(torch_version=torch.__version__, error=str(exc))

    if not info.available:
        return info

    try:
        info.cuda_version = torch.version.cuda or ""
        info.device_count = torch.cuda.device_count()
        info.all_devices = [
            torch.cuda.get_device_name(i) for i in range(info.device_count)
        ]
        info.name = info.all_devices[0] if info.all_devices else ""
        props = torch.cuda.get_device_properties(0)
        info.total_memory_gb = props.total_memory / (1024 ** 3)
        free, _ = torch.cuda.mem_get_info(0)
        info.free_memory_gb = free / (1024 ** 3)
    except Exception as exc:  # pragma: no cover
        info.error = f"GPU видна, но детали недоступны: {exc}"

    return info


def require_gpu() -> GPUInfo:
    """Return a :class:`GPUInfo` if a GPU is available, otherwise raise.

    Use this at the very start of any operation that should refuse to run on
    CPU (fine-tuning, in particular). The raised ``RuntimeError`` is meant to
    be shown to the user verbatim — it includes the actionable Colab steps.
    """

    info = get_gpu_info()
    if not info.available:
        msg = (
            "GPU не обнаружена — обучение требует CUDA.\n\n"
            "В Google Colab включите GPU: Runtime → Change runtime type → "
            "Hardware accelerator → T4 GPU, затем заново запустите ячейку с "
            "приложением.\n\n"
            f"Подробности: {info.error or 'torch.cuda.is_available() = False'}"
        )
        raise RuntimeError(msg)
    return info
