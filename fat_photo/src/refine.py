"""Optional Stable Diffusion img2img refinement.

The geometric warp gives us correct silhouettes but soft / blurry textures
(especially around the warped belly and chin). We optionally run a low-strength
img2img pass with a "fat / overweight" prompt to add realistic skin folds and
clean up artefacts.

This module is lazy — Stable Diffusion is only loaded the first time
``refine`` is called, so users who only want the geometric warp pay no cost.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

_PIPE = None  # cached pipeline (loaded once)


# Realistic Vision is a free, open Stable Diffusion 1.5 fine-tune that handles
# human anatomy noticeably better than the vanilla SD 1.5 weights and fits
# easily on a free Colab T4 (~6 GB VRAM in fp16).
DEFAULT_MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
FALLBACK_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Prompt + negative prompt tuned for "make this person heavier" while
# preserving identity. The img2img strength is what actually controls how
# much the model is allowed to change.
POSITIVE_PROMPT = (
    "RAW photo, full body portrait of an overweight, obese, very fat person, "
    "thick neck, double chin, chubby cheeks, large belly, wide hips, "
    "skin folds, photorealistic, natural skin texture, soft studio lighting, "
    "high detail, 8k uhd, dslr"
)
NEGATIVE_PROMPT = (
    "thin, skinny, slim, deformed, distorted, disfigured, extra limbs, "
    "extra fingers, mutated, ugly, blurry, low quality, cartoon, anime, "
    "painting, illustration, watermark, text, logo"
)


def _is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_pipeline(model_id: str):
    """Lazily build and cache a Stable Diffusion img2img pipeline."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except Exception:
        # Some Hugging Face mirrors don't have a VAE alongside Realistic Vision.
        # Fall back to vanilla SD 1.5 — same architecture, slightly worse anatomy.
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            FALLBACK_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_attention_slicing()
    else:
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()

    _PIPE = pipe
    return _PIPE


def refine(
    image_rgb: np.ndarray,
    strength: float = 0.25,
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = 25,
    guidance_scale: float = 6.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Run SD img2img on the already-warped image.

    Parameters
    ----------
    image_rgb: HxWx3 uint8 RGB array.
    strength: 0..1 — how much the diffusion is allowed to change. Low values
        (~0.2) just clean up texture; high values (~0.5) actively reshape.
    model_id: HF model id to use.
    steps, guidance_scale: standard diffusion params.
    seed: optional integer seed for reproducible output.

    Returns
    -------
    HxWx3 uint8 RGB array.
    """
    if not _is_torch_available():
        # No torch → just return the input. The pipeline still works
        # without refinement; this is purely an optional polish step.
        return image_rgb

    pipe = _load_pipeline(model_id)

    import torch

    pil = Image.fromarray(image_rgb).convert("RGB")
    # SD 1.5 likes side lengths divisible by 8 and works best around 512–768 px.
    w, h = pil.size
    max_side = 768
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale != 1.0:
        new_size = (max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8))
        pil_resized = pil.resize(new_size, Image.LANCZOS)
    else:
        new_size = ((w // 8) * 8, (h // 8) * 8)
        pil_resized = pil.resize(new_size, Image.LANCZOS) if new_size != (w, h) else pil

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

    result = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil_resized,
        strength=float(max(0.05, min(0.95, strength))),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images[0]

    if result.size != (w, h):
        result = result.resize((w, h), Image.LANCZOS)
    return np.asarray(result.convert("RGB"))
