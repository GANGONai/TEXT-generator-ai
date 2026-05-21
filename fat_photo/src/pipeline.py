"""High-level entry point: ``fatten_photo(image, intensity, use_ai)``.

This is what the Gradio app calls. It orchestrates landmark detection,
geometric warping and (optionally) Stable Diffusion refinement, and returns
both the final image and a short status string for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .fatten import fatten_geometric
from .landmarks import detect_face, detect_person_mask, detect_pose
from .warp import blend_with_mask, warp_image
from .fatten import build_face_controls, build_body_controls


@dataclass
class FattenResult:
    image: Image.Image
    message: str
    detected_face: bool
    detected_body: bool


def _to_rgb_uint8(image) -> np.ndarray:
    """Accept PIL Image / ndarray / file path and return HxWx3 uint8 RGB."""
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def fatten_photo(
    image,
    intensity: float = 0.5,
    use_ai_refine: bool = False,
    ai_strength: float = 0.25,
    ai_steps: int = 25,
    seed: Optional[int] = None,
    keep_background: bool = True,
) -> FattenResult:
    """Make the person in ``image`` look heavier.

    Parameters
    ----------
    image: PIL Image, numpy array or path. Any size; the output keeps the
        original resolution.
    intensity: 0..1 (the UI slider is 0..100 and gets divided by 100).
    use_ai_refine: if True, runs Stable Diffusion img2img on the warped image
        to add realistic texture. Requires a GPU for reasonable speed.
    ai_strength: img2img denoising strength when ``use_ai_refine`` is on.
    ai_steps: number of diffusion steps.
    seed: optional integer seed for reproducible AI refinement.
    keep_background: blend the warped person back over the original
        background using a soft person mask (so the floor/sky/etc don't warp).
    """
    rgb = _to_rgb_uint8(image)
    h, w = rgb.shape[:2]

    face = detect_face(rgb)
    pose = detect_pose(rgb)

    if face is None and pose is None:
        return FattenResult(
            image=Image.fromarray(rgb),
            message="Не нашёл ни лица, ни тела на фото — попробуй фото получше или другое.",
            detected_face=False,
            detected_body=False,
        )

    warped = fatten_geometric(rgb, face, pose, intensity)

    if keep_background:
        mask = detect_person_mask(rgb)
        if mask is not None:
            # The warp expands the person's silhouette outward, so a tight mask
            # would clip the new fat back to the original outline. We compute a
            # "warped mask" by running the same warp on the mask and then take
            # the union of the original and warped masks — this covers every
            # pixel that is part of the person *before or after* the deform.
            warped_mask = _warp_mask_like_image(mask, rgb, face, pose, intensity)
            union = np.maximum(mask, warped_mask)
            # Dilate slightly to hide aliasing seams along the silhouette edge.
            union = _soft_dilate(union, radius=max(3, int(0.01 * max(h, w))))
            warped = blend_with_mask(rgb, warped, union)

    if use_ai_refine:
        from .refine import refine  # local import keeps torch optional
        try:
            warped = refine(
                warped,
                strength=ai_strength,
                steps=ai_steps,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            # Refinement is optional — never fail the whole pipeline on it.
            return FattenResult(
                image=Image.fromarray(warped),
                message=(
                    "Готово (без AI-улучшения, ошибка диффузии: "
                    f"{exc.__class__.__name__}: {exc})."
                ),
                detected_face=face is not None,
                detected_body=pose is not None,
            )

    parts = []
    if face is not None:
        parts.append("лицо")
    if pose is not None:
        parts.append("тело")
    detected_str = " + ".join(parts) if parts else "—"
    suffix = " + AI-доработка" if use_ai_refine else ""
    msg = f"Готово. Определил: {detected_str}{suffix}. Интенсивность: {int(intensity * 100)}%."
    return FattenResult(
        image=Image.fromarray(warped),
        message=msg,
        detected_face=face is not None,
        detected_body=pose is not None,
    )


def _warp_mask_like_image(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    face,
    pose,
    intensity: float,
) -> np.ndarray:
    """Apply the same control-point warp to the person mask.

    We rebuild the control points instead of caching them so that ``fatten_geometric``
    stays free of side effects. The mask is converted to uint8 for the warp
    (which expects an image-like ndarray) and back to float32 in [0, 1].
    """
    intensity = float(max(0.0, min(1.0, intensity)))
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    sig_chunks: list[np.ndarray] = []

    if face is not None:
        fs, fd, fsig = build_face_controls(face, intensity)
        if len(fs):
            src_chunks.append(fs)
            dst_chunks.append(fd)
            sig_chunks.append(fsig)
    if pose is not None:
        h_img, w_img = image_rgb.shape[:2]
        bs, bd, bsig = build_body_controls(pose, intensity, image_size=(w_img, h_img))
        if len(bs):
            src_chunks.append(bs)
            dst_chunks.append(bd)
            sig_chunks.append(bsig)

    if not src_chunks:
        return mask.astype(np.float32)

    src = np.concatenate(src_chunks, axis=0)
    dst = np.concatenate(dst_chunks, axis=0)
    sigmas = np.concatenate(sig_chunks, axis=0)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    warped_u8 = warp_image(mask_u8, src, dst, sigmas)
    return warped_u8.astype(np.float32) / 255.0


def _soft_dilate(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Soft dilation of a float mask in [0, 1] using a separable box blur.

    NumPy-only — we avoid scipy/cv2 dependencies in the core warp path.
    """
    if radius < 1:
        return mask.astype(np.float32)
    m = mask.astype(np.float32)
    # Saturate-then-blur gives a soft expansion.
    m = np.clip(m * 1.2 + 0.05, 0.0, 1.0)
    # Box blur in both directions.
    k = 2 * radius + 1
    kernel = np.ones(k, dtype=np.float32) / k
    # Pad-reflect to keep edges clean.
    pad = radius
    m_padded = np.pad(m, ((pad, pad), (pad, pad)), mode="edge")
    blurred = np.apply_along_axis(
        lambda r: np.convolve(r, kernel, mode="valid"), axis=1, arr=m_padded
    )
    blurred = np.apply_along_axis(
        lambda c: np.convolve(c, kernel, mode="valid"), axis=0, arr=blurred
    )
    return np.clip(blurred, 0.0, 1.0)
