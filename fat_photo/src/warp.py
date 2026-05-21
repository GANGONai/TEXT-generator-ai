"""Radial-basis-function (Gaussian) image warp driven by control points.

Given ``N`` pairs of source ``P[i]`` and target ``Q[i]`` points in image space,
we want to deform the image so that ``P[i]`` ends up at ``Q[i]`` while the rest
of the image smoothly follows along.

We use a backward map: for every output pixel ``v`` we estimate where it
came from in the input, then bilinearly sample there. The estimate is a
Gaussian-weighted average of "this point was displaced by ``d_i = P[i] - Q[i]``"
hints from each control pair:

    src(v) = v + sum_i w_i(v) * (P[i] - Q[i])
    w_i(v) = exp(-||v - Q[i]||^2 / (2 * sigma_i^2)) / sum_j w_j(v) + base

The ``base`` term ensures that far away from any control point the
displacement decays to zero, leaving the background untouched. Each control
point gets its own ``sigma`` so we can have very localised pushes (e.g. on the
chin) and broad ones (e.g. on the belly).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _bilinear_sample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``image`` at floating-point coordinates ``(map_x, map_y)``.

    Pure NumPy implementation (no OpenCV dep). Out-of-bounds reads are clamped
    to the image edge so the warped result has no black borders.
    """
    h, w = image.shape[:2]
    x = np.clip(map_x, 0, w - 1)
    y = np.clip(map_y, 0, h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    img = image.astype(np.float32)
    if img.ndim == 2:
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]
        top = Ia * (1 - wx) + Ib * wx
        bot = Ic * (1 - wx) + Id * wx
        out = top * (1 - wy) + bot * wy
    else:
        wx3 = wx[..., None]
        wy3 = wy[..., None]
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]
        top = Ia * (1 - wx3) + Ib * wx3
        bot = Ic * (1 - wx3) + Id * wx3
        out = top * (1 - wy3) + bot * wy3

    return np.clip(out, 0, 255).astype(image.dtype)


def warp_image(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    sigmas: Sequence[float] | float,
) -> np.ndarray:
    """Warp ``image`` so that ``src_points`` move to ``dst_points``.

    Parameters
    ----------
    image: HxW or HxWxC uint8 array.
    src_points: (N, 2) float array — where each control point starts.
    dst_points: (N, 2) float array — where each control point should end up.
    sigmas: per-point Gaussian falloff radius in pixels (or a single scalar).

    Returns
    -------
    Warped image with the same shape and dtype as ``image``.
    """
    if src_points.shape != dst_points.shape or src_points.shape[1] != 2:
        raise ValueError("src/dst points must be (N, 2) arrays of the same shape")
    if len(src_points) == 0:
        return image.copy()

    h, w = image.shape[:2]
    sigmas_arr = np.asarray(sigmas, dtype=np.float32).reshape(-1)
    if sigmas_arr.size == 1:
        sigmas_arr = np.full(len(src_points), float(sigmas_arr[0]), dtype=np.float32)
    if sigmas_arr.shape[0] != src_points.shape[0]:
        raise ValueError("`sigmas` must be a scalar or have one entry per control point")

    # Build the output coordinate grid once.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = np.stack([xx, yy], axis=-1)  # (H, W, 2)

    # Displacement we want to *undo* to get the source pixel: src = v + (P - Q).
    displacements = (src_points - dst_points).astype(np.float32)  # (N, 2)

    # Compute weights w_i(v) = exp(-||v - Q_i||^2 / (2 * sigma_i^2))
    # We accumulate sum_i w_i * d_i and sum_i w_i in chunks to keep memory low.
    accum_disp = np.zeros((h, w, 2), dtype=np.float32)
    accum_w = np.zeros((h, w), dtype=np.float32)

    for i in range(len(dst_points)):
        q = dst_points[i]
        d = displacements[i]
        sigma = max(float(sigmas_arr[i]), 1.0)

        dx = grid[..., 0] - q[0]
        dy = grid[..., 1] - q[1]
        dist2 = dx * dx + dy * dy

        # Skip pixels more than 4 sigma away — they contribute < 0.0003.
        cutoff = (4.0 * sigma) ** 2
        active = dist2 < cutoff
        if not np.any(active):
            continue
        w_i = np.zeros_like(dist2)
        w_i[active] = np.exp(-dist2[active] / (2.0 * sigma * sigma))

        accum_disp[..., 0] += w_i * d[0]
        accum_disp[..., 1] += w_i * d[1]
        accum_w += w_i

    # Normalise. Where the total weight is tiny, displacement -> 0
    # (which means "this pixel is far from every control point, leave alone").
    eps = 1e-3
    norm = np.maximum(accum_w, eps)[..., None]
    displacement_field = accum_disp / norm
    # Suppress displacement when the raw accumulated weight is small so the
    # background outside the influence zone stays put.
    falloff = np.clip(accum_w, 0.0, 1.0)[..., None]
    displacement_field *= falloff

    map_x = grid[..., 0] + displacement_field[..., 0]
    map_y = grid[..., 1] + displacement_field[..., 1]

    return _bilinear_sample(image, map_x, map_y)


def push_outward(
    points: np.ndarray,
    centre: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Return new positions for ``points`` pushed radially away from ``centre``.

    The displacement is proportional to the distance from ``centre`` so close
    points move less than far points, which keeps the relative shape sensible.
    """
    delta = points - centre
    return points + delta * amount


def fan_outward(
    points: np.ndarray,
    axis: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Push ``points`` perpendicular to a given ``axis`` direction.

    Useful for "widen the torso laterally" without changing the vertical
    position of the points. ``axis`` is a 2-vector; we project the
    point->centre vector onto the perpendicular of ``axis``.
    """
    n = np.array([-axis[1], axis[0]], dtype=np.float32)
    n /= max(np.linalg.norm(n), 1e-6)
    centre = points.mean(axis=0)
    rel = points - centre
    side = np.sign(rel @ n)  # which side of the axis each point is on
    side[side == 0] = 1.0
    return points + np.outer(side, n) * amount


def blend_with_mask(
    base: np.ndarray, top: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Alpha-blend ``top`` over ``base`` using a soft float ``mask`` in [0, 1]."""
    if mask.ndim == 2:
        mask = mask[..., None]
    base_f = base.astype(np.float32)
    top_f = top.astype(np.float32)
    out = base_f * (1.0 - mask) + top_f * mask
    return np.clip(out, 0, 255).astype(base.dtype)
