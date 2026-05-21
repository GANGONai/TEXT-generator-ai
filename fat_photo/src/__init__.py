"""fat_photo: turn a person on a photo into a heavier version of themselves.

Pipeline:
1. Detect face + body landmarks with MediaPipe.
2. Compute a set of (source, target) control-point pairs that push the
   silhouette outward proportionally to the user's "intensity" slider.
3. Warp the image with an RBF (Gaussian-kernel) displacement field.
4. Optionally refine with Stable Diffusion img2img to add realistic fat/skin
   textures and clean up warp artefacts.

Everything runs on a free Google Colab T4 GPU (CPU fallback also works,
without the AI refinement step).
"""

from .pipeline import fatten_photo

__all__ = ["fatten_photo"]
