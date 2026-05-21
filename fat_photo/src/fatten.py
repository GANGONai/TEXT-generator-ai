"""Build (source, target) control points for a "fatten" warp.

The strategy is the same for face and body:
1. Take a set of landmarks that sit on the outline of a body part.
2. Push them outward perpendicular to the part's main axis, by an amount
   proportional to a ``strength`` value in [0, 1].
3. Pair each point ``P`` with its pushed counterpart ``Q`` so that the warp
   routine knows "this pixel needs to end up over there".

The amount of push for each body part is calibrated empirically:
- Face widening is in units of the inter-ocular distance (so it scales with
  how large the face is in the frame).
- Body widening is in units of the shoulder width.
- The "double chin" puff is a small downward push on the chin landmarks.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .landmarks import (
    FaceLandmarks,
    LEFT_EYE_OUTER,
    PoseLandmarks,
    RIGHT_EYE_OUTER,
)
from .warp import warp_image


# Per-part scale factors. ``strength=1.0`` means "fully fat"; users typically
# pick 0.3–0.7. The face uses the inter-ocular distance as a unit, the body
# uses the shoulder width.
FACE_JAW_PUSH = 0.55        # widen jawline by 55% of eye-distance at strength=1
FACE_CHEEK_PUSH = 0.40      # cheek widening
FACE_CHIN_DOWN = 0.45       # double-chin push
BODY_TORSO_PUSH = 0.40      # widen torso by 40% of shoulder width
BODY_BELLY_PUSH = 0.55      # belly bulge (perpendicular to shoulder line)
BODY_HIP_PUSH = 0.55        # hip widening
BODY_THIGH_PUSH = 0.45      # thigh widening


def _normal(vec: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``vec`` (2D), pointing 90° clockwise."""
    n = np.array([-vec[1], vec[0]], dtype=np.float32)
    norm = float(np.linalg.norm(n))
    return n / max(norm, 1e-6)


def _add_pair(
    src: List[np.ndarray],
    dst: List[np.ndarray],
    sigmas: List[float],
    p_src: np.ndarray,
    p_dst: np.ndarray,
    sigma: float,
) -> None:
    src.append(p_src.astype(np.float32))
    dst.append(p_dst.astype(np.float32))
    sigmas.append(float(sigma))


def build_face_controls(
    face: FaceLandmarks, strength: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build face control points for a fattening warp."""
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    sigmas: List[float] = []

    unit = face.eye_distance  # pixels per "1 face unit"
    face_centre_x = float(face.points[:, 0].mean())
    chin_tip = face.chin[0]

    # Reference Y for upper-vs-lower face. Anything above the nose tip is
    # mostly forehead/eyes — we want to *barely* warp those. Anything well
    # below the nose tip (cheeks, jaw, chin) gets the full push. The factor
    # `lower_weight(y)` smoothly goes from ~0 at the nose to 1 at the chin
    # so eyes / glasses don't get stretched.
    nose_tip_y = float(face.points[1, 1])  # FaceMesh index 1 = nose tip
    chin_y = float(chin_tip[1])
    face_height = max(chin_y - nose_tip_y, unit * 0.5)

    def lower_weight(y: float) -> float:
        # Cubic ease-in: ~0 at the nose, steep ramp-up below it, =1 at the chin.
        # Cubing keeps the eyes / glasses almost untouched even at strength=1.
        t = (y - nose_tip_y) / face_height
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * t

    # --- Jawline widening: push left/right jaw outward, weighted by how far
    # below the nose the landmark is.
    for p in face.jaw_left:
        w = lower_weight(p[1])
        offset = (face_centre_x - p[0]) / max(abs(face_centre_x - p[0]), 1.0)
        q = p + np.array([offset * FACE_JAW_PUSH * unit * strength * w, 0.0])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.5)
    for p in face.jaw_right:
        w = lower_weight(p[1])
        offset = (face_centre_x - p[0]) / max(abs(face_centre_x - p[0]), 1.0)
        q = p + np.array([offset * FACE_JAW_PUSH * unit * strength * w, 0.0])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.5)

    # --- Cheek puff: push cheeks outward laterally (cheek landmarks are
    # already below the nose, but apply the weight anyway for safety).
    for p in face.cheek_left:
        w = lower_weight(p[1])
        q = p + np.array([-FACE_CHEEK_PUSH * unit * strength * w, 0.0])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.4)
    for p in face.cheek_right:
        w = lower_weight(p[1])
        q = p + np.array([FACE_CHEEK_PUSH * unit * strength * w, 0.0])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.4)

    # --- Double chin: push chin downward + add a soft fat ring below.
    for p in face.chin:
        q = p + np.array([0.0, FACE_CHIN_DOWN * unit * strength])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.5)

    # An extra ring of pull-down landmarks beneath the chin to spread the fat.
    for dx in (-0.5, -0.25, 0.0, 0.25, 0.5):
        p = chin_tip + np.array([dx * unit, unit * 0.2])
        q = p + np.array([0.0, FACE_CHIN_DOWN * unit * strength * 0.5])
        _add_pair(src, dst, sigmas, p, q, sigma=unit * 0.45)

    # --- Anchor points: identical src and dst, so the warp is *forced* to keep
    # these locations in place. We put them at the eyes, eyebrows, nose tip
    # and forehead so the upper face doesn't get dragged outward by the jaw
    # warp's Gaussian falloff. This is the standard trick from MLS / RBF
    # deformation literature for preserving regions of an image.
    anchors = [
        face.points[LEFT_EYE_OUTER],
        face.points[RIGHT_EYE_OUTER],
        face.points[33],   # left eye outer (re-anchor for stronger pin)
        face.points[133],  # left eye inner
        face.points[362],  # right eye inner
        face.points[263],  # right eye outer
        face.points[1],    # nose tip
        face.points[10],   # forehead top
        face.points[151],  # forehead centre
        face.points[105],  # left eyebrow
        face.points[334],  # right eyebrow
    ]
    for p in anchors:
        _add_pair(src, dst, sigmas, p, p, sigma=unit * 0.6)

    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        np.asarray(sigmas, dtype=np.float32),
    )


def build_body_controls(
    pose: PoseLandmarks, strength: float, image_size: Tuple[int, int] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build body control points for a fattening warp.

    ``image_size`` is ``(width, height)`` of the source image (optional). When
    provided, we drop control points that fall outside the image, which avoids
    extreme warps when only the head/shoulders are visible (and the hip/knee
    points are extrapolated off-screen by MediaPipe).
    """
    empty = (
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    sigmas: List[float] = []

    shoulder_vec = pose.right_shoulder - pose.left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_vec))
    if shoulder_width < 5.0:
        return empty

    # If hips are not visible (only the upper body is in frame), skip body
    # warping entirely — the extrapolated hip / belly points would be off the
    # image and produce ugly streaking artefacts at the bottom edge.
    hip_vis = float(
        min(
            pose.visibility[23],  # POSE_LEFT_HIP
            pose.visibility[24],  # POSE_RIGHT_HIP
        )
    )
    hips_in_frame = True
    if image_size is not None:
        w_img, h_img = image_size
        for hip in (pose.left_hip, pose.right_hip):
            if not (0 <= hip[0] < w_img and 0 <= hip[1] < h_img):
                hips_in_frame = False
                break
    if hip_vis < 0.4 or not hips_in_frame:
        return empty

    shoulder_dir = shoulder_vec / shoulder_width
    body_normal = _normal(shoulder_dir)  # perpendicular to shoulder line

    hip_centre = 0.5 * (pose.left_hip + pose.right_hip)
    shoulder_centre = 0.5 * (pose.left_shoulder + pose.right_shoulder)
    torso_vec = hip_centre - shoulder_centre
    torso_len = float(np.linalg.norm(torso_vec))
    body_centre = 0.5 * (hip_centre + shoulder_centre)

    # --- Widen torso outline. Pick a few points down the side of the torso.
    for t in np.linspace(0.1, 0.95, 6):
        # left side: shoulder + t * (hip - shoulder), pushed left
        p_l = pose.left_shoulder + t * (pose.left_hip - pose.left_shoulder)
        q_l = p_l - shoulder_dir * BODY_TORSO_PUSH * shoulder_width * strength
        _add_pair(src, dst, sigmas, p_l, q_l, sigma=shoulder_width * 0.6)

        p_r = pose.right_shoulder + t * (pose.right_hip - pose.right_shoulder)
        q_r = p_r + shoulder_dir * BODY_TORSO_PUSH * shoulder_width * strength
        _add_pair(src, dst, sigmas, p_r, q_r, sigma=shoulder_width * 0.6)

    # --- Hip widening.
    q_lh = pose.left_hip - shoulder_dir * BODY_HIP_PUSH * shoulder_width * strength
    q_rh = pose.right_hip + shoulder_dir * BODY_HIP_PUSH * shoulder_width * strength
    _add_pair(src, dst, sigmas, pose.left_hip, q_lh, sigma=shoulder_width * 0.7)
    _add_pair(src, dst, sigmas, pose.right_hip, q_rh, sigma=shoulder_width * 0.7)

    # --- Belly bulge. The belly is roughly 60% down the torso. We push the
    # midline outward along the body normal — note this is in the image plane,
    # so for a frontal photo the "outward" direction is mostly downward/forward
    # on the silhouette. We push BOTH sides outward perpendicular to shoulders.
    belly_centre = shoulder_centre + 0.6 * torso_vec
    bulge = BODY_BELLY_PUSH * shoulder_width * strength
    # Belly silhouette = belly_centre +/- shoulder_dir * shoulder_width/2
    # and we additionally push downward (body_normal points down for typical
    # standing pose).
    for offset in np.linspace(-0.55, 0.55, 5):
        p = belly_centre + shoulder_dir * shoulder_width * offset
        q = p + body_normal * bulge
        _add_pair(src, dst, sigmas, p, q, sigma=shoulder_width * 0.8)
    _ = torso_len  # silence flake8 about unused; useful when extending warp
    _ = body_centre

    # --- Thigh widening (only if knees are visible).
    if pose.left_knee is not None and pose.right_knee is not None:
        thigh_width = max(
            float(np.linalg.norm(pose.right_hip - pose.left_hip)), shoulder_width
        )
        for t in np.linspace(0.2, 0.85, 4):
            p_l = pose.left_hip + t * (pose.left_knee - pose.left_hip)
            q_l = p_l - shoulder_dir * BODY_THIGH_PUSH * thigh_width * strength
            _add_pair(src, dst, sigmas, p_l, q_l, sigma=thigh_width * 0.6)

            p_r = pose.right_hip + t * (pose.right_knee - pose.right_hip)
            q_r = p_r + shoulder_dir * BODY_THIGH_PUSH * thigh_width * strength
            _add_pair(src, dst, sigmas, p_r, q_r, sigma=thigh_width * 0.6)

    return (
        np.asarray(src, dtype=np.float32),
        np.asarray(dst, dtype=np.float32),
        np.asarray(sigmas, dtype=np.float32),
    )


def fatten_geometric(
    image_rgb: np.ndarray,
    face: FaceLandmarks | None,
    pose: PoseLandmarks | None,
    strength: float,
) -> np.ndarray:
    """Apply face + body warping and return the fattened image.

    ``strength`` is clamped to [0, 1]. If neither face nor body is detected we
    return the input untouched (the pipeline then tells the user no person
    was found).
    """
    strength = float(max(0.0, min(1.0, strength)))
    src_chunks: List[np.ndarray] = []
    dst_chunks: List[np.ndarray] = []
    sigma_chunks: List[np.ndarray] = []

    h, w = image_rgb.shape[:2]
    image_size = (w, h)

    if face is not None:
        fs, fd, fsig = build_face_controls(face, strength)
        if len(fs):
            src_chunks.append(fs)
            dst_chunks.append(fd)
            sigma_chunks.append(fsig)

    if pose is not None:
        bs, bd, bsig = build_body_controls(pose, strength, image_size=image_size)
        if len(bs):
            src_chunks.append(bs)
            dst_chunks.append(bd)
            sigma_chunks.append(bsig)

    if not src_chunks:
        return image_rgb.copy()

    src = np.concatenate(src_chunks, axis=0)
    dst = np.concatenate(dst_chunks, axis=0)
    sigmas = np.concatenate(sigma_chunks, axis=0)
    return warp_image(image_rgb, src, dst, sigmas)
