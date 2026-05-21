"""MediaPipe-based face and body landmark detection.

MediaPipe is the lightest free option available — pretrained models that
detect 478 face-mesh points, 33 pose points, and a person segmentation mask
on CPU in well under a second.

We support BOTH MediaPipe APIs transparently:
- The legacy ``mediapipe.solutions`` API (still available on Python 3.11,
  which is what Google Colab uses today). No model downloads required.
- The modern ``mediapipe.tasks`` API (the only one available on Python 3.12+).
  This API requires ``.task`` model bundles which we download on first use
  from MediaPipe's public model zoo (~6 MB total) and cache under the user's
  home directory.

The exported functions return ``FaceLandmarks`` / ``PoseLandmarks`` / mask
arrays — the caller doesn't need to know which API was used internally.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# FaceMesh / Pose landmark indices.
# --------------------------------------------------------------------------- #

# Outer jawline — left side (temple to chin) and right side.
JAW_LEFT_IDX = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
JAW_RIGHT_IDX = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

# Chin tip + soft-tissue under the chin (double-chin area).
CHIN_IDX = [152, 175, 199, 200, 18]

# Cheek puff points.
CHEEK_LEFT_IDX = [216, 207, 187, 147]
CHEEK_RIGHT_IDX = [436, 427, 411, 376]

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Pose landmark indices (MediaPipe Pose, 33 points).
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_KNEE = 25
POSE_RIGHT_KNEE = 26


# --------------------------------------------------------------------------- #
# Result dataclasses.
# --------------------------------------------------------------------------- #


@dataclass
class FaceLandmarks:
    points: np.ndarray  # (468 or 478, 2) pixel coords
    jaw_left: np.ndarray
    jaw_right: np.ndarray
    chin: np.ndarray
    cheek_left: np.ndarray
    cheek_right: np.ndarray
    eye_distance: float


@dataclass
class PoseLandmarks:
    points: np.ndarray  # (33, 2)
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray
    left_knee: Optional[np.ndarray]
    right_knee: Optional[np.ndarray]
    # Visibility scores in [0, 1] for each of the 33 landmarks. Useful to
    # decide whether a body part is actually in frame — when only the head and
    # shoulders are visible, the "hip" points returned by MediaPipe are
    # extrapolated off-screen and shouldn't be used to drive a belly warp.
    visibility: np.ndarray


# --------------------------------------------------------------------------- #
# Model file URLs (used by the modern ``tasks`` API). All free from Google.
# --------------------------------------------------------------------------- #

_TASK_MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}


def _cache_dir() -> Path:
    home = Path(os.environ.get("FAT_PHOTO_CACHE", str(Path.home() / ".cache" / "fat_photo")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def _ensure_model(name: str) -> Path:
    """Download (once) and return the path to a MediaPipe .task model."""
    path = _cache_dir() / name
    if path.exists() and path.stat().st_size > 1024:
        return path
    url = _TASK_MODEL_URLS[name]
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# Legacy ``mediapipe.solutions`` backend (Colab Python 3.11).
# --------------------------------------------------------------------------- #


def _has_solutions() -> bool:
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions")
    except Exception:
        return False


def _solutions_to_pixels(landmarks, w: int, h: int) -> np.ndarray:
    return np.array(
        [[lm.x * w, lm.y * h] for lm in landmarks.landmark],
        dtype=np.float32,
    )


def _detect_face_solutions(image_rgb: np.ndarray) -> Optional[FaceLandmarks]:
    import mediapipe as mp

    h, w = image_rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
    ) as fm:
        result = fm.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    pts = _solutions_to_pixels(result.multi_face_landmarks[0], w, h)
    return _face_from_points(pts)


def _detect_pose_solutions(image_rgb: np.ndarray) -> Optional[PoseLandmarks]:
    import mediapipe as mp

    h, w = image_rgb.shape[:2]
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    ) as pose:
        result = pose.process(image_rgb)
    if not result.pose_landmarks:
        return None
    pts = _solutions_to_pixels(result.pose_landmarks, w, h)
    vis = np.array(
        [lm.visibility for lm in result.pose_landmarks.landmark], dtype=np.float32
    )
    return _pose_from_points(pts, vis)


def _detect_mask_solutions(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    import mediapipe as mp

    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(image_rgb)
    if result.segmentation_mask is None:
        return None
    return result.segmentation_mask.astype(np.float32)


# --------------------------------------------------------------------------- #
# Modern ``mediapipe.tasks`` backend (Python 3.12+).
# --------------------------------------------------------------------------- #


def _detect_face_tasks(image_rgb: np.ndarray) -> Optional[FaceLandmarks]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model("face_landmarker.task")
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as fm:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = fm.detect(mp_image)
    if not result.face_landmarks:
        return None

    h, w = image_rgb.shape[:2]
    pts = np.array(
        [[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
        dtype=np.float32,
    )
    return _face_from_points(pts)


def _detect_pose_tasks(image_rgb: np.ndarray) -> Optional[PoseLandmarks]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model("pose_landmarker.task")
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    with mp_vision.PoseLandmarker.create_from_options(options) as pose:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = pose.detect(mp_image)
    if not result.pose_landmarks:
        return None

    h, w = image_rgb.shape[:2]
    landmarks = result.pose_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    vis = np.array([float(lm.visibility) for lm in landmarks], dtype=np.float32)
    return _pose_from_points(pts, vis)


def _detect_mask_tasks(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_model("selfie_segmenter.tflite")
    options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )
    with mp_vision.ImageSegmenter.create_from_options(options) as seg:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = seg.segment(mp_image)
    if not result.confidence_masks:
        return None
    # The selfie_segmenter model returns a single confidence mask where 1.0
    # means "person" and 0.0 means "background".
    mask_view = result.confidence_masks[0].numpy_view()
    arr = np.array(mask_view, dtype=np.float32)
    # Some versions return (H, W, 1); the rest of the pipeline wants (H, W).
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


# --------------------------------------------------------------------------- #
# Shared helpers — convert raw point arrays to our dataclasses.
# --------------------------------------------------------------------------- #


def _face_from_points(pts: np.ndarray) -> FaceLandmarks:
    eye_dist = float(np.linalg.norm(pts[LEFT_EYE_OUTER] - pts[RIGHT_EYE_OUTER]))
    return FaceLandmarks(
        points=pts,
        jaw_left=pts[JAW_LEFT_IDX],
        jaw_right=pts[JAW_RIGHT_IDX],
        chin=pts[CHIN_IDX],
        cheek_left=pts[CHEEK_LEFT_IDX],
        cheek_right=pts[CHEEK_RIGHT_IDX],
        eye_distance=max(eye_dist, 1.0),
    )


def _pose_from_points(pts: np.ndarray, visibility: np.ndarray) -> PoseLandmarks:
    def _maybe(idx: int) -> Optional[np.ndarray]:
        return pts[idx] if visibility[idx] > 0.3 else None

    return PoseLandmarks(
        points=pts,
        left_shoulder=pts[POSE_LEFT_SHOULDER],
        right_shoulder=pts[POSE_RIGHT_SHOULDER],
        left_hip=pts[POSE_LEFT_HIP],
        right_hip=pts[POSE_RIGHT_HIP],
        left_knee=_maybe(POSE_LEFT_KNEE),
        right_knee=_maybe(POSE_RIGHT_KNEE),
        visibility=visibility,
    )


# --------------------------------------------------------------------------- #
# Public functions — pick the right backend automatically.
# --------------------------------------------------------------------------- #


def detect_face(image_rgb: np.ndarray) -> Optional[FaceLandmarks]:
    try:
        if _has_solutions():
            return _detect_face_solutions(image_rgb)
        return _detect_face_tasks(image_rgb)
    except Exception:
        return None


def detect_pose(image_rgb: np.ndarray) -> Optional[PoseLandmarks]:
    try:
        if _has_solutions():
            return _detect_pose_solutions(image_rgb)
        return _detect_pose_tasks(image_rgb)
    except Exception:
        return None


def detect_person_mask(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    try:
        if _has_solutions():
            return _detect_mask_solutions(image_rgb)
        return _detect_mask_tasks(image_rgb)
    except Exception:
        return None
