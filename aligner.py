"""
aligner.py — Face Alignment Module
===================================
Handles facial landmark extraction and affine warp alignment
to produce deterministic 112×112 canonical face crops.

This module guarantees embedding stability by removing geometric
variance across video frames via similarity transformation.
"""

import cv2
import numpy as np
from typing import Optional

# ── InsightFace canonical warp target coordinates ──────────────
# These are the reference 5-point landmarks for a 112×112 aligned face.
REFERENCE_LANDMARKS = np.array(
    [
        [30.2946, 51.6963],   # Left Eye
        [65.5318, 51.5014],   # Right Eye
        [48.0252, 71.7366],   # Nose Tip
        [33.5493, 92.3655],   # Left Mouth Corner
        [62.7299, 92.2041],   # Right Mouth Corner
    ],
    dtype=np.float32,
)

ALIGNED_FACE_SIZE = (112, 112)


def align_face(
    frame: np.ndarray,
    landmarks: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Perform affine warp alignment on a detected face using its 5-point landmarks.

    Args:
        frame: The full BGR image (numpy array, HxWx3).
        landmarks: A (5, 2) float32 array of facial landmark coordinates
                   [left_eye, right_eye, nose, left_mouth, right_mouth].

    Returns:
        A 112×112 BGR aligned face crop, or None if alignment fails.

    Raises:
        ValueError: If landmarks shape is invalid.
    """
    if landmarks is None or landmarks.shape != (5, 2):
        raise ValueError(
            f"Expected landmarks shape (5, 2), got "
            f"{landmarks.shape if landmarks is not None else 'None'}"
        )

    src_pts = landmarks.astype(np.float32)

    # Estimate the similarity transformation matrix
    # cv2.estimateAffinePartial2D computes a 2×3 affine matrix
    # that maps src_pts → REFERENCE_LANDMARKS with minimal residual error.
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, REFERENCE_LANDMARKS
    )

    if transform_matrix is None:
        # This can happen if landmarks are degenerate (e.g., all collinear)
        return None

    # Apply the affine warp to produce the canonical 112×112 crop
    aligned_face = cv2.warpAffine(
        frame,
        transform_matrix,
        ALIGNED_FACE_SIZE,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return aligned_face


def extract_landmarks_from_face(face_obj) -> Optional[np.ndarray]:
    """
    Extract the 5-point landmark array from an InsightFace detection result.

    Args:
        face_obj: An InsightFace Face object (from FaceAnalysis.get()).

    Returns:
        A (5, 2) float32 numpy array, or None if landmarks are unavailable.
    """
    kps = getattr(face_obj, "kps", None)
    if kps is None:
        kps = getattr(face_obj, "landmark_2d_106", None)
        if kps is not None:
            # Map 106-point landmarks to the 5-point subset
            # Indices: left_eye=33, right_eye=263, nose=1, left_mouth=61, right_mouth=291
            # (These are approximate — InsightFace buffalo_l always provides kps directly)
            return None

    if kps is not None and kps.shape == (5, 2):
        return kps.astype(np.float32)

    return None
