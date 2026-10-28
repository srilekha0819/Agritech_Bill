from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from werkzeug.utils import secure_filename


MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
MAX_MEDIAN_REPROJECTION_PX = 4.0
MIN_FEATURE_COVERAGE = 0.05
MIN_FRAME_CORNER_FRACTION = 0.75


def resolve_uploaded_image(upload_dir: Path, filename: str) -> Optional[Path]:
    safe_filename = secure_filename(filename)
    for path in upload_dir.iterdir():
        if path.is_file() and secure_filename(path.name) == safe_filename:
            return path
    return None


def image_features(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors, norm


def align_image_to_reference(
    moving_image: np.ndarray,
    reference_image: np.ndarray,
    ratio_threshold: float = 0.72,
    ransac_threshold_px: float = 5.0,
) -> Dict[str, Any]:
    """Estimate a projective transform from moving-image pixels into reference pixels."""
    moving_keypoints, moving_descriptors, norm = image_features(moving_image)
    reference_keypoints, reference_descriptors, reference_norm = image_features(reference_image)
    if moving_descriptors is None or reference_descriptors is None or norm != reference_norm:
        return {
            "homography": None,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_px": None,
            "moving_coverage": 0.0,
            "reference_coverage": 0.0,
            "frame_corner_fraction": 0.0,
            "quality": "LOW_CONFIDENCE",
        }

    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(moving_descriptors, reference_descriptors, k=2)
    good_matches = [
        first for first, second in pairs
        if first.distance < ratio_threshold * second.distance
    ]
    if len(good_matches) < 4:
        return {
            "homography": None,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_px": None,
            "moving_coverage": 0.0,
            "reference_coverage": 0.0,
            "frame_corner_fraction": 0.0,
            "quality": "LOW_CONFIDENCE",
        }

    moving_points = np.float32([
        moving_keypoints[match.queryIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    reference_points = np.float32([
        reference_keypoints[match.trainIdx].pt for match in good_matches
    ]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        moving_points, reference_points, cv2.RANSAC, ransac_threshold_px
    )
    if homography is None or inlier_mask is None:
        return {
            "homography": None,
            "good_matches": len(good_matches),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_px": None,
            "moving_coverage": 0.0,
            "reference_coverage": 0.0,
            "frame_corner_fraction": 0.0,
            "quality": "LOW_CONFIDENCE",
        }

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(len(good_matches), 1)
    projected_points = cv2.perspectiveTransform(moving_points, homography)
    errors = np.linalg.norm(
        (projected_points - reference_points).reshape(-1, 2), axis=1
    )
    median_error = float(np.median(errors[inliers])) if inlier_count else float("inf")
    moving_height, moving_width = moving_image.shape[:2]
    reference_height, reference_width = reference_image.shape[:2]
    moving_hull = cv2.convexHull(moving_points[inliers].reshape(-1, 2)) if inlier_count >= 3 else None
    reference_hull = cv2.convexHull(reference_points[inliers].reshape(-1, 2)) if inlier_count >= 3 else None
    moving_coverage = (
        float(cv2.contourArea(moving_hull)) / max(moving_width * moving_height, 1)
        if moving_hull is not None else 0.0
    )
    reference_coverage = (
        float(cv2.contourArea(reference_hull)) / max(reference_width * reference_height, 1)
        if reference_hull is not None else 0.0
    )
    moving_corners = np.float32([
        [0, 0],
        [moving_width, 0],
        [moving_width, moving_height],
        [0, moving_height],
    ]).reshape(-1, 1, 2)
    mapped_corners = cv2.perspectiveTransform(moving_corners, homography).reshape(-1, 2)
    frame_corner_fraction = float(np.mean(
        (mapped_corners[:, 0] >= -0.1 * reference_width)
        & (mapped_corners[:, 0] <= 1.1 * reference_width)
        & (mapped_corners[:, 1] >= -0.1 * reference_height)
        & (mapped_corners[:, 1] <= 1.1 * reference_height)
    ))
    quality = "HIGH_CONFIDENCE"
    if (
        inlier_count < MIN_INLIERS
        or inlier_ratio < MIN_INLIER_RATIO
        or median_error > MAX_MEDIAN_REPROJECTION_PX
        or min(moving_coverage, reference_coverage) < MIN_FEATURE_COVERAGE
        or frame_corner_fraction < MIN_FRAME_CORNER_FRACTION
    ):
        quality = "LOW_CONFIDENCE"

    return {
        "homography": homography,
        "good_matches": len(good_matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_px": median_error,
        "moving_coverage": moving_coverage,
        "reference_coverage": reference_coverage,
        "frame_corner_fraction": frame_corner_fraction,
        "quality": quality,
    }


def transform_bbox_xywh(
    bbox: List[float],
    homography: np.ndarray,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform an xywh box to an xyxy box and its projective quadrilateral."""
    x, y, width, height = [float(value) for value in bbox]
    corners = np.float32([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]).reshape(-1, 1, 2)
    polygon = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    box = np.array([
        np.clip(x_min, 0, target_width),
        np.clip(y_min, 0, target_height),
        np.clip(x_max, 0, target_width),
        np.clip(y_max, 0, target_height),
    ], dtype=np.float64)
    return box, polygon.astype(np.float64)


def box_iou_xyxy(first: List[float], second: List[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(area_first + area_second - intersection, 1e-9)

