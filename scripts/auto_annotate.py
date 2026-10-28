import argparse
import json
import math
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from scripts.region_alignment import (
    align_image_to_reference,
    resolve_uploaded_image,
)


def safe_stem(filename: str) -> str:
    return Path(secure_filename(filename)).stem or "bill"


def reference_polygon(normalized_bbox: List[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = normalized_bbox
    return np.float32([
        [x1 * width, y1 * height],
        [x2 * width, y1 * height],
        [x2 * width, y2 * height],
        [x1 * width, y2 * height],
    ])


def clip_bbox_xyxy(polygon: np.ndarray, width: int, height: int) -> List[int]:
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    return [
        int(np.clip(math.floor(x1), 0, width)),
        int(np.clip(math.floor(y1), 0, height)),
        int(np.clip(math.ceil(x2), 0, width)),
        int(np.clip(math.ceil(y2), 0, height)),
    ]


def warp_region(image: np.ndarray, target_polygon: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    output_width, output_height = output_size
    destination = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ])
    transform = cv2.getPerspectiveTransform(target_polygon.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def region_status(
    alignment_quality: str,
    template_stability: str,
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    crop: np.ndarray,
) -> str:
    if (
        np.any(polygon[:, 0] < 0)
        or np.any(polygon[:, 1] < 0)
        or np.any(polygon[:, 0] > image_width)
        or np.any(polygon[:, 1] > image_height)
    ):
        return "OUT_OF_BOUNDS"
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "INVALID_CROP"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if float(gray.std()) < 1.5:
        return "INVALID_CROP"
    if alignment_quality != "HIGH_CONFIDENCE" or template_stability != "STABLE":
        return "LOW_CONFIDENCE"
    return "PENDING_REVIEW"


def draw_preview(image: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
    preview = image.copy()
    colors = {
        "PENDING_REVIEW": (0, 190, 255),
        "LOW_CONFIDENCE": (0, 0, 255),
        "OUT_OF_BOUNDS": (255, 0, 255),
        "INVALID_CROP": (0, 0, 180),
    }
    scale = max(image.shape[1] / 1600.0, image.shape[0] / 1600.0, 1.0)
    for index, region in enumerate(regions, start=1):
        polygon = np.round(np.asarray(region["polygon_xy"], dtype=np.float32)).astype(np.int32)
        color = colors[region["status"]]
        cv2.polylines(preview, [polygon], True, color, max(2, round(2 * scale)))
        label = f"{index}: {region['region_type'][:24]} [{region['status']}]"
        origin_x, origin_y = polygon[0]
        cv2.putText(
            preview,
            label,
            (int(max(0, origin_x)), int(max(20 * scale, origin_y - 5 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * scale,
            color,
            max(1, round(scale)),
            cv2.LINE_AA,
        )
    return preview


def save_contact_sheet(regions: List[Dict[str, Any]], output_path: Path) -> None:
    tile_width, tile_height, columns = 300, 170, 4
    rows = max(1, math.ceil(len(regions) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, region in enumerate(regions):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_width, row * tile_height
        tile = sheet[y0:y0 + tile_height, x0:x0 + tile_width]
        crop_path = region.get("crop_path")
        crop = cv2.imread(str(BASE_DIR / crop_path)) if crop_path else None
        label = f"{region['region_id']} {region['status']}"
        cv2.putText(tile, label[:42], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
        if crop is not None:
            available_width, available_height = tile_width - 12, tile_height - 30
            ratio = min(available_width / crop.shape[1], available_height / crop.shape[0])
            resized = cv2.resize(
                crop,
                (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))),
                interpolation=cv2.INTER_AREA,
            )
            x = x0 + (tile_width - resized.shape[1]) // 2
            y = y0 + 24 + (available_height - resized.shape[0]) // 2
            sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def select_reference_alignment(
    image: np.ndarray,
    template: Dict[str, Any],
    reference_images: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    candidates = []
    orientations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for angle, rot_code in orientations:
        rot_img = image if rot_code is None else cv2.rotate(image, rot_code)
        for source in template["manual_source_images"]:
            filename = source["filename"]
            reference_image = reference_images.get(filename)
            if reference_image is None:
                continue
            registration = align_image_to_reference(rot_img, reference_image)
            if registration["homography"] is None:
                continue
            candidate = {
                **{key: value for key, value in registration.items() if key != "homography"},
                "homography": registration["homography"],
                "rotation_angle": angle,
                "matched_reference_filename": filename,
                "matched_reference_image_id": source["source_image_id"],
                "matched_reference_width": source["width"],
                "matched_reference_height": source["height"],
                "matched_reference_regions": source["regions"],
            }
            score = (
                registration["quality"] == "HIGH_CONFIDENCE",
                registration["frame_corner_fraction"],
                registration["inlier_ratio"],
                registration["inliers"],
                -(registration["median_reprojection_px"] or float("inf")),
            )
            candidates.append((score, candidate))

    if not candidates:
        return {
            "homography": None,
            "rotation_angle": 0,
            "quality": "LOW_CONFIDENCE",
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reprojection_px": None,
            "moving_coverage": 0.0,
            "reference_coverage": 0.0,
            "frame_corner_fraction": 0.0,
            "matched_reference_filename": None,
            "matched_reference_image_id": None,
            "matched_reference_width": None,
            "matched_reference_height": None,
            "matched_reference_regions": [],
        }
    return max(candidates, key=lambda item: item[0])[1]


def generate_for_image(
    image_path: Path,
    template: Dict[str, Any],
    reference_images: Dict[str, np.ndarray],
    output_root: Path,
) -> Dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    alignment = select_reference_alignment(image, template, reference_images)
    rot_angle = alignment.get("rotation_angle", 0)
    if rot_angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rot_angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rot_angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    homography = alignment["homography"]
    image_height, image_width = image.shape[:2]
    image_id = uuid.uuid4().hex
    image_dir = output_root / "crops" / image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    regions = []

    if homography is None:
        for index, template_region in enumerate(template["regions"], start=1):
            regions.append({
                "id": uuid.uuid4().hex,
                "region_id": template_region["region_key"],
                "region_type": ", ".join(template_region["region_types"]),
                "status": "LOW_CONFIDENCE",
                "reason": "No reliable projective alignment could be estimated.",
                "bbox_xyxy": None,
                "normalized_bbox_xyxy": None,
                "source_annotation_ids": template_region["source_annotation_ids"],
                "crop_path": None,
            })
    else:
        reference_width = alignment["matched_reference_width"]
        reference_height = alignment["matched_reference_height"]
        reference_to_image = np.linalg.inv(homography)
        for index, template_region in enumerate(alignment["matched_reference_regions"], start=1):
            reference_points = reference_polygon(
                template_region["normalized_bbox_xyxy"], reference_width, reference_height
            ).reshape(-1, 1, 2)
            polygon = cv2.perspectiveTransform(reference_points, reference_to_image).reshape(-1, 2)
            bbox = clip_bbox_xyxy(polygon, image_width, image_height)
            output_width = max(4, int(template_region.get("crop_width") or template_region["bbox_xywh"][2]))
            output_height = max(4, int(template_region.get("crop_height") or template_region["bbox_xywh"][3]))
            crop = warp_region(image, polygon, (output_width, output_height))
            status = region_status(
                alignment["quality"],
                template_region["stability"],
                polygon,
                image_width,
                image_height,
                crop,
            )
            crop_name = f"region_{index:03d}.png"
            crop_path = image_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            x1, y1, x2, y2 = bbox
            regions.append({
                "id": uuid.uuid4().hex,
                "region_id": template_region["region_key"],
                "region_type": template_region["region_type"],
                "status": status,
                "reason": (
                    "Projective alignment or source template requires review."
                    if status == "LOW_CONFIDENCE" else "Generated crop requires visual approval."
                ),
                "bbox_xyxy": bbox,
                "normalized_bbox_xyxy": [
                    x1 / image_width,
                    y1 / image_height,
                    x2 / image_width,
                    y2 / image_height,
                ],
                "polygon_xy": polygon.tolist(),
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "crop_path": crop_path.relative_to(BASE_DIR).as_posix(),
                "template_stability": template_region["stability"],
                "source_annotation_ids": [template_region["annotation_id"]],
                "source_annotation_image_id": alignment["matched_reference_image_id"],
                "source_filename": alignment["matched_reference_filename"],
                "source_bbox_xywh": template_region["bbox_xywh"],
            })

    preview_path = output_root / "previews" / f"{image_id}_preview.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), draw_preview(image, regions))
    contact_path = output_root / "contact_sheets" / f"{image_id}_crops.jpg"
    save_contact_sheet(regions, contact_path)

    record = {
        "schema_version": 1,
        "alignment_method": "sift_ransac_best_manual_reference_v2",
        "annotation_source": "automatic_alignment",
        "source_image_id": image_id,
        "source_image": image_path.name,
        "bill_group_id": image_path.stem,
        "image_width": image_width,
        "image_height": image_height,
        "reference_image": template["reference_filename"],
        "alignment": {
            key: value for key, value in alignment.items() if key != "homography"
        },
        "status": "PENDING_REVIEW",
        "preview_path": preview_path.relative_to(BASE_DIR).as_posix(),
        "contact_sheet_path": contact_path.relative_to(BASE_DIR).as_posix(),
        "regions": regions,
    }
    annotations_dir = output_root / "records"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    record_path = annotations_dir / f"{image_id}.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer manual region templates to real images using SIFT features and RANSAC homographies."
    )
    parser.add_argument("--input-dir", type=Path, default=Path(config.UPLOADS_DIR))
    parser.add_argument(
        "--template",
        type=Path,
        default=BASE_DIR / "annotations" / "templates" / "region_templates.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Automatic run directory. Defaults to annotations/automatic/<UTC timestamp>.",
    )
    args = parser.parse_args()
    if not args.template.exists():
        raise FileNotFoundError("Build the region template first with scripts/build_region_templates.py.")
    template = json.loads(args.template.read_text(encoding="utf-8"))
    reference_images = {}
    for source in template["manual_source_images"]:
        reference_path = resolve_uploaded_image(args.input_dir, source["filename"])
        if reference_path is None:
            continue
        reference_image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference_image is not None:
            reference_images[source["filename"]] = reference_image
    if len(reference_images) < 2:
        raise ValueError("At least two manually annotated source images must be available for alignment.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_dir or BASE_DIR / "annotations" / "automatic" / run_id
    if output_root.exists():
        raise FileExistsError(f"Automatic run directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manual_filenames = {secure_filename(item["filename"]) for item in template["manual_source_images"]}
    image_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        and secure_filename(path.name) not in manual_filenames
    )
    summary = Counter()
    for image_path in image_paths:
        record = generate_for_image(image_path, template, reference_images, output_root)
        counts = Counter(region["status"] for region in record["regions"])
        summary.update(counts)
        print(
            f"{image_path.name}: alignment={record['alignment']['quality']} "
            f"inliers={record['alignment']['inliers']} "
            f"median_error={record['alignment']['median_reprojection_px']!s} "
            f"regions={dict(counts)} preview={record['preview_path']}"
        )
    run_summary = {
        "run_id": run_id,
        "alignment_method": "sift_ransac_best_manual_reference_v2",
        "images_processed": len(image_paths),
        "status_counts": dict(summary),
        "training_eligible": 0,
        "note": "Automatic crops remain review proposals. No text labels are assigned.",
    }
    (output_root / "summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRun summary: {json.dumps(run_summary, ensure_ascii=False)}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
