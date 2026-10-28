import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from scripts.region_alignment import box_iou_xyxy, resolve_uploaded_image


def normalize_region_name(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).split())


def build_templates(manifest_path: Path, upload_dir: Path, output_path: Path) -> Dict[str, Any]:
    grouped = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {manifest_path}:{line_number}: {error}") from error
            if record.get("annotation_source") == "manual":
                grouped[str(record["source_image"])].append(record)
    if len(grouped) < 2:
        raise ValueError("At least two manually annotated source images are required to assess transfer.")

    source_images = []
    region_samples = defaultdict(list)
    for source_id, records in grouped.items():
        filename = records[0].get("source_filename", "")
        image_path = resolve_uploaded_image(upload_dir, filename)
        if image_path is None:
            raise FileNotFoundError(f"Could not resolve source image for manual annotation: {filename}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode source image: {image_path}")
        image_height, image_width = image.shape[:2]
        source_regions = []
        for record in records:
            field_name = str(record.get("field_name", "")).strip()
            bbox = record.get("bbox")
            if not field_name or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = [float(value) for value in bbox]
            if width <= 0 or height <= 0:
                continue
            normalized = [
                x / image_width,
                y / image_height,
                (x + width) / image_width,
                (y + height) / image_height,
            ]
            crop_path = BASE_DIR / str(record.get("crop_path", ""))
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR) if crop_path.is_file() else None
            crop_width = int(crop.shape[1]) if crop is not None else None
            crop_height = int(crop.shape[0]) if crop is not None else None
            sample = {
                "region_key": normalize_region_name(field_name),
                "region_type": field_name,
                "annotation_id": record.get("id"),
                "bbox_xywh": [int(x), int(y), int(width), int(height)],
                "normalized_bbox_xyxy": normalized,
                "crop_path": record.get("crop_path"),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "text_type": record.get("type", "unknown"),
                "language": record.get("language", ""),
            }
            source_regions.append(sample)
            region_samples[sample["region_key"]].append({
                **sample,
                "source_image_id": source_id,
                "source_filename": filename,
            })
        source_images.append({
            "source_image_id": source_id,
            "filename": filename,
            "width": image_width,
            "height": image_height,
            "aspect_ratio": image_width / image_height,
            "regions": source_regions,
        })

    stability_by_key = {}
    region_summaries = []
    for key, samples in region_samples.items():
        boxes = np.asarray([sample["normalized_bbox_xyxy"] for sample in samples], dtype=np.float64)
        source_ids = {sample["source_image_id"] for sample in samples}
        pairwise_ious = [
            box_iou_xyxy(boxes[i].tolist(), boxes[j].tolist())
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
            if samples[i]["source_image_id"] != samples[j]["source_image_id"]
        ]
        median_iou = float(np.median(pairwise_ious)) if pairwise_ious else 0.0
        stable = len(source_ids) >= 2 and median_iou >= 0.60
        stability_by_key[key] = "STABLE" if stable else "LOW_CONFIDENCE"
        coordinate_std = np.std(boxes, axis=0)
        summary = {
            "region_key": key,
            "region_types": sorted({sample["region_type"] for sample in samples}),
            "source_image_count": len(source_ids),
            "annotation_count": len(samples),
            "normalized_bbox_mean_xyxy": np.mean(boxes, axis=0).tolist(),
            "normalized_bbox_min_xyxy": np.min(boxes, axis=0).tolist(),
            "normalized_bbox_max_xyxy": np.max(boxes, axis=0).tolist(),
            "normalized_bbox_std_xyxy": coordinate_std.tolist(),
            "median_pairwise_iou": median_iou,
            "stability": stability_by_key[key],
            "source_annotation_ids": [sample["annotation_id"] for sample in samples],
        }
        region_summaries.append(summary)

    for source in source_images:
        for region in source["regions"]:
            region["stability"] = stability_by_key[region["region_key"]]

    reference = source_images[0]
    template = {
        "schema_version": 2,
        "coordinate_space": "normalized_xyxy per manual source image; automatic transfer uses direct alignment to the selected source image",
        "reference_image_id": reference["source_image_id"],
        "reference_filename": reference["filename"],
        "manual_source_images": source_images,
        "regions": region_summaries,
        "transfer_warning": "Do not reuse boxes from one source unless the new image aligns strongly to that specific source and the region stability permits transfer.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Build region templates directly from unchanged manual annotations.")
    parser.add_argument("--manifest", type=Path, default=Path(config.OCR_ANNOTATIONS_PATH))
    parser.add_argument("--upload-dir", type=Path, default=Path(config.UPLOADS_DIR))
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "annotations" / "templates" / "region_templates.json",
    )
    args = parser.parse_args()
    template = build_templates(args.manifest, args.upload_dir, args.output)
    print(f"Manual reference images: {len(template['manual_source_images'])}")
    for region in template["regions"]:
        types_str = ', '.join(region['region_types']).encode('ascii', errors='replace').decode('ascii')
        print(
            f"{types_str}: sources={region['source_image_count']} "
            f"mean_xyxy={[round(value, 4) for value in region['normalized_bbox_mean_xyxy']]} "
            f"std={[round(value, 4) for value in region['normalized_bbox_std_xyxy']]} "
            f"median_iou={region['median_pairwise_iou']:.3f} {region['stability']}"
        )
    print(f"Template written to {args.output}")


if __name__ == "__main__":
    main()
