import os
import json
import re
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from pipeline.bill_pipeline import BillIntelligencePipeline

app = Flask(
    __name__,
    template_folder=config.TEMPLATES_DIR,
    static_folder=config.STATIC_DIR
)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

os.makedirs(config.UPLOADS_DIR, exist_ok=True)
os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)

# Initialize pipeline
pipeline = BillIntelligencePipeline()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/annotate")
def annotate():
    return render_template("annotate.html")


@app.route("/review-crops")
def review_crops():
    return render_template("review_crops.html")


def _reviewed_record_path(source_kind: str, run_id: str, source_image_id: str, annotation_id: str) -> Path:
    reviewed_root = Path(config.BASE_DIR) / "annotations" / "reviewed"
    if source_kind == "manual":
        return reviewed_root / "manual" / f"{annotation_id}.json"
    return reviewed_root / "automatic" / run_id / source_image_id / f"{annotation_id}.json"


def _pending_review_items():
    annotation_root = Path(config.BASE_DIR) / "annotations"
    items = []
    manifest_path = Path(config.OCR_ANNOTATIONS_PATH)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("annotation_source") != "manual":
                    continue
                annotation_id = str(record.get("id", ""))
                review_path = _reviewed_record_path("manual", "", "", annotation_id)
                if not annotation_id:
                    continue
                reviewed = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
                if reviewed is not None and (
                    reviewed.get("review_status") != "APPROVED"
                    or reviewed.get("bill_group_id")
                ):
                    continue
                display_record = reviewed or record
                source_path = None
                source_filename = record.get("source_filename", "")
                for uploaded_path in Path(config.UPLOADS_DIR).iterdir():
                    if uploaded_path.is_file() and secure_filename(uploaded_path.name) == source_filename:
                        source_path = uploaded_path.relative_to(config.BASE_DIR).as_posix()
                        break
                items.append({
                    "annotation_id": annotation_id,
                    "source_kind": "manual",
                    "source_image": source_filename,
                    "bill_group_id": display_record.get("bill_group_id") or Path(source_filename).stem,
                    "region_type": display_record.get("field_name", ""),
                    "status": "GROUP_ID_REQUIRED" if reviewed is not None else record.get("status", "PENDING_REVIEW"),
                    "text": display_record.get("text", ""),
                    "translated_text": display_record.get("translated_text", ""),
                    "text_type": display_record.get("type", "printed"),
                    "language": display_record.get("language", ""),
                    "bbox_xywh": record.get("bbox"),
                    "image_width": record.get("image_width"),
                    "image_height": record.get("image_height"),
                    "crop_path": record.get("crop_path"),
                    "source_path": source_path,
                    "preview_path": None,
                })

    automatic_root = annotation_root / "automatic"
    if automatic_root.exists():
        for record_path in sorted(automatic_root.glob("*/records/*.json"), reverse=True):
            run_id = record_path.parents[1].name
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("alignment_method") != "sift_ransac_best_manual_reference_v2":
                continue
            for region in record.get("regions", []):
                annotation_id = str(region.get("id", ""))
                review_path = _reviewed_record_path(
                    "automatic", run_id, record.get("source_image_id", ""), annotation_id
                )
                if not annotation_id:
                    continue
                reviewed = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else None
                if reviewed is not None and (
                    reviewed.get("status") != "APPROVED"
                    or reviewed.get("bill_group_id")
                ):
                    continue
                items.append({
                    "annotation_id": annotation_id,
                    "source_kind": "automatic",
                    "run_id": run_id,
                    "source_image_id": record.get("source_image_id"),
                    "source_image": record.get("source_image", ""),
                    "bill_group_id": record.get("bill_group_id") or Path(record.get("source_image", "bill")).stem,
                    "region_type": (reviewed or region).get("region_type", ""),
                    "status": "GROUP_ID_REQUIRED" if reviewed is not None else region.get("status", "LOW_CONFIDENCE"),
                    "text": (reviewed or {}).get("text", ""),
                    "translated_text": (reviewed or {}).get("translated_text", ""),
                    "text_type": (reviewed or {}).get("text_type", "printed"),
                    "language": (reviewed or {}).get("language", ""),
                    "bbox_xyxy": region.get("bbox_xyxy"),
                    "image_width": record.get("image_width"),
                    "image_height": record.get("image_height"),
                    "crop_path": (reviewed or region).get("crop_path"),
                    "source_path": None,
                    "preview_path": record.get("preview_path"),
                })
    return items


@app.route("/api/review/queue", methods=["GET"])
def review_queue():
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(100, max(1, int(request.args.get("limit", 30))))
    except ValueError:
        return jsonify({"success": False, "error": "Offset and limit must be integers."}), 400
    items = _pending_review_items()
    return jsonify({
        "success": True,
        "items": items[offset:offset + limit],
        "offset": offset,
        "limit": limit,
        "pending_count": len(items),
    })


@app.route("/api/review/file", methods=["GET"])
def review_file():
    requested_path = request.args.get("path", "")
    base_dir = Path(config.BASE_DIR).resolve()
    candidate = (base_dir / requested_path).resolve()
    allowed_roots = [
        (base_dir / "annotations").resolve(),
        (base_dir / "data_generation" / "annotated_crops").resolve(),
        Path(config.UPLOADS_DIR).resolve(),
    ]
    if not candidate.is_file() or not any(root == candidate or root in candidate.parents for root in allowed_roots):
        return jsonify({"success": False, "error": "Review image was not found."}), 404
    return send_from_directory(str(candidate.parent), candidate.name)


@app.route("/api/review/decision", methods=["POST"])
def save_review_decision():
    payload = request.get_json(silent=True) or {}
    source_kind = payload.get("source_kind")
    annotation_id = str(payload.get("annotation_id", ""))
    decision = payload.get("decision")
    transcription = str(payload.get("text", "")).strip()
    error_category = str(payload.get("error_category", "")).strip()
    error_categories = {
        "WRONG_ALIGNMENT", "WRONG_CROP", "MISSING_REGION", "PARTIAL_CROP",
        "WRONG_REGION_TYPE", "BAD_IMAGE_QUALITY", "OCR_ERROR", "HANDWRITING_ERROR",
        "NUMBER_ERROR", "TOKENIZER_ERROR", "DECODING_ERROR", "TRANSLATION_ERROR",
        "JSON_MAPPING_ERROR", "OUT_OF_BOUNDS", "INVALID_CROP", "OTHER",
    }
    if source_kind not in {"manual", "automatic"} or not annotation_id:
        return jsonify({"success": False, "error": "A valid annotation is required."}), 400
    if not re.fullmatch(r"[a-f0-9]{32}", annotation_id):
        return jsonify({"success": False, "error": "Annotation identifier is invalid."}), 400
    if decision not in {"APPROVED", "REJECTED"}:
        return jsonify({"success": False, "error": "Decision must be APPROVED or REJECTED."}), 400
    if decision == "APPROVED" and not transcription:
        return jsonify({"success": False, "error": "Approved crops need a manually verified transcription."}), 400
    if decision == "REJECTED" and error_category not in error_categories:
        return jsonify({"success": False, "error": "Choose an error category for rejected crops."}), 400

    source_record = None
    run_id = ""
    source_image_id = ""
    if source_kind == "manual":
        manifest_path = Path(config.OCR_ANNOTATIONS_PATH)
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as manifest:
                for line in manifest:
                    if line.strip():
                        candidate = json.loads(line)
                        if candidate.get("id") == annotation_id and candidate.get("annotation_source") == "manual":
                            source_record = candidate
                            break
    else:
        run_id = str(payload.get("run_id", ""))
        source_image_id = str(payload.get("source_image_id", ""))
        if (
            not re.fullmatch(r"\d{8}T\d{6}Z", run_id)
            or not re.fullmatch(r"[a-f0-9]{32}", source_image_id)
        ):
            return jsonify({"success": False, "error": "Automatic annotation identifiers are invalid."}), 400
        automatic_path = (
            Path(config.BASE_DIR) / "annotations" / "automatic" / run_id
            / "records" / f"{source_image_id}.json"
        )
        if automatic_path.exists():
            candidate = json.loads(automatic_path.read_text(encoding="utf-8"))
            region = next((item for item in candidate.get("regions", []) if item.get("id") == annotation_id), None)
            if region is not None:
                source_record = {"image_record": candidate, "region": region}

    if source_record is None:
        return jsonify({"success": False, "error": "Annotation record was not found."}), 404
    crop_path = source_record.get("crop_path") if source_kind == "manual" else source_record["region"].get("crop_path")
    if decision == "APPROVED" and (not crop_path or not (Path(config.BASE_DIR) / crop_path).is_file()):
        return jsonify({"success": False, "error": "This crop cannot be approved because its image is missing."}), 400

    if source_kind == "manual":
        reviewed = dict(source_record)
        reviewed.update({
            "source_kind": "manual",
            "review_status": decision,
            "bill_group_id": str(payload.get("bill_group_id", source_record.get("bill_group_id", Path(source_record.get("source_filename", "bill")).stem))).strip(),
            "error_category": error_category if decision == "REJECTED" else "",
            "text": transcription if decision == "APPROVED" else "",
            "translated_text": str(payload.get("translated_text", source_record.get("translated_text", ""))).strip(),
            "type": payload.get("text_type", source_record.get("type", "printed")),
            "language": str(payload.get("language", source_record.get("language", ""))).strip(),
        })
        review_path = _reviewed_record_path("manual", "", "", annotation_id)
    else:
        image_record = source_record["image_record"]
        region = source_record["region"]
        reviewed = {
            "schema_version": 1,
            "source_kind": "automatic",
            "run_id": run_id,
            "source_image_id": source_image_id,
            "source_image": image_record.get("source_image"),
            "bill_group_id": str(payload.get("bill_group_id", image_record.get("bill_group_id", Path(image_record.get("source_image", "bill")).stem))).strip(),
            "annotation_id": annotation_id,
            "region_id": region.get("region_id"),
            "region_type": region.get("region_type"),
            "status": decision,
            "error_category": error_category if decision == "REJECTED" else "",
            "text": transcription if decision == "APPROVED" else "",
            "translated_text": str(payload.get("translated_text", "")).strip(),
            "text_type": payload.get("text_type", "printed"),
            "language": str(payload.get("language", "")).strip(),
            "crop_path": crop_path,
            "bbox_xyxy": region.get("bbox_xyxy"),
            "normalized_bbox_xyxy": region.get("normalized_bbox_xyxy"),
            "image_width": image_record.get("image_width"),
            "image_height": image_record.get("image_height"),
            "alignment": image_record.get("alignment"),
            "source_annotation_ids": region.get("source_annotation_ids", []),
        }
        review_path = _reviewed_record_path("automatic", run_id, source_image_id, annotation_id)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = review_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, review_path)
    return jsonify({"success": True, "decision": decision, "review_path": review_path.relative_to(config.BASE_DIR).as_posix()})


@app.route("/api/annotation-template", methods=["GET"])
def annotation_template():
    manifest_path = config.OCR_ANNOTATIONS_PATH
    if not os.path.exists(manifest_path):
        return jsonify({"success": True, "layout": []})

    layouts = {}
    with open(manifest_path, "r", encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("annotation_source") != "manual":
                continue
            bbox = record.get("bbox_normalized")
            if not bbox:
                width = record.get("image_width")
                height = record.get("image_height")
                pixel_bbox = record.get("bbox")
                if not width or not height or not pixel_bbox:
                    continue
                x, y, box_width, box_height = pixel_bbox
                bbox = [x / width, y / height, box_width / width, box_height / height]

            source_id = record.get("source_image")
            if not source_id:
                continue
            layout = layouts.setdefault(source_id, [])
            layout.append({
                "field_name": record.get("field_name", ""),
                "bbox_normalized": bbox,
                "type": record.get("type", "printed"),
                "language": record.get("language", ""),
            })

    layout = next(iter(layouts.values()), [])
    return jsonify({"success": True, "layout": layout})


@app.route("/api/uploads/list", methods=["GET"])
def list_uploads():
    uploads_dir = Path(config.UPLOADS_DIR)
    images = []
    if uploads_dir.exists():
        for file in sorted(uploads_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                images.append({
                    "filename": file.name,
                    "size_bytes": file.stat().st_size,
                    "modified": file.stat().st_mtime,
                })
    return jsonify({"success": True, "images": images})


@app.route("/api/uploads/raw/<path:filename>", methods=["GET"])
def get_uploaded_image(filename):
    safe_name = secure_filename(filename)
    upload_path = Path(config.UPLOADS_DIR) / safe_name
    if not upload_path.is_file():
        candidates = [p for p in Path(config.UPLOADS_DIR).iterdir() if p.name == filename]
        if candidates:
            upload_path = candidates[0]
        else:
            return jsonify({"success": False, "error": "Image not found."}), 404
    return send_from_directory(str(upload_path.parent), upload_path.name)


@app.route("/api/annotations/ink-tighten", methods=["POST"])
def ink_tighten():
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename", "")
    rotation = int(payload.get("rotation", 0)) % 360
    bbox = payload.get("bbox", [])
    if not bbox or len(bbox) != 4:
        return jsonify({"success": False, "error": "BBox [x, y, w, h] required."}), 400

    target_path = Path(config.UPLOADS_DIR) / secure_filename(filename)
    if not target_path.is_file():
        candidates = [p for p in Path(config.UPLOADS_DIR).iterdir() if p.name == filename]
        if candidates:
            target_path = candidates[0]
        else:
            return jsonify({"success": False, "error": "Image file not found."}), 404

    image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"success": False, "error": "Could not decode image."}), 400

    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img_h, img_w = image.shape[:2]
    x, y, w, h = [int(round(float(v))) for v in bbox]
    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))
    w = max(4, min(img_w - x, w))
    h = max(4, min(img_h - y, h))

    patch = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Otsu inverse threshold to find dark ink
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of ink
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({"success": True, "tightened_bbox": [x, y, w, h]})

    # Combine contours with significant area
    min_area = 4
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not significant:
        significant = contours

    all_pts = np.vstack(significant)
    bx, by, bw, bh = cv2.boundingRect(all_pts)
    # Add modest padding (2-4px)
    pad = 3
    new_x = max(0, min(img_w - 1, x + bx - pad))
    new_y = max(0, min(img_h - 1, y + by - pad))
    new_w = min(img_w - new_x, bw + 2 * pad)
    new_h = min(img_h - new_y, bh + 2 * pad)

    return jsonify({"success": True, "tightened_bbox": [new_x, new_y, new_w, new_h]})


@app.route("/api/annotations/save", methods=["POST"])
def save_annotations():
    uploaded_image = request.files.get("image")
    existing_filename = str(request.form.get("existing_filename", "")).strip()

    if not uploaded_image and not existing_filename:
        return jsonify({"success": False, "error": "Choose or upload an image first."}), 400

    try:
        annotations = json.loads(request.form.get("annotations", "[]"))
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "Annotation data is invalid."}), 400
    if not isinstance(annotations, list) or not annotations:
        return jsonify({"success": False, "error": "Draw and transcribe at least one crop."}), 400

    image = None
    original_name = ""
    if uploaded_image and uploaded_image.filename:
        original_name = secure_filename(uploaded_image.filename)
        image_bytes = uploaded_image.read()
        if len(image_bytes) > 25 * 1024 * 1024:
            return jsonify({"success": False, "error": "Image exceeds the 25 MB limit."}), 413
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            # Save raw upload to uploads directory
            dest_raw = Path(config.UPLOADS_DIR) / original_name
            cv2.imwrite(str(dest_raw), image)
    elif existing_filename:
        safe_name = secure_filename(existing_filename)
        target_path = Path(config.UPLOADS_DIR) / safe_name
        if not target_path.is_file():
            candidates = [p for p in Path(config.UPLOADS_DIR).iterdir() if p.name == existing_filename]
            if candidates:
                target_path = candidates[0]
        if target_path.is_file():
            original_name = target_path.name
            image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "error": "The image could not be loaded."}), 400

    # Apply rotation so all bounding boxes correspond to the upright image
    rotation = int(request.form.get("rotation", 0)) % 360
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save the upright image into uploads so template builders and alignment reference it correctly
    if rotation != 0:
        upright_name = f"upright_{original_name}"
        cv2.imwrite(str(Path(config.UPLOADS_DIR) / upright_name), image)
        reference_filename = upright_name
    else:
        reference_filename = original_name

    image_height, image_width = image.shape[:2]
    image_id = uuid.uuid4().hex
    bill_group_id = str(request.form.get("bill_group_id", "")).strip() or Path(reference_filename).stem
    crops_dir = os.path.join(config.BASE_DIR, "data_generation", "annotated_crops")
    os.makedirs(crops_dir, exist_ok=True)
    reviewed_manual_dir = Path(config.BASE_DIR) / "annotations" / "reviewed" / "manual"
    reviewed_manual_dir.mkdir(parents=True, exist_ok=True)

    records = []
    saved_crop_paths = []
    try:
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Each crop annotation must be an object.")
            field_name = str(annotation.get("field_name", "")).strip()
            text = str(annotation.get("text", "")).strip()
            text_type = str(annotation.get("type", "")).strip().lower()
            field_category = str(annotation.get("field_category", "")).strip()
            language = str(annotation.get("language", "")).strip()
            bbox = annotation.get("bbox")
            if not field_name or not text:
                raise ValueError("Every crop needs a field name and exact transcription.")
            if text_type not in {"printed", "handwritten", "digits"}:
                text_type = "printed"
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Each crop needs an image-space bounding box.")

            x, y, box_width, box_height = [int(round(float(value))) for value in bbox]
            x = max(0, min(image_width - 1, x))
            y = max(0, min(image_height - 1, y))
            box_width = max(1, min(image_width - x, box_width))
            box_height = max(1, min(image_height - y, box_height))

            crop = image[y:y + box_height, x:x + box_width]
            success, encoded_crop = cv2.imencode(".png", crop)
            if not success:
                raise ValueError("A crop could not be encoded.")

            annotation_id = uuid.uuid4().hex
            crop_filename = f"manual_{annotation_id}.png"
            crop_path = os.path.join(crops_dir, crop_filename)
            with open(crop_path, "wb") as crop_file:
                crop_file.write(encoded_crop.tobytes())
            saved_crop_paths.append(crop_path)

            record = {
                "id": annotation_id,
                "source_image": image_id,
                "bill_group_id": bill_group_id,
                "source_filename": reference_filename,
                "image_width": image_width,
                "image_height": image_height,
                "bbox": [x, y, box_width, box_height],
                "bbox_normalized": [
                    x / image_width,
                    y / image_height,
                    box_width / image_width,
                    box_height / image_height,
                ],
                "field_name": field_name,
                "field_category": field_category,
                "crop_path": os.path.relpath(crop_path, config.BASE_DIR).replace(os.sep, "/"),
                "text": text,
                "translated_text": str(annotation.get("translated_text", "")).strip(),
                "type": text_type,
                "language": language or ("te" if any(ord(c) >= 0x0c00 and ord(c) <= 0x0c7f for c in text) else "en"),
                "annotation_source": "manual",
                "status": "APPROVED",
                "rotation_applied": rotation,
            }
            records.append(record)

            # Persist directly into annotations/reviewed/manual/<id>.json as APPROVED
            reviewed_path = reviewed_manual_dir / f"{annotation_id}.json"
            reviewed_data = dict(record)
            reviewed_data["review_status"] = "APPROVED"
            reviewed_path.write_text(json.dumps(reviewed_data, ensure_ascii=False, indent=2), encoding="utf-8")

        with open(config.OCR_ANNOTATIONS_PATH, "a", encoding="utf-8") as manifest:
            for record in records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (ValueError, TypeError, OSError) as error:
        for crop_path in saved_crop_paths:
            if os.path.exists(crop_path):
                os.remove(crop_path)
        return jsonify({"success": False, "error": str(error)}), 400

    return jsonify({
        "success": True,
        "saved_crops": len(records),
        "image_width": image_width,
        "image_height": image_height,
        "reference_filename": reference_filename,
    })

@app.route("/api/process", methods=["POST"])
def process_upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    suffix = os.path.splitext(filename)[1]
    file_descriptor, save_path = tempfile.mkstemp(
        prefix="upload_", suffix=suffix, dir=config.UPLOADS_DIR
    )
    os.close(file_descriptor)
    try:
        file.save(save_path)
        rotation = int(request.form.get("rotation", 0))
        result = pipeline.process_image_path(save_path, manual_rotation=rotation)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except FileNotFoundError:
            pass

@app.route("/api/model-info", methods=["GET"])
def model_info():
    ocr_engine = pipeline.ocr_engine
    vocab_size = len(ocr_engine.vocab) if ocr_engine.vocab is not None else 0
    info = {
        "ocr_model": {
            "name": "CRNN (7-Layer CNN + BiLSTM + CTC Decoder)",
            "framework": "PyTorch",
            "loaded": ocr_engine.is_ready,
            "status": "Ready — Trained on printed bill crops" if ocr_engine.is_ready else "Not loaded",
            "architecture": "7-Layer CNN + 2-Layer Bidirectional LSTM + CTC Beam Search",
            "vocab_size": vocab_size,
            "training_source": "User manual & verified auto crops (no synthetic data)",
            "supported_languages": ["Telugu (te)", "English (en)", "Digits"],
            "input_resolution": f"{config.OCR_IMG_H}x{config.OCR_IMG_W}",
        },
        "device": str(config.DEVICE),
        "external_apis": "None — 100% local neural inference"
    }
    return jsonify({"success": True, "info": info})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "LangBill AI"})


@app.errorhandler(413)
def request_entity_too_large(error):
    max_mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({
        "success": False,
        "error": f"Uploaded file exceeds the maximum permitted size of {max_mb} MB."
    }), 413


@app.errorhandler(404)
def not_found_handler(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Requested API endpoint not found."}), 404
    return "<h1>404 Not Found</h1>", 404


@app.errorhandler(500)
def internal_error_handler(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500
    return "<h1>500 Internal Server Error</h1>", 500


if __name__ == "__main__":
    print(f"Starting LangBill AI Server on http://{config.HOST}:{config.PORT} (Debug={config.DEBUG}) ...")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
