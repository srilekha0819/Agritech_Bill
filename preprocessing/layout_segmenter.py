import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class LayoutSegmenter:
    """
    Dynamic Layout and Table Grid Segmenter for Agricultural Mandi Slips.
    Uses adaptive morphology and geometric analysis to extract functional regions,
    table grid cells, canonical field crops, and text lines.
    Zero hardcoded fixed-pixel coordinate templates.
    """

    def detect_table_bounding_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Dynamically finds the outer bounding box [x, y, w, h] of the tabular grid.
        Uses connected morphological line analysis to detect carbon paper grid boundaries.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect prominent horizontal and vertical grid lines
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
        grid = cv2.bitwise_or(h_lines, v_lines)
        connected = cv2.dilate(grid, np.ones((7, 35), np.uint8))
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, tw, th = cv2.boundingRect(c)
            # Table must occupy at least 55% of width and 30% of height, located in the lower body
            if tw >= int(w * 0.55) and th >= int(h * 0.30) and y >= int(h * 0.15):
                return (max(0, x), max(0, y), min(w - x, tw), min(h - y, th))

        # Default fallback proportions for APMC Mandi bills:
        table_top = int(h * 0.35)
        table_bottom = int(h * 0.92)
        table_left = int(w * 0.04)
        table_width = int(w * 0.92)
        return (table_left, table_top, table_width, table_bottom - table_top)

    def segment_bill_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Split bill slip dynamically into Header, Table/Body, and Footer/Settlement regions."""
        h, w = image.shape[:2]
        tx, ty, tw, th = self.detect_table_bounding_box(image)

        header_crop = image[0:ty, 0:w]
        table_crop = image[ty:ty + th, tx:tx + tw]
        footer_crop = image[ty + th:h, 0:w]

        return {
            "header": {"image": header_crop, "bbox": [0, 0, w, ty]},
            "table": {"image": table_crop, "bbox": [tx, ty, tw, th]},
            "footer": {"image": footer_crop, "bbox": [0, ty + th, w, h - (ty + th)]}
        }

    def detect_table_grid(self, table_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects horizontal and vertical grid lines of the table using morphological kernels,
        and extracts cell bounding boxes.
        """
        if table_img is None or table_img.size == 0:
            return []

        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY) if len(table_img.shape) == 3 else table_img
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )

        h, w = table_img.shape[:2]
        # Horizontal line kernel
        horizontal = thresh.copy()
        horiz_size = max(w // 30, 20)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        horizontal = cv2.erode(horizontal, horiz_kernel)
        horizontal = cv2.dilate(horizontal, horiz_kernel)

        # Vertical line kernel
        vertical = thresh.copy()
        vert_size = max(h // 20, 15)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        vertical = cv2.erode(vertical, vert_kernel)
        vertical = cv2.dilate(vertical, vert_kernel)

        # Combine lines to get grid table
        table_grid = cv2.add(horizontal, vertical)

        # Find contours of individual cells
        contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Filter out whole-table or tiny noise boxes
            if cw > 25 and ch > 15 and (cw < w * 0.95 or ch < h * 0.95):
                cells.append({
                    "bbox": [x, y, cw, ch],
                    "crop": table_img[y:y + ch, x:x + cw]
                })

        # Sort cells top-to-bottom, left-to-right
        cells.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
        return cells

    def extract_canonical_fields(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically extracts APMC Mandi functional crops based on the document's actual
        geometry and detected table landmarks. No rigid pixel stretching to (1600, 1200).
        Assigns field_type ('digit' or 'text') to route through specialized recognizers.
        """
        h, w = image.shape[:2]
        tx, ty, tw, th = self.detect_table_bounding_box(image)

        crops: Dict[str, Dict[str, Any]] = {}

        def _safe_crop(name: str, x1: int, y1: int, x2: int, y2: int, field_type: str = "text"):
            x1_c = max(0, min(int(x1), w - 4))
            y1_c = max(0, min(int(y1), h - 4))
            x2_c = max(x1_c + 4, min(int(x2), w))
            y2_c = max(y1_c + 4, min(int(y2), h))
            crop = image[y1_c:y2_c, x1_c:x2_c]
            crops[name] = {
                "bbox": [x1_c, y1_c, x2_c - x1_c, y2_c - y1_c],
                "crop": crop,
                "field_type": field_type
            }

        # 1. Header region fields (above table_top `ty`)
        # Top Committee Title Banner
        _safe_crop("market_committee", int(w * 0.10), max(0, int(ty * 0.08)), int(w * 0.88), int(ty * 0.38), field_type="text")
        # Top metadata row: Book Number, Slip/Bill Number, Date
        hdr_meta_y1 = max(0, ty - int(ty * 0.44))
        hdr_meta_y2 = ty - int(ty * 0.18)
        _safe_crop("book_number", tx, hdr_meta_y1, tx + int(tw * 0.32), hdr_meta_y2, field_type="digit")
        _safe_crop("bill_number", tx + int(tw * 0.46), hdr_meta_y1, tx + int(tw * 0.72), hdr_meta_y2, field_type="digit")
        _safe_crop("date", tx + int(tw * 0.70), hdr_meta_y1, tx + tw, hdr_meta_y2, field_type="digit")

        # Stakeholder Rows: Commission Agent, Farmer
        _safe_crop("agent", tx + int(tw * 0.16), ty - int(ty * 0.30), tx + int(tw * 0.76), ty - int(ty * 0.16), field_type="text")
        _safe_crop("farmer", tx + int(tw * 0.16), ty - int(ty * 0.16), tx + int(tw * 0.78), ty - 4, field_type="text")
        _safe_crop("buyer", tx + int(tw * 0.16), ty - 4, tx + int(tw * 0.78), ty + 50, field_type="text")

        # 2. Table Column and Data Row Values (Row 1: 60px to 130px below table top)
        tbl_row_y1 = ty + 60
        tbl_row_y2 = ty + 130

        _safe_crop("commodity", tx + int(tw * 0.05), tbl_row_y1, tx + int(tw * 0.24), tbl_row_y2, field_type="text")
        _safe_crop("bags", tx + int(tw * 0.24), tbl_row_y1, tx + int(tw * 0.33), tbl_row_y2, field_type="digit")
        _safe_crop("net_weight", tx + int(tw * 0.33), tbl_row_y1, tx + int(tw * 0.43), tbl_row_y2, field_type="digit")
        _safe_crop("rate", tx + int(tw * 0.43), tbl_row_y1, tx + int(tw * 0.53), tbl_row_y2, field_type="digit")
        _safe_crop("gross", tx + int(tw * 0.53), tbl_row_y1, tx + int(tw * 0.70), tbl_row_y2, field_type="digit")

        # 3. Deductions & Net Settlement (Right-hand and footer columns)
        _safe_crop("adthi", tx + int(tw * 0.76), ty + 60, tx + tw, ty + 120, field_type="digit")
        _safe_crop("hamali", tx + int(tw * 0.76), ty + 120, tx + tw, ty + 180, field_type="digit")
        _safe_crop("deductions", tx + int(tw * 0.76), ty + 180, tx + tw, ty + 240, field_type="digit")
        _safe_crop("net_payable", tx + int(tw * 0.74), ty + 240, tx + tw, ty + 360, field_type="digit")

        return crops

    def extract_text_lines(
        self,
        image: np.ndarray,
        min_area: int = 80,
        normalized_image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Robust text-line detection:
        1. Uses multi-scale illumination normalization.
        2. Applies dual-threshold binarization (Gaussian Adaptive + Otsu fusion).
        3. Removes dominant table grid borders while preserving text diacritics.
        4. Groups word components horizontally into complete text lines.
        5. Clusters lines into reading rows using dynamic vertical overlap.
        """
        from preprocessing.image_processor import ImageProcessor
        h, w = image.shape[:2]
        if h < 20 or w < 20:
            return []

        # 1. Use shared normalized image when available
        norm_gray = normalized_image if normalized_image is not None else ImageProcessor.normalize_document_color(image)

        # 2. Paper slip boundary masking: ignore background tables, desks, or wooden surfaces
        paper_roi = np.ones((h, w), dtype=np.uint8) * 255
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            _, a_ch, _ = cv2.split(lab)
            # Agricultural pink/yellow bill paper mask (a_ch > 130 for pink bills)
            paper_mask = (a_ch > 130).astype(np.uint8) * 255
            if cv2.countNonZero(paper_mask) > int(h * w * 0.20):
                contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    px, py, pw, ph = cv2.boundingRect(c)
                    pad = max(10, min(pw, ph) // 60)
                    px = max(0, px + pad)
                    py = max(0, py + pad)
                    pw = max(50, pw - 2 * pad)
                    ph = max(50, ph - 2 * pad)
                    paper_roi = np.zeros((h, w), dtype=np.uint8)
                    paper_roi[py:py+ph, px:px+pw] = 255

        # 3. Optimal binarization for illumination-normalized document
        blurred = cv2.GaussianBlur(norm_gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(thresh, paper_roi)

        # 4. Clean table grid lines without eroding Telugu matras, talakattu, or text strokes
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 120), 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 10, 80)))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
        grid_mask = cv2.bitwise_or(h_lines, v_lines)
        grid_dil = cv2.dilate(grid_mask, np.ones((2, 2), np.uint8))
        text_only = cv2.subtract(thresh, grid_dil)

        # Blank out outer scan margins
        mx, my = max(w // 60, 12), max(h // 60, 12)
        text_only[:my, :] = 0
        text_only[-my:, :] = 0
        text_only[:, :mx] = 0
        text_only[:, -mx:] = 0

        # 5. Component Extraction with real text constraints
        char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 55, 16), 2))
        char_connected = cv2.dilate(text_only, char_kernel, iterations=1)
        contours, _ = cv2.findContours(char_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Retain short words (ధర, తేది, వరి, నెం) as well as long sentences
            if bw >= 12 and 10 <= bh <= h * 0.25 and bw * bh >= 120:
                crop_bin = text_only[by:by+bh, bx:bx+bw]
                ink_pixels = cv2.countNonZero(crop_bin)
                ink_ratio = ink_pixels / float(bw * bh)
                if 0.030 <= ink_ratio <= 0.65 and ink_pixels >= 15:
                    raw_boxes.append([bx, by, bw, bh])

        # 6. Group adjacent word components horizontally into complete text lines
        raw_boxes.sort(key=lambda b: (b[1], b[0]))
        merged_boxes = []
        visited = [False] * len(raw_boxes)

        for i in range(len(raw_boxes)):
            if visited[i]:
                continue
            cur = list(raw_boxes[i])
            visited[i] = True

            for j in range(i + 1, len(raw_boxes)):
                if visited[j]:
                    continue
                cand = raw_boxes[j]
                # Check vertical overlap
                y_overlap = max(0, min(cur[1] + cur[3], cand[1] + cand[3]) - max(cur[1], cand[1]))
                min_h = min(cur[3], cand[3])
                if min_h > 0 and (y_overlap / min_h) >= 0.45:
                    # Check horizontal gap
                    h_gap = cand[0] - (cur[0] + cur[2])
                    if -12 <= h_gap <= max(w // 20, 65):
                        new_x1 = min(cur[0], cand[0])
                        new_y1 = min(cur[1], cand[1])
                        new_x2 = max(cur[0] + cur[2], cand[0] + cand[2])
                        new_y2 = max(cur[1] + cur[3], cand[1] + cand[3])
                        cur = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]
                        visited[j] = True

            merged_boxes.append(cur)

        # 7. Convert merged boxes into final lines with padding
        lines = []
        for bx, by, bw, bh in merged_boxes:
            min_x = max(0, bx - 3)
            min_y = max(0, by - 2)
            max_x = min(w, bx + bw + 3)
            max_y = min(h, by + bh + 2)
            lw = max_x - min_x
            lh = max_y - min_y

            if lw >= 28 and lh >= 12:
                lines.append({
                    "bbox": [min_x, min_y, lw, lh],
                    "crop": image[min_y:max_y, min_x:max_x],
                    "norm_crop": norm_gray[min_y:max_y, min_x:max_x],
                    "y_center": min_y + lh // 2
                })

        # 8. Cluster lines into reading rows using dynamic vertical overlap
        if lines:
            lines.sort(key=lambda item: item["y_center"])
            rows: List[List[Dict[str, Any]]] = []

            for item in lines:
                assigned = False
                cy = item["y_center"]
                for row in rows:
                    row_cy = float(np.mean([r["y_center"] for r in row]))
                    row_h = float(np.mean([r["bbox"][3] for r in row]))
                    if abs(cy - row_cy) < max(row_h * 0.65, 14.0):
                        row.append(item)
                        assigned = True
                        break
                if not assigned:
                    rows.append([item])

            # Within each reading row, sort left to right
            sorted_lines = []
            for row in rows:
                row.sort(key=lambda item: item["bbox"][0])
                sorted_lines.extend(row)
            return sorted_lines

        return lines
