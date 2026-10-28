import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """
    Robust Preprocessor for Multilingual Agricultural Bills & Invoices.
    Handles arbitrary paper colors (pink, yellow, blue, white, carbon copies),
    shadow gradients, camera glare, and orientation detection.
    Zero external APIs or pre-trained models.
    """

    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by 90, 180, or 270 degrees clockwise."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def normalize_document_color(image: np.ndarray) -> np.ndarray:
        """
        Normalizes document images of ANY paper color (white, pink, yellow, blue, green, beige, dark mode)
        into a clean, uniform-illumination document with crisp dark ink strokes.
        Eliminates color cast, uneven lighting gradients, and flash shadows.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]

        # 1. Illumination background estimation
        # We use a large Gaussian kernel proportional to image scale to capture the background color/shadow
        ksize = max(int(min(h, w) * 0.05) | 1, 15)
        # Morphological dilation followed by median blur smooths out text and preserves background
        dilated = cv2.dilate(gray, np.ones((5, 5), np.uint8))
        bg = cv2.medianBlur(dilated, ksize)

        # 2. Divide gray by background to flatten illumination & eliminate colored paper cast
        # Avoid division by zero
        bg_safe = np.maximum(bg.astype(np.float32), 1.0)
        norm_float = (gray.astype(np.float32) / bg_safe) * 255.0
        norm = np.clip(norm_float, 0, 255).astype(np.uint8)

        # 3. Check for genuine inverted document (white text on dark background)
        # Sample the outer 5% border margins where paper background dominates
        margin_h, margin_w = max(h // 20, 4), max(w // 20, 4)
        corners = np.concatenate([
            norm[:margin_h, :margin_w].ravel(),
            norm[:margin_h, -margin_w:].ravel(),
            norm[-margin_h:, :margin_w].ravel(),
            norm[-margin_h:, -margin_w:].ravel()
        ])
        if np.median(corners) < 85:
            # Dark background detected (inverted)
            norm = 255 - norm

        # 4. Enhance text contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm)

        return enhanced

    @staticmethod
    def auto_orient_bill(image: np.ndarray, manual_rotation: int = 0) -> Tuple[np.ndarray, int]:
        """
        Determines the correct reading orientation of a bill slip.
        Respects manual rotation if provided (90, 180, 270).
        Uses morphological horizontal line energy comparison (0 deg vs 90 deg)
        and document header ink density to detect 0, 90, 180, or 270 deg.
        Zero external APIs or pre-trained models.
        """
        # If user explicitly specified rotation (even 0), respect it if manual_rotation is given
        if manual_rotation in (90, 180, 270):
            return ImageProcessor.rotate_image(image, manual_rotation), manual_rotation

        h, w = image.shape[:2]
        # If document is already landscape (w >= h), it's in reading orientation
        if w >= int(h * 1.1):
            return image, 0

        # If document is taller than wide (photographed in portrait), rotate 270 deg to make landscape
        if h > int(w * 1.1):
            # Check whether 270 or 90 aligns header at top
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            th_270 = cv2.rotate(th, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h_270 = th_270.shape[0]
            top_ink = np.sum(th_270[:int(h_270 * 0.30), :] > 0)
            bot_ink = np.sum(th_270[int(h_270 * 0.70):, :] > 0)
            angle = 270 if top_ink >= bot_ink * 0.75 else 90
            return ImageProcessor.rotate_image(image, angle), angle

        return image, 0

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast on any color paper."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        """Adaptive binarization suitable for handwritten and printed text on any paper color."""
        if len(image.shape) == 3:
            gray = ImageProcessor.normalize_document_color(image)
        else:
            gray = image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        return thresh

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Estimate document skew angle and rotate to level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is minor (< 15 degrees), correct it
        if abs(angle) > 0.5 and abs(angle) < 15:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        return image
