
import logging
import cv2

from typing import Tuple, Optional, List, Dict
from pathlib import Path

from .appointments.relative_appointment import RelativeAppointment
from . import image_utils
from . import ocr

MAX_BOX_AREA = 400*400

MORPH_KERNEL_HEIGHT = 1
MORPH_KERNEL_WIDTH_RATIO = .0001
MIN_BOX_H = 2
MIN_BOX_W = 2
IOU_MERGE_THRESH = 0.6

# COLOR
STRIP_SIDE_PIXELS = 50


class AppointmentsExtractor:

    def __init__(self, roi: Optional[Tuple[int, int, int, int]] = None):
        self._roi = roi

    @staticmethod
    def _dominant_color_near_line(cv_img, box) -> Tuple[int, int, int]:
        x, y, w, h = box
        strip = cv_img[y: y + h, max(0, x - STRIP_SIDE_PIXELS): x + min(STRIP_SIDE_PIXELS, w)]
        pixels = strip.reshape(-1, 3)
        counts: Dict[Tuple[int, int, int], int] = {}
        for r, g, b in pixels:
            key = (int(r), int(g), int(b))
            counts[key] = counts.get(key, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else (0, 0, 0)

    @staticmethod
    def _find_text_boxes(
            cv_img) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        _, thresh_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(1, int(cv_img.shape[1] *
                        MORPH_KERNEL_WIDTH_RATIO)), MORPH_KERNEL_HEIGHT),
        )

        def _boxes_from(mask):
            dilated = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            out = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if h < MIN_BOX_H or w < MIN_BOX_W:
                    continue
                out.append((x, y, w, h))
            return out

        boxes = _boxes_from(thresh_inv) + _boxes_from(thresh)
        for b in boxes:
            if b[2] * b[3] > MAX_BOX_AREA:
                boxes.remove(b)

        logging.getLogger(__name__).debug(f"Boxes before merge: {boxes}")

        def iou(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1 + w1, x2 + w2)
            yb = min(y1 + h1, y2 + h2)
            inter = max(0, xb - xa) * max(0, yb - ya)
            if inter == 0:
                return 0.0
            area1 = w1 * h1
            area2 = w2 * h2
            return inter / float(area1 + area2 - inter)

        merged: List[Tuple[int, int, int, int]] = []
        for b in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            if not any(iou(b, m) > IOU_MERGE_THRESH for m in merged):
                merged.append(b)

        merged.sort(key=lambda b: b[1])

        logging.getLogger(__name__).debug(f"Boxes after merge: {merged}")

        return merged

    def extract_appointments(
        self, image_path: Path
    ) -> List[RelativeAppointment]:
        appointments_ocr = ocr.OCR()

        pil_img, cv_img = image_utils.read_images(image_path)
        offset_x = offset_y = 0
        if self._roi:
            rx, ry, rw, rh = self._roi
            pil_img = pil_img.crop((rx, ry, rx + rw, ry + rh))
            cv_img = cv_img[ry: ry + rh, rx: rx + rw]
            offset_x, offset_y = rx, ry

        boxes = AppointmentsExtractor._find_text_boxes(cv_img)

        logging.getLogger(__name__).debug(f"Detected boxes: {boxes}")

        appointments: List[RelativeAppointment] = []
        for box in boxes:
            text = appointments_ocr.ocr_box(pil_img, box)
            if not text:
                text = "unrecognized"

            color = AppointmentsExtractor._dominant_color_near_line(
                cv_img, box)
            gx, gy, gw, gh = box[0] + offset_x, box[1] + \
                offset_y, box[2], box[3]
            appointments.append(
                RelativeAppointment(title=' '.join(text[::].split()),
                                    color=color, bbox=(gx, gy, gw, gh))
            )

        return appointments
