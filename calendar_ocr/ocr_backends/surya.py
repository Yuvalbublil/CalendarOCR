import logging
from typing import Tuple

from PIL import Image

from .base import BaseOCR


class SuryaOCRBackend(BaseOCR):
    """Surya OCR backend (v0.17+).

    Predictors are heavy — initialized once and reused for every box.
    Bboxes are passed directly so no DetectionPredictor is needed;
    Surya is language-agnostic so no language code is required.
    """

    def __init__(self):
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        foundation = FoundationPredictor()
        self._recognition = RecognitionPredictor(foundation)
        logging.getLogger(__name__).debug("Surya OCR backend initialised")

    def ocr_box(self, pil_img: Image.Image, box: Tuple[int, int, int, int]) -> str:
        x, y, w, h = box
        crop = pil_img.crop((x, y, x + w, y + h)).convert("RGB")

        # bboxes shape: List[per-image List[per-box [x1,y1,x2,y2]]]
        # Passing the full-crop bbox skips internal detection entirely.
        predictions = self._recognition([crop], bboxes=[[[0, 0, w, h]]])

        texts = []
        for page in predictions:
            for line in page.text_lines:
                if line.text.strip():
                    texts.append(line.text.strip())
        return " ".join(texts)
