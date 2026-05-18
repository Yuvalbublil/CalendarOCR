import logging
import re
from typing import Tuple

import pytesseract
import cv2
import numpy as np
from PIL import Image

from .base import BaseOCR


class TesseractOCR(BaseOCR):

    UPSCALE_FACTOR = 3
    CLAHE_CLIP = 2.0
    CLAHE_TILE = (8, 8)
    ADAPTIVE_BLOCK_SIZE = 31
    ADAPTIVE_C = 9
    TESSERACT_CONFIGS = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist="*"',
        "--psm 3 --oem 3",
        r'--psm 7 -c tessedit_char_whitelist="*"',
    ]

    @staticmethod
    def _enhance_for_ocr(pil_image: Image.Image) -> Image.Image:
        img_np = np.array(pil_image.convert('L'))
        magnified_np = cv2.resize(img_np, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        _, binarized_np = cv2.threshold(magnified_np, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binarized_np)

    @staticmethod
    def _get_ocr_best_candidate(variants, tesseract_configs):
        best_text = ""
        highest_score = -1

        for img in variants:
            processed_img = TesseractOCR._enhance_for_ocr(img)

            for cfg in tesseract_configs:
                data = pytesseract.image_to_data(
                    processed_img, lang="heb", config=cfg, output_type=pytesseract.Output.DICT)

                confidences = []
                text_parts = []

                for i in range(len(data['text'])):
                    word = str(data['text'][i]).strip()
                    conf = float(data['conf'][i])
                    is_symbol_word = bool(re.match(r'^[*#\-_₪]+$', word))
                    logging.getLogger(__name__).debug(f"is_symbol_word: {is_symbol_word}")

                    if word:
                        if conf > -1 or is_symbol_word:
                            text_parts.append(word)
                            confidences.append(max(conf, 1.0))

                if not text_parts:
                    continue

                current_full_text = " ".join(text_parts).strip()
                avg_confidence = sum(confidences) / len(confidences)
                hebrew_char_count = len(re.findall(r'[֐-׿]', current_full_text))
                current_score = avg_confidence + (hebrew_char_count * 2)

                if current_score > highest_score:
                    highest_score = current_score
                    best_text = current_full_text

        return best_text

    def ocr_box(self, pil_img: Image.Image, box: Tuple[int, int, int, int]) -> str:
        x, y, w, h = box
        crop = pil_img.crop((x, y, x + w, y + h))
        crop = crop.resize(
            (max(1, w * self.UPSCALE_FACTOR), max(1, h * self.UPSCALE_FACTOR)),
            Image.Resampling.LANCZOS,
        )
        gray = np.array(crop.convert("L"))

        def to_pil(arr):
            return Image.fromarray(arr)

        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP, tileGridSize=self.CLAHE_TILE)
        gray_clahe = clahe.apply(gray)

        variants = [
            crop,
            to_pil(gray),
            to_pil(gray_clahe),
            to_pil(255 - gray),
            to_pil(255 - gray_clahe),
        ]
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.ADAPTIVE_BLOCK_SIZE,
            self.ADAPTIVE_C,
        )
        variants.append(to_pil(adaptive))
        variants.append(to_pil(255 - adaptive))

        logging.getLogger(__name__).debug(box)
        return self._get_ocr_best_candidate(variants, self.TESSERACT_CONFIGS)
