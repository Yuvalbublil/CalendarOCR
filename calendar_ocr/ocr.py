import logging
import re

import pytesseract
import cv2
import numpy as np

from PIL import Image


class OCR:

    UPSCALE_FACTOR = 3
    CLAHE_CLIP = 2.0
    CLAHE_TILE = (8, 8)
    ADAPTIVE_BLOCK_SIZE = 31
    ADAPTIVE_C = 9
    LINE_SPLIT_MIN_RUN = 1
    LINE_SPLIT_MIN_HEIGHT = 8
    TESSERACT_CONFIGS = ['--oem 3 --psm 7 -c tessedit_char_whitelist="*"',
                         "--psm 3 --oem 3",
                         r'--psm 7 -c tessedit_char_whitelist="*"',
                         ]

    @staticmethod
    def _enhance_for_ocr(pil_image: Image.Image) -> Image.Image:
        """
        Applies key pre-processing steps (grayscale, upscaling, Otsu binarization) 
        to a PIL Image object to improve OCR accuracy.

        Args:
            pil_image: The original PIL Image object (e.g., from Image.open()).

        Returns:
            A new PIL Image object that is enhanced for Tesseract OCR.
        """

        # 1. Convert PIL Image to Grayscale NumPy array (required by OpenCV)
        # The 'L' mode is 8-bit grayscale
        img_np = np.array(pil_image.convert('L'))

        # 2. Upscale (Magnify) the image 4x for better detail recognition (DPI increase)
        # INTER_CUBIC is a high-quality interpolation method
        magnified_np = cv2.resize(img_np, None, fx=4.0,
                                  fy=4.0, interpolation=cv2.INTER_CUBIC)

        # 3. Aggressive Binarization using Otsu's method
        # This automatically finds the best threshold to convert the image to pure black and white.
        # THRESH_BINARY + THRESH_OTSU ensures text is black (0) and background is white (255)
        _, binarized_np = cv2.threshold(
            magnified_np, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Convert the processed NumPy array back into a PIL Image object
        img_pil_final = Image.fromarray(binarized_np)

        return img_pil_final

    @staticmethod
    def _get_ocr_best_candidate(variants, tesseract_configs):
        best_text = ""
        highest_score = -1

        for img in variants:
            processed_img = OCR._enhance_for_ocr(img)

            for cfg in tesseract_configs:
                data = pytesseract.image_to_data(
                    processed_img, lang="heb", config=cfg, output_type=pytesseract.Output.DICT)

                confidences = []
                text_parts = []

                for i in range(len(data['text'])):
                    word = str(data['text'][i]).strip()
                    conf = float(data['conf'][i])

                    is_symbol_word = bool(re.match(r'^[*#\-_₪]+$', word))
                    logging.getLogger(__name__).debug(
                        f"is_symbol_word: {is_symbol_word}")

                    if word:
                        if conf > -1 or is_symbol_word:
                            text_parts.append(word)
                            confidences.append(max(conf, 1.0))

                if not text_parts:
                    continue

                current_full_text = " ".join(text_parts).strip()
                avg_confidence = sum(confidences) / len(confidences)

                hebrew_char_count = len(re.findall(
                    r'[\u0590-\u05FF]', current_full_text))

                current_score = avg_confidence + (hebrew_char_count * 2)

                if current_score > highest_score:
                    highest_score = current_score
                    best_text = current_full_text

        return best_text

    def ocr_box(self, pil_img: Image.Image, box) -> str:
        x, y, w, h = box
        crop = pil_img.crop((x, y, x + w, y + h))
        # Upscale to help OCR on small/low-contrast text.
        crop = crop.resize(
            (max(1, w * OCR.UPSCALE_FACTOR), max(1, h * OCR.UPSCALE_FACTOR)),
            Image.Resampling.LANCZOS,
        )
        gray = np.array(crop.convert("L"))

        def to_pil(arr):
            return Image.fromarray(arr)

        # CLAHE to boost contrast on colored backgrounds.
        clahe = cv2.createCLAHE(clipLimit=OCR.CLAHE_CLIP,
                                tileGridSize=OCR.CLAHE_TILE)
        gray_clahe = clahe.apply(gray)

        # Variants to handle dark-on-light and light-on-dark.
        variants = [
            crop,  # original RGB
            to_pil(gray),
            to_pil(gray_clahe),
            to_pil(255 - gray),
            to_pil(255 - gray_clahe),
        ]
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            OCR.ADAPTIVE_BLOCK_SIZE,
            OCR.ADAPTIVE_C,
        )
        variants.append(to_pil(adaptive))
        variants.append(to_pil(255 - adaptive))

        logging.getLogger(__name__).debug(box)

        return OCR._get_ocr_best_candidate(variants, OCR.TESSERACT_CONFIGS)
