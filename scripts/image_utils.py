import logging

from pathlib import Path

import cv2

from PIL import Image


def read_images(image_path: Path):
    logging.getLogger(__name__).info(f"Reading image: {image_path}")
    pil_img = Image.open(image_path).convert("RGB")
    cv_img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    return pil_img, cv_img
