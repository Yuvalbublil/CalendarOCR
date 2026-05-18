from abc import ABC, abstractmethod
from typing import Tuple

from PIL import Image


class BaseOCR(ABC):
    @abstractmethod
    def ocr_box(self, pil_img: Image.Image, box: Tuple[int, int, int, int]) -> str: ...
