from typing import Any, Dict

from .base import BaseOCR
from .tesseract import TesseractOCR
from .surya import SuryaOCRBackend

__all__ = ["BaseOCR", "TesseractOCR", "SuryaOCRBackend", "create_ocr"]


def create_ocr(config: Dict[str, Any]) -> BaseOCR:
    """Instantiate the OCR backend specified in config.

    Config shape (all keys optional):
        ocr:
          backend: tesseract   # 'tesseract' (default) or 'surya'
    """
    ocr_cfg = config.get("ocr", {})
    backend = ocr_cfg.get("backend", "tesseract")

    if backend == "surya":
        return SuryaOCRBackend()

    if backend == "tesseract":
        return TesseractOCR()

    raise ValueError(
        f"Unknown OCR backend '{backend}'. Valid options: 'tesseract', 'surya'."
    )
