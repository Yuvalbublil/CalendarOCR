import pytest

from calendar_ocr.ocr_backends import create_ocr, TesseractOCR, SuryaOCRBackend


class TestCreateOcr:
    def test_default_is_tesseract(self):
        assert isinstance(create_ocr({}), TesseractOCR)

    def test_explicit_tesseract(self):
        assert isinstance(create_ocr({"ocr": {"backend": "tesseract"}}), TesseractOCR)

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            create_ocr({"ocr": {"backend": "unknown"}})

    def test_error_message_contains_backend_name(self):
        with pytest.raises(ValueError, match="foobar"):
            create_ocr({"ocr": {"backend": "foobar"}})

    def test_surya(self):
        pytest.importorskip("surya", reason="surya-ocr not installed")
        assert isinstance(create_ocr({"ocr": {"backend": "surya"}}), SuryaOCRBackend)
