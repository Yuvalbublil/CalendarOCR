import datetime
from pathlib import Path

import numpy as np
import pytest

from calendar_ocr.appointments.appointment import Appointment
from calendar_ocr.appointments.relative_appointment import RelativeAppointment
from calendar_ocr.ocr_backends.base import BaseOCR

REPO_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that read real images from disk")
    config.addinivalue_line("markers", "tesseract: tests that require system Tesseract + heb.traineddata")


# ── Synthetic numpy images ────────────────────────────────────────────────────

@pytest.fixture()
def solid_red_cv_img():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)
    return img


@pytest.fixture()
def white_cv_img():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture()
def text_strip_cv_img():
    """White image with a black horizontal band at rows 40–60."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[40:60, 20:180] = 0
    return img


@pytest.fixture()
def two_band_cv_img():
    """White image with black bands at rows 20–30 and 80–90."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[20:30, 20:180] = 0
    img[80:90, 20:180] = 0
    return img


@pytest.fixture()
def colored_strip_cv_img():
    """50×200 image: left half green (0,200,0), right half blue (0,0,200)."""
    img = np.zeros((50, 200, 3), dtype=np.uint8)
    img[:, :100] = (0, 200, 0)
    img[:, 100:] = (0, 0, 200)
    return img


# ── Domain object fixtures ────────────────────────────────────────────────────

@pytest.fixture()
def base_day():
    return datetime.datetime(2024, 3, 15, 0, 0, 0)


@pytest.fixture()
def sample_appointment(base_day):
    return Appointment(
        title="Meeting",
        color=(121, 134, 203),
        start_time=base_day + datetime.timedelta(hours=9),
        duration=datetime.timedelta(hours=1),
    )


@pytest.fixture()
def sample_relative_appointment():
    return RelativeAppointment(
        title="Doctor",
        color=(51, 182, 121),
        bbox=(10, 188, 200, 48),
    )


@pytest.fixture()
def time_config():
    return {"base_hour": 8, "hour_size": 48}


@pytest.fixture()
def roi_config():
    return (9, 140, 426, 576)


@pytest.fixture(scope="session")
def real_image_path():
    p = REPO_ROOT / "photo_calender.jpg"
    if not p.exists():
        pytest.skip("photo_calender.jpg not present in repo root")
    return p


# ── Fake OCR backends ─────────────────────────────────────────────────────────

class _FakeOCR(BaseOCR):
    def __init__(self, text="fake text"):
        self._text = text

    def ocr_box(self, pil_img, box):
        return self._text


@pytest.fixture()
def fake_ocr():
    return _FakeOCR()


@pytest.fixture()
def empty_ocr():
    return _FakeOCR(text="")
