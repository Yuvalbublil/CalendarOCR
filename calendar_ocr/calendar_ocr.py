import logging
import yaml

from pathlib import Path
from typing import Optional, List, Dict, Any

from .google_calendar import GoogleCalendar
from .appointments.appointment import Appointment
from .appointments_extractor import AppointmentsExtractor
from .ocr_backends import create_ocr
from .processors import CalendarProcessor

CONFIG_DEFAULT: Dict[str, Any] = {}


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return CONFIG_DEFAULT
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to use a config file. Install pyyaml.")
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Config must be a YAML mapping")
    return data


def read_roi(config: Dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    roi = tuple(config.get("roi", [])) if isinstance(
        config.get("roi"), list) else None

    if roi and len(roi) != 4:
        msg = f"Invalid roi in config; expected [x, y, w, h] Got {roi}"
        logging.getLogger(__name__).exception(msg)
        raise ValueError(msg)

    roi_int = tuple(int(v) for v in roi) if roi else None
    return roi_int  # type: ignore


class CalendarOCR:
    def __init__(self, config_path: Optional[Path] = None, google_calendar: Optional[GoogleCalendar] = None):
        config = _load_config(config_path)
        self._roi = read_roi(config)
        self._time_config = config.get("time", {})
        ocr_backend = create_ocr(config)
        self._appointments_extractor = AppointmentsExtractor(self._roi, ocr_backend)
        self._processor = CalendarProcessor(
            self._roi,
            self._time_config,
            self._appointments_extractor,
            google_calendar
        )

    def process_image(self, image_path: Path) -> List[Appointment]:
        return self._processor.process(image_path)
