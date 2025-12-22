"""
Lightweight OCR pipeline for the provided calendar image format.
Outputs Appointment objects with title, time, and dominant color swatch.
Hebrew OCR via Tesseract (lang=heb).

Requirements inside conda env `opencv_py310`:
  - system tesseract-ocr + Hebrew data (heb.traineddata)
  - pip install pytesseract
  - opencv-python (cv2) and pillow (PIL)
"""

import time_utils
import datetime
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional

import argparse
from typing import Any, Dict, List

import numpy as np
import yaml


import image_utils
import draw_utils
from appointment import Appointment
from appointments_extractor import AppointmentsExtractor


CONFIG_DEFAULT = {}

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)


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


def get_args():
    parser = argparse.ArgumentParser(
        description="OCR calendar appointments (Hebrew)"
    )
    parser.add_argument("image", type=Path, help="Path to calendar image")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to YAML config (supports roi: [x,y,w,h])",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Save debug image with bounding boxes and text overlay",
    )
    parser.add_argument(
        "--debug-path",
        type=Path,
        help="Optional output path for debug image (used with -v)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory to store intermediate debug masks/overlays",
    )
    return parser.parse_args()


def read_roi(config):
    roi = tuple(config.get("roi", [])) if isinstance(
        config.get("roi"), list) else None

    if roi and len(roi) != 4:
        msg = f"Invalid roi in config; expected [x, y, w, h] Got {roi}"
        logging.getLogger(__name__).exception(msg)
        raise ValueError(msg)

    roi_int = tuple(int(v) for v in roi) if roi else None
    return roi_int


def main() -> None:

    args = get_args()

    image_path = args.image
    if not image_path.exists():
        msg = f"Image not found: {image_path}"
        logging.getLogger(__name__).info(msg)
        raise FileNotFoundError(f"Image not found: {image_path}")

    config = _load_config(args.config)
    roi = read_roi(config)
    time_config = config["time"]

    debug_dir: Optional[Path] = None
    if args.verbose:
        debug_dir = args.debug_dir or image_path.with_name(
            f"{image_path.stem}_debug_artifacts"
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
    appointments_extractor = AppointmentsExtractor()

    appts = appointments_extractor.extract_appointments(
        image_path,
        verbose=args.verbose,
        debug_dir=debug_dir,
        roi=roi)

    for appt in appts:
        appt = time_utils.add_time(
            config["roi"], time_config, time_utils.get_today_datetime(), appt)

    for appt in appts:
        logging.getLogger(__name__).info(
            asdict(appt) | {"color_hex": appt.to_hex()})

    if args.verbose:
        _, cv_img = image_utils.read_images(image_path)
        debug_path: Path = (
            args.debug_path
            if args.debug_path
            else (debug_dir / "overlay.jpg" if debug_dir else image_path.with_name(f"{image_path.stem}_debug.jpg"))
        )
        draw_utils.draw_debug_image(cv_img, appts, debug_path)

        logging.getLogger(__name__).info(f"Debug image saved to: {debug_path}")


if __name__ == "__main__":
    main()
