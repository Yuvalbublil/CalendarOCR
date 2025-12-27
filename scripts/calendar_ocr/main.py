"""
Lightweight OCR pipeline for the provided calendar image format.
Outputs RelativeAppointment objects with title, time, and dominant color swatch.
Hebrew OCR via Tesseract (lang=heb).

Requirements inside conda env `opencv_py310`:
  - system tesseract-ocr + Hebrew data (heb.traineddata)
  - pip install pytesseract
  - opencv-python (cv2) and pillow (PIL)
"""

import argparse
import logging
import sys
import datetime

from pathlib import Path

from google_calendar import GoogleCalendar
from calendar_ocr import CalendarOCR

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)


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
    return parser.parse_args()


def main() -> None:
    args = get_args()

    CAL_ID = '933761ffc4a60226eff2909014f540d1c3d4aefa72497b9293c4ab3a975fbf3f@group.calendar.google.com'
    TOKEN = 'token.json'
    CREDS = 'credentials.json'

    image_path = args.image

    if not image_path.exists():
        msg = f"Image not found: {image_path}"
        logging.getLogger(__name__).info(msg)
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    calendar_ocr = CalendarOCR(
        args.config, GoogleCalendar(CAL_ID, TOKEN, CREDS))
    appointments = calendar_ocr.process_image(image_path)

    print(f"Extracted {len(appointments)} appointments.")

    print(GoogleCalendar(CAL_ID, TOKEN, CREDS).delete_all_meetings_in_day(
        datetime.date.today()))


if __name__ == "__main__":
    main()
