"""
Interactive ROI marker to save a crop box into roi.yaml.

Usage (inside env with opencv & pyyaml):
  python mark_roi.py photo_calendar.jpg --output roi.yaml

Controls:
  - Drag to select ROI (OpenCV selectROI)
  - Press ENTER/SPACE to confirm, or 'c' to cancel and reselect
  - Press ESC to quit without saving
"""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import yaml


def save_roi(roi: Tuple[int, int, int, int], path: Path) -> None:
    data = {"roi": [int(v) for v in roi]}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select ROI and save to YAML")
    parser.add_argument("image", type=Path, help="Image to annotate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("roi.yaml"),
        help="Output YAML file (default: roi.yaml)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit("Failed to load image.")

    # OpenCV selectROI uses BGR; returns (x, y, w, h)
    roi = cv2.selectROI(
        "Select ROI (ENTER to confirm, ESC to quit)", img, False, False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise SystemExit("No ROI selected; nothing saved.")

    save_roi((x, y, w, h), args.output)
    print(f"Saved ROI to {args.output}: {roi}")


if __name__ == "__main__":
    main()
