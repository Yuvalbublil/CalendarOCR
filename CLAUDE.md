# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Calendar OCR is a Python pipeline for extracting appointment information from scanned/photographed Hebrew calendar images using Tesseract OCR, then syncing the results to Google Calendar.

## Setup

Requires a conda environment (Python 3.14) and system-level Tesseract with Hebrew language data:

```bash
conda env create -f environment.yml
conda activate calendar-ocr
pip install -e .
```

Tesseract must have `heb.traineddata` installed (Hebrew OCR support).

## Common Commands

```bash
# Run the CLI on an image
calendar-ocr <image_path> -c roi.yaml

# With debug visualization
calendar-ocr <image_path> -c roi.yaml --verbose

# Delete today's events before adding new ones
calendar-ocr <image_path> -c roi.yaml --delete-after

# Interactive ROI selector — creates/updates roi.yaml
python helpers/mark_roi.py <image_path> -o roi.yaml

# Type checking
mypy calendar_ocr/

# Linting / formatting
flake8 calendar_ocr/
black calendar_ocr/

# Tests (framework configured, no tests written yet)
pytest
```

## Architecture

### Data Flow

```
Image → ROI crop → contour detection → text boxes
     → Tesseract OCR (6 image variants × 3 PSM configs, best scored)
     → pixel Y-coord → absolute time (via roi.yaml time config)
     → dominant color detection
     → Google Calendar sync
```

### Key Classes

| Class | File | Role |
|---|---|---|
| `CalendarOCR` | `calendar_ocr/calendar_ocr.py` | Facade — loads config, creates processor |
| `CalendarProcessor` | `calendar_ocr/processors/calendar.py` | Orchestrates extraction → time → sync |
| `AppointmentsExtractor` | `calendar_ocr/appointments_extractor.py` | Detects text bounding boxes from image |
| `OCR` | `calendar_ocr/ocr.py` | Tesseract wrapper with image preprocessing |
| `GoogleCalendar` | `calendar_ocr/google_calendar.py` | Google Calendar API v3 (OAuth2) |
| `Appointment` | `calendar_ocr/appointments/appointment.py` | Dataclass with absolute datetime |
| `RelativeAppointment` | `calendar_ocr/appointments/relative_appointment.py` | Intermediate: bbox instead of time |

### Two-Stage Appointment Model

1. **`RelativeAppointment`** — produced by `AppointmentsExtractor`; has a pixel bounding box (`bbox`) and detected text/color.
2. **`Appointment`** — produced by `time_utils.add_time()`; bbox converted to `start_time` + `duration` using the `roi.yaml` time config.

### Configuration (`roi.yaml`)

```yaml
roi: [x, y, w, h]        # Region of interest in pixels
time:
  base_hour: 8            # Calendar top = 8:00 AM
  hour_size: 48           # Pixels per hour
```

Use `helpers/mark_roi.py` to interactively generate this file for a new calendar layout.

### OCR Strategy (`ocr.py`)

For each text box, 6 image variants are generated (original RGB, grayscale, CLAHE, inverted, inverted-CLAHE, adaptive threshold) and tested against 3 Tesseract PSM configs (3 and 7). Results are scored by OCR confidence + Hebrew character count; the best result is used.

### Text Box Detection (`appointments_extractor.py`)

Uses OpenCV morphological dilation + contour detection on both normal and inverted binary masks. Overlapping boxes are merged via IOU. Boxes are filtered by minimum size (2×2px) and maximum area (400×400px), then sorted top-to-bottom by Y coordinate.

### Google Calendar Auth

OAuth2 flow using `credentials.json` (downloaded from Google Cloud Console). Token is cached to `token.json` and auto-refreshed. Both files are gitignored. RGB appointment colors are mapped to Google Calendar's 11-color palette.

### Hebrew Text Rendering

Hebrew text in debug visualizations (`draw_utils.py`) requires `arabic-reshaper` + `python-bidi` for correct RTL rendering. Font is loaded via `font.py`.

