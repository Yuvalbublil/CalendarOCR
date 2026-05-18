import numpy as np
import pytest

from calendar_ocr.appointments_extractor import AppointmentsExtractor
from calendar_ocr.appointments.relative_appointment import RelativeAppointment


# ── _dominant_color_near_line ─────────────────────────────────────────────────

class TestDominantColorNearLine:
    def test_solid_red_image(self, solid_red_cv_img):
        color = AppointmentsExtractor._dominant_color_near_line(solid_red_cv_img, (10, 10, 20, 20))
        assert color == (255, 0, 0)

    def test_green_strip(self, colored_strip_cv_img):
        # x=80, w=20 → strip is cv_img[:, max(0,80-50): 80+min(50,20)] = [:, 30:100]
        # columns 30–100 are all green (left half is 0–99)
        color = AppointmentsExtractor._dominant_color_near_line(colored_strip_cv_img, (80, 0, 20, 50))
        assert color == (0, 200, 0)

    def test_blue_strip(self, colored_strip_cv_img):
        # x=150, w=20 → strip is cv_img[:, max(0,150-50): 150+min(50,20)] = [:, 100:170]
        # columns 100–170 are all blue (right half is 100–199)
        color = AppointmentsExtractor._dominant_color_near_line(colored_strip_cv_img, (150, 0, 20, 50))
        assert color == (0, 0, 200)

    def test_empty_strip_fallback(self, solid_red_cv_img):
        # h=0 → strip has zero rows → empty → fallback (0,0,0)
        color = AppointmentsExtractor._dominant_color_near_line(solid_red_cv_img, (10, 10, 20, 0))
        assert color == (0, 0, 0)


# ── _find_text_boxes ──────────────────────────────────────────────────────────

class TestFindTextBoxes:
    def test_white_image_returns_few_or_no_boxes(self, white_cv_img):
        boxes = AppointmentsExtractor._find_text_boxes(white_cv_img)
        # Otsu on a uniform white image yields little; allow a couple of spurious contours
        assert len(boxes) <= 2

    def test_detects_black_band(self, text_strip_cv_img):
        boxes = AppointmentsExtractor._find_text_boxes(text_strip_cv_img)
        assert len(boxes) >= 1
        # At least one box should overlap the band region (rows 40–60)
        assert any(35 <= b[1] <= 65 for b in boxes)

    def test_sorted_by_y(self, two_band_cv_img):
        boxes = AppointmentsExtractor._find_text_boxes(two_band_cv_img)
        assert len(boxes) >= 2
        ys = [b[1] for b in boxes]
        assert ys == sorted(ys)

    def test_filters_tiny_dots(self):
        # Single-pixel dot — should be filtered out (MIN_BOX_H=2, MIN_BOX_W=2)
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        img[25, 25] = 0
        boxes = AppointmentsExtractor._find_text_boxes(img)
        # Any returned box must be at least 2×2
        for x, y, w, h in boxes:
            assert w >= 2 and h >= 2


# ── extract_appointments ──────────────────────────────────────────────────────

class TestExtractAppointments:
    @pytest.mark.integration
    def test_returns_list(self, real_image_path, fake_ocr):
        extractor = AppointmentsExtractor(roi=None, ocr=fake_ocr)
        result = extractor.extract_appointments(real_image_path)
        assert isinstance(result, list)

    @pytest.mark.integration
    def test_all_are_relative_appointments(self, real_image_path, fake_ocr):
        extractor = AppointmentsExtractor(roi=None, ocr=fake_ocr)
        result = extractor.extract_appointments(real_image_path)
        assert all(isinstance(a, RelativeAppointment) for a in result)

    @pytest.mark.integration
    def test_empty_ocr_uses_unrecognized(self, real_image_path, empty_ocr):
        extractor = AppointmentsExtractor(roi=None, ocr=empty_ocr)
        result = extractor.extract_appointments(real_image_path)
        if result:
            assert all(a.title == "unrecognized" for a in result)

    @pytest.mark.integration
    def test_with_roi_bbox_offset(self, real_image_path, fake_ocr, roi_config):
        extractor = AppointmentsExtractor(roi=roi_config, ocr=fake_ocr)
        result = extractor.extract_appointments(real_image_path)
        rx = roi_config[0]
        for a in result:
            assert a.bbox[0] >= rx
