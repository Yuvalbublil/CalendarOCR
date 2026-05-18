import math

import pytest

from calendar_ocr.google_calendar import GoogleCalendar


class TestRgbDistance:
    def test_identical_colors(self):
        assert GoogleCalendar._rgb_distance((100, 100, 100), (100, 100, 100)) == 0.0

    def test_known_3_4_5_triangle(self):
        assert GoogleCalendar._rgb_distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    def test_symmetric(self):
        a, b = (1, 2, 3), (4, 5, 6)
        assert GoogleCalendar._rgb_distance(a, b) == pytest.approx(GoogleCalendar._rgb_distance(b, a))

    def test_single_channel(self):
        assert GoogleCalendar._rgb_distance((255, 0, 0), (0, 0, 0)) == pytest.approx(255.0)


class TestGetClosestColorId:
    @pytest.mark.parametrize("color_id,rgb", list(GoogleCalendar.COLOR_MAP.items()))
    def test_exact_colors_map_to_themselves(self, color_id, rgb):
        assert GoogleCalendar._get_closest_color_id(rgb) == color_id

    def test_near_lavender(self):
        assert GoogleCalendar._get_closest_color_id((120, 135, 200)) == "1"

    def test_returns_string(self):
        result = GoogleCalendar._get_closest_color_id((128, 128, 128))
        assert isinstance(result, str)

    def test_black_maps_to_basil(self):
        # Basil '10' (11,128,67) is closest to (0,0,0) at distance ~144.9; Graphite '8' is ~168.0
        assert GoogleCalendar._get_closest_color_id((0, 0, 0)) == "10"
