import datetime

import pytest

from calendar_ocr.appointments.appointment import Appointment
from calendar_ocr.appointments.relative_appointment import RelativeAppointment


# ── Appointment ───────────────────────────────────────────────────────────────

class TestAppointmentEndTime:
    def test_equals_start_plus_duration(self, base_day):
        appt = Appointment(
            title="A",
            color=(0, 0, 0),
            start_time=base_day + datetime.timedelta(hours=9),
            duration=datetime.timedelta(hours=1),
        )
        assert appt.end_time == base_day + datetime.timedelta(hours=10)

    def test_zero_duration(self, base_day):
        appt = Appointment(
            title="A",
            color=(0, 0, 0),
            start_time=base_day,
            duration=datetime.timedelta(0),
        )
        assert appt.end_time == appt.start_time

    def test_multi_day_duration(self, base_day):
        appt = Appointment(
            title="A",
            color=(0, 0, 0),
            start_time=base_day + datetime.timedelta(hours=9),
            duration=datetime.timedelta(hours=25),
        )
        expected = base_day + datetime.timedelta(hours=34)
        assert appt.end_time == expected


class TestAppointmentToHex:
    def test_known_color(self):
        appt = Appointment(
            title="A", color=(121, 134, 203),
            start_time=datetime.datetime(2024, 1, 1), duration=datetime.timedelta(hours=1)
        )
        assert appt.to_hex() == "#7986cb"

    def test_black(self):
        appt = Appointment(
            title="A", color=(0, 0, 0),
            start_time=datetime.datetime(2024, 1, 1), duration=datetime.timedelta(hours=1)
        )
        assert appt.to_hex() == "#000000"

    def test_white(self):
        appt = Appointment(
            title="A", color=(255, 255, 255),
            start_time=datetime.datetime(2024, 1, 1), duration=datetime.timedelta(hours=1)
        )
        assert appt.to_hex() == "#ffffff"

    def test_zero_padding(self):
        appt = Appointment(
            title="A", color=(1, 2, 3),
            start_time=datetime.datetime(2024, 1, 1), duration=datetime.timedelta(hours=1)
        )
        assert appt.to_hex() == "#010203"


class TestAppointmentDefaults:
    def test_default_timezone(self, sample_appointment):
        assert sample_appointment.timezone == "Asia/Jerusalem"

    def test_google_event_id_defaults_none(self, sample_appointment):
        assert sample_appointment.google_event_id is None

    def test_dataclass_equality(self, base_day):
        kwargs = dict(
            title="X", color=(1, 2, 3),
            start_time=base_day, duration=datetime.timedelta(hours=1)
        )
        assert Appointment(**kwargs) == Appointment(**kwargs)


# ── RelativeAppointment ───────────────────────────────────────────────────────

class TestRelativeAppointment:
    def test_to_hex(self):
        ra = RelativeAppointment(title="A", color=(51, 182, 121), bbox=(0, 0, 10, 10))
        assert ra.to_hex() == "#33b679"

    def test_bbox_stored(self):
        ra = RelativeAppointment(title="A", color=(0, 0, 0), bbox=(10, 20, 30, 40))
        assert ra.bbox == (10, 20, 30, 40)
