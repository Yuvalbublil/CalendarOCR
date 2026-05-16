import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import AppointmentProcessor
from ..appointments.appointment import Appointment
from ..appointments_extractor import AppointmentsExtractor
from ..google_calendar import GoogleCalendar
from .. import time_utils


class CalendarProcessor(AppointmentProcessor):
    """Concrete processor for calendar OCR with time addition and Google Calendar sync."""

    def __init__(
        self,
        roi: Optional[tuple[int, int, int, int]],
        time_config: Dict[str, Any],
        appointments_extractor: AppointmentsExtractor,
        google_calendar: Optional[GoogleCalendar] = None,
    ):
        self._roi = roi
        self._time_config = time_config
        self._appointments_extractor = appointments_extractor
        self._google_calendar = google_calendar

    def process(self, image_path: Path) -> List[Appointment]:
        """Extract appointments from image, add time info, and sync to Google Calendar."""
        ocr_appts = self._appointments_extractor.extract_appointments(
            image_path)

        appts: List[Appointment] = []
        for rel_appt in ocr_appts:
            appts.append(
                time_utils.add_time(
                    self._roi,
                    self._time_config,
                    time_utils.get_today_datetime(),
                    rel_appt
                )
            )  # TODO: change today to the correct day

        for appt_obj in appts:
            logging.getLogger(__name__).debug(
                {"appointment": appt_obj, "color_hex": appt_obj.to_hex()}
            )

        if self._google_calendar:
            for appt in appts:
                self._google_calendar.add_appointment(appt)

        return appts
