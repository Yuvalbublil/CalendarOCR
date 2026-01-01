from typing import Tuple, Optional
from dataclasses import dataclass
import datetime


@dataclass
class Appointment:
    title: str
    color: Tuple[int, int, int]  # RGB
    start_time: datetime.datetime
    duration: datetime.timedelta
    timezone: str = 'Asia/Jerusalem'  # Default time zone
    google_event_id: Optional[str] = None

    @property
    def end_time(self) -> datetime.datetime:
        return self.start_time + self.duration

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)


@dataclass
class RelativeAppointment:
    title: str
    color: Tuple[int, int, int]  # RGB
    bbox: Tuple[int, int, int, int]  # x, y, w, h

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)
