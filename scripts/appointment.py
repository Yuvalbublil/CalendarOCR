from typing import Tuple
from dataclasses import dataclass
import datetime


@dataclass
class Appointment:
    title: str
    color: Tuple[int, int, int]  # RGB
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    start_time: datetime.timedelta = None
    duration: datetime.timedelta = None

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)
