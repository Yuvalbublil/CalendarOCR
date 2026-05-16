from dataclasses import dataclass
from typing import Tuple


@dataclass
class RelativeAppointment:
    title: str
    color: Tuple[int, int, int]  # RGB
    bbox: Tuple[int, int, int, int]  # x, y, w, h

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)
