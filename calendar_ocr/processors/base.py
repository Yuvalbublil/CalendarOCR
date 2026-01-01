from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..appointment import Appointment


class AppointmentProcessor(ABC):
    """Abstract base class for processing appointments from images."""

    @abstractmethod
    def process(self, image_path: Path) -> List[Appointment]:
        """Process an image and return a list of appointments."""
        pass
