import datetime
import logging
from zoneinfo import ZoneInfo

import numpy as np

from .appointment import RelativeAppointment, Appointment

MINUTES_PER_HOUR = 60


def round(value: float, round_base: float):
    return np.round(value/round_base) * round_base


def add_time(roi_config: tuple | None, time_config: dict, base_day: datetime.datetime, appt: RelativeAppointment) -> Appointment:
    MINUTES_TO_ROUND = 5

    logging.getLogger(__name__).debug(f"appt.bbox {appt.bbox}")
    start_y, block_size = appt.bbox[1], appt.bbox[3]
    start_y_in_roi = start_y - (roi_config[1] if roi_config else 0)
    start_time_hour = start_y_in_roi / \
        time_config["hour_size"] + time_config["base_hour"]

    start_time_hour = round(start_time_hour, MINUTES_TO_ROUND/MINUTES_PER_HOUR)

    block_size_hour = round(
        block_size / time_config["hour_size"], MINUTES_TO_ROUND/MINUTES_PER_HOUR)

    logging.getLogger(__name__).debug(
        f"{appt.title} , {start_time_hour}, {block_size_hour}")

    start_time = base_day + datetime.timedelta(hours=start_time_hour)
    duration = datetime.timedelta(hours=block_size_hour)

    return Appointment(title=appt.title,
                       color=appt.color,
                       start_time=start_time,
                       duration=duration
                       )


def get_today_datetime() -> datetime.datetime:
    tz = ZoneInfo("Asia/Jerusalem")  # change as needed
    start_of_today = datetime.datetime.now(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start_of_today
