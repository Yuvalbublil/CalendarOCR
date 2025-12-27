import datetime
import logging
from zoneinfo import ZoneInfo

import numpy as np

from appointment import RelativeAppointment


def round(value: float, round_base: int):
    return np.round(value/round_base) * round_base


def add_time(roi_config: dict, time_config: dict, base_day: datetime.datetime, appt: RelativeAppointment):
    print(f"appt.bbox {appt.bbox}")
    start_y, block_size = appt.bbox[1], appt.bbox[3]
    start_y_in_roi = start_y - roi_config[1]
    start_time_hour = start_y_in_roi / \
        time_config["hour_size"] + time_config["base_hour"]
    start_time_hour = round(start_time_hour, 5/60)
    block_size_hour = round(block_size / time_config["hour_size"], 5/60)

    logging.getLogger(__name__).debug(
        f"{appt.title} , {start_time_hour}, {block_size_hour}")
    appt.start_time = base_day + datetime.timedelta(hours=start_time_hour)
    appt.duration = datetime.timedelta(hours=block_size_hour)
    return appt


def get_today_datetime() -> datetime.datetime:
    tz = ZoneInfo("Asia/Jerusalem")  # change as needed
    start_of_today = datetime.datetime.now(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return start_of_today
