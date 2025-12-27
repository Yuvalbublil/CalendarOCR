import cv2
import arabic_reshaper
import numpy as np

from bidi.algorithm import get_display
from PIL import Image, ImageDraw
from typing import List, Tuple
from pathlib import Path

from font import get_font


def draw_debug_image(
    cv_img, appts: List["RelativeAppointment"], out_path: Path, offset: Tuple[int, int] = (0, 0)
) -> Path:
    debug_img = cv_img.copy()
    ox, oy = offset

    # Convert to Pillow
    pil_img = Image.fromarray(debug_img)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(18)

    for appt in appts:
        x, y, w, h = appt.bbox
        x_local, y_local = x - ox, y - oy

        # Draw bounding box
        draw.rectangle([x_local, y_local, x_local + w,
                       y_local + h], outline=(0, 255, 0), width=2)

        # Handle Hebrew Text
        label = f"{appt.title}".strip()

        if label:
            # Reshape and handle BiDi
            reshaped_text = arabic_reshaper.reshape(label)
            bidi_text = get_display(reshaped_text)

            # Draw text
            text_pos = (x_local, max(0, y_local - 22))
            draw.text(text_pos, bidi_text, font=font, fill=(255, 0, 0))

    # Convert back to BGR and save
    final_img = np.array(pil_img)
    bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)

    return out_path
