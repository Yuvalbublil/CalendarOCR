from pathlib import Path
from PIL import ImageFont


def get_font(size: int):
    """Finds a valid Hebrew-supporting font on the system."""
    # List of common Hebrew font paths on Linux
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/hebrew/culmus/DavidLibre-Regular.ttf"
    ]

    for path in font_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    # If no specific font is found, try 'arial' (common in many environments)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        print("Warning: No Hebrew font found. Text may not render correctly.")
        return ImageFont.load_default()
