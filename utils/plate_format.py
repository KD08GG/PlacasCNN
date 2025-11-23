# utils/plate_format.py
import re

# ejemplo AAA-999-A
PLATE_REGEX = re.compile(r'^[A-Z]{3}[-\s]?\d{3}[-\s]?[A-Z]$')

def is_valid_plate(s):
    if not s:
        return False
    s = s.replace(" ", "").replace("-", "").upper()
    return bool(re.fullmatch(r'[A-Z]{3}\d{3}[A-Z]', s))
