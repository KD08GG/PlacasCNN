# recognizers/easyocr_fallback.py
import easyocr
import numpy as np

class EasyOCRFallback:
    def __init__(self, langs=['en']):
        self.reader = easyocr.Reader(langs, gpu=False)

    def read_plate(self, img):
        # img: numpy BGR or gray
        try:
            res = self.reader.readtext(img, detail=0, paragraph=False)
            if not res:
                return None
            # concatenar resultados
            return "".join(res).upper()
        except Exception as e:
            print(f"[EasyOCR] error: {e}")
            return None
