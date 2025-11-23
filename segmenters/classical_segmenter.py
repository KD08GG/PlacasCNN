# segmenters/classical_segmenter.py
import cv2
import numpy as np

def sort_by_x(bbox):
    x, y, w, h = bbox
    return x

class ClassicalSegmenter:
    """
    Segmentador por procesamiento clásico:
    - grayscale -> threshold -> morphology -> connected components
    - retorna lista de recortes (chars) ordenados por X
    """
    def __init__(self, min_w=8, min_h=12):
        self.min_w = min_w
        self.min_h = min_h

    def segment(self, plate_img):
        img = plate_img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Resize height to ~100 for consistent CC detection
        h = 100
        scale = h / gray.shape[0]
        gray_resized = cv2.resize(gray, (int(gray.shape[1]*scale), h), interpolation=cv2.INTER_LINEAR)
        # adaptive threshold
        th = cv2.adaptiveThreshold(gray_resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,10)
        # morphology to join parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        # find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        Horig = img.shape[0]
        Worig = img.shape[1]
        for c in contours:
            x,y,w,hc = cv2.boundingRect(c)
            # filter by size (in resized coords)
            if w < 6 or hc < 12:
                continue
            # map back to original image coords
            x0 = int(x / scale)
            y0 = int(y / scale)
            w0 = int(w / scale)
            h0 = int(hc / scale)
            boxes.append((x0,y0,w0,h0))
        if not boxes:
            return []
        boxes = sorted(boxes, key=sort_by_x)
        chars = []
        for (x,y,w,h) in boxes:
            ch = plate_img[y:y+h, x:x+w]
            chars.append(ch)
        return chars
