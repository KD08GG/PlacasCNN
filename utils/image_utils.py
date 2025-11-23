# utils/image_utils.py
import cv2
import numpy as np

def resize_and_pad(img, size, pad_value=255):
    h, w = img.shape[:2]
    scale = size / max(h,w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size,size) + (() if resized.ndim==2 else (3,),), pad_value, dtype=resized.dtype)
    top = (size - nh)//2
    left = (size - nw)//2
    if resized.ndim == 2:
        canvas[top:top+nh, left:left+nw] = resized
    else:
        canvas[top:top+nh, left:left+nw, :] = resized
    return canvas
