from __future__ import annotations

import cv2
import numpy as np


def inpaint(image: np.ndarray, mask: np.ndarray, radius: int = 3, method: str = "telea") -> np.ndarray:
    algorithm = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    binary_mask = (mask > 0).astype("uint8") * 255
    return cv2.inpaint(image, binary_mask, radius, algorithm)
