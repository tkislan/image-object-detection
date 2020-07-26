from typing import Tuple
import numpy as np

def adjust_detection_box(raw_image_np: np.ndarray, box: np.ndarray, dimensions: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    height, width, _ = raw_image_np.shape

    height_ratio = (dimensions[0][1] - dimensions[0][0]) / height
    height_offset = dimensions[0][0] / height
    width_ratio = (dimensions[1][1] - dimensions[1][0]) / width
    width_offset = dimensions[1][0] / width

    return np.array([
        box[0] * height_ratio + height_offset,
        box[1] * width_ratio + width_offset,
        box[2] * height_ratio + height_offset,
        box[3] * width_ratio + width_offset
    ])
