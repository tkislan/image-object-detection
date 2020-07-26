from image_object_detection.utils.model_data import ModelData
from typing import List, Tuple

import cv2
import numpy as np

from image_object_detection.detection.config import TENSORRT


def split_image_into_squares(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width, _ = image.shape
    return (
        image[:,:height,:],
        image[:,width-height:,:]
    )

def get_split_image_dimensions(image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    height, width, _ = image.shape
    return [
        ((0, height), (0, height)),
        ((0, height), (width-height, width)),
    ]


def transform_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, ModelData.INPUT_SHAPE[1:], interpolation=cv2.INTER_CUBIC)

    image = image.transpose((2, 0, 1))
    # Normalize to [-1.0, 1.0] interval (expected by model)
    image = (2.0 / 255.0) * image - 1.0
    image = image.ravel()
    return image

class CameraImageContainer:
    def __init__(
        self,
        raw_image_np: np.ndarray,
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        cropped_images: List[np.ndarray],
        detailed: bool
    ):
        self.raw_image_np = raw_image_np
        self.dimensions = dimensions
        self.cropped_images = cropped_images
        self.detailed = detailed
    
    @classmethod
    def create(
        cls,
        raw_image_np: np.ndarray,
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        detailed: bool = False
    ) -> 'CameraImageContainer':
        cropped_images = [
            raw_image_np[
                crop_image_dimensions[0][0]:crop_image_dimensions[0][1],
                crop_image_dimensions[1][0]:crop_image_dimensions[1][1],
                :
            ]
            for crop_image_dimensions in dimensions
        ]
        if TENSORRT:
            cropped_images = [
                transform_image(image) for image in cropped_images
            ]
        return cls(raw_image_np, dimensions, cropped_images, detailed)
