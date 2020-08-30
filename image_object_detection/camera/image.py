from typing import List, Optional, Tuple

import cv2
import numpy as np

from image_object_detection.detection.config import TENSORRT
from image_object_detection.utils.model_data import ModelData
from image_object_detection.utils.timing import get_current_time_millis


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
        camera_name: str,
        raw_image_np: np.ndarray,
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        cropped_images: List[np.ndarray],
        detailed: bool,
        created_at: int
    ):
        self.camera_name = camera_name
        self.raw_image_np = raw_image_np
        self.dimensions = dimensions
        self.cropped_images = cropped_images
        self.detailed = detailed
        # self.created_at = time.perf_counter() * 1000
        self.created_at = created_at
    
    @classmethod
    def create(
        cls,
        camera_name: str,
        raw_image_np: np.ndarray,
        dimensions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        detailed: bool = False,
        created_at: Optional[int] = None
    ) -> 'CameraImageContainer':
        if created_at is None:
            created_at = get_current_time_millis()

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
        return cls(camera_name, raw_image_np, dimensions, cropped_images, detailed, created_at)
