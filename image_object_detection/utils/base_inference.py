from image_object_detection.utils.detection_output import DetectionOutput
from typing import List
import numpy as np
from image_object_detection.utils.detection_output import DetectionOutput


class BaseInferenceWrapper:
    def infer(self, img_np: np.ndarray) -> List[DetectionOutput]:
        raise NotImplemented()

    def infer_batch(self, imgs_np: List[np.ndarray]) -> List[List[DetectionOutput]]:
        raise NotImplemented()