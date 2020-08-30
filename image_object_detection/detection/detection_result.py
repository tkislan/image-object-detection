import itertools
from typing import List
from image_object_detection.camera.image import CameraImageContainer
from image_object_detection.utils.detection_output import DetectionOutput

class DetectionResult:
    def __init__(
        self,
        image_container: CameraImageContainer,
        image_results: List[List[DetectionOutput]],
    ):
        self.image_container = image_container
        self.image_results = image_results
    
    @classmethod
    def empty(cls, image_container: CameraImageContainer) -> 'DetectionResult':
        return cls(image_container, [])
    
    @property
    def has_detection(self) -> bool:
        return any(len(results) > 0 for results in self.image_results)