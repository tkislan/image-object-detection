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