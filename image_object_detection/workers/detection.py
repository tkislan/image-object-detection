import queue
from typing import List

from image_object_detection.camera.image import CameraImageContainer
from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.utils.base_inference import BaseInferenceWrapper
from image_object_detection.utils.inference_util import run_inference_container
from image_object_detection.workers.base import BaseWorker


class DetectionWorker(BaseWorker):
    def __init__(
        self,
        inference_wrapper: BaseInferenceWrapper,
        image_queue: 'queue.Queue[CameraImageContainer]',
        result_queue: 'queue.Queue[DetectionResult]'
    ):
        super().__init__()
        self._inference_wrapper = inference_wrapper
        self._image_queue = image_queue
        self._result_queue = result_queue

    def run_processing(self):
        try:
            image_container = self._image_queue.get(block=True, timeout=1)

            print('running inference')
            detection_result, detailed_image_containers = run_inference_container(self._inference_wrapper, image_container)
            if detection_result is not None:
                self._result_queue.put(detection_result)
                return

            detailed_detection_results: List[DetectionResult] = []
            for detailed_image_container in detailed_image_containers:
                print('running detailed inference')
                detection_result, _ = run_inference_container(self._inference_wrapper, detailed_image_container)
                if detection_result is not None:
                    detailed_detection_results.append(detection_result)
            
            if len(detailed_detection_results) > 0:
                for detailed_detection_result in detailed_detection_results:
                    self._result_queue.put(detailed_detection_result)
            else:
                self._result_queue.put(DetectionResult.empty(image_container))
        except queue.Empty:
            pass
