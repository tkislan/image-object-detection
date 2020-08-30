import queue

from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.detection.visualizer import Visualizer
from image_object_detection.workers.base import BaseWorker


class VisualizerWorker(BaseWorker):
    def __init__(self, result_queue: 'queue.Queue[DetectionResult]', visualized_queue: 'queue.Queue[DetectionResult]'):
        super().__init__()
        self._result_queue = result_queue
        self._visualized_queue = visualized_queue

    def run_processing(self):
        try:
            detection_result: DetectionResult = self._result_queue.get(block=True, timeout=1)

            Visualizer.process_detection_result(detection_result)

            try:
                self._visualized_queue.put(detection_result)
            except queue.Full:
                print('Visualized Queue full')
            
            # import time
            # from image_object_detection.detection.image import save_image
            # save_image(
            #     detection_result.image_container.raw_image_np,
            #     f'./camera_output_{int(round(time.time() * 1000))}.jpg'
            # )
        except queue.Empty:
            pass
