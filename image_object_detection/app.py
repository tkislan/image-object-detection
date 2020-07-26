import os
import queue
import time
from typing import List

from image_object_detection.camera.image import CameraImageContainer
from image_object_detection.detection.config import TENSORRT
from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.utils.base_inference import BaseInferenceWrapper
from image_object_detection.workers.base import BaseWorker
from image_object_detection.workers.camera import CameraFeedWorker
from image_object_detection.workers.detection import DetectionWorker
from image_object_detection.workers.visualizer import VisualizerWorker

# Model used for inference
WORKSPACE_DIR = '/workspace'
CAMERA_URLS = os.environ.get('CAMERA_URLS', 'rtsp://admin:Hikvision@localhost:5554/ch1/main/av_stream').split(',')

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}

def get_inference_wrapper() -> BaseInferenceWrapper:
    if TENSORRT == True:
        import image_object_detection.utils.trt_inference as trt_inference_utils # TRT inference wrappers

        return trt_inference_utils.TRTInference('/opt/models/engine.bin')
    else:
        import image_object_detection.utils.inference as inference_utils # TF inference wrappers
        return inference_utils.TensorflowInference('/opt/models/graph.pb')

def main():
    print('initializing session')
    inference_wrapper = get_inference_wrapper()
    print('initialization done')

    QUEUE_MAXSIZE_PER_CAMERA = 5
    camera_urls = ['rtsp://admin:Hikvision@localhost:5554/ch1/main/av_stream']
    # camera_urls = [
    #     'rtsp://admin:Hikvision@192.168.1.200:554/ch1/main/av_stream',
    #     'rtsp://admin:Hikvision@192.168.1.201:554/ch1/main/av_stream',
    # ]

    image_queue: 'queue.Queue[CameraImageContainer]' = queue.Queue(maxsize=QUEUE_MAXSIZE_PER_CAMERA * len(camera_urls))
    result_queue: 'queue.Queue[DetectionResult]' = queue.Queue()

    workers: List[BaseWorker] = [
        *[CameraFeedWorker(camera_url, image_queue) for camera_url in camera_urls], 
        VisualizerWorker(result_queue), 
        DetectionWorker(inference_wrapper, image_queue, result_queue),
    ]

    for worker in workers:
        worker.start()
        if isinstance(worker, CameraFeedWorker):
            worker.enable_read()
    
    print('starting wait')
    time.sleep(3)

    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(error)
    print('Run time elapsed')

    for worker in workers:
        worker.stop()
    
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
