import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from image_object_detection.camera.image import CameraImageContainer, get_split_image_dimensions
from image_object_detection.workers.base import BaseWorker


class CameraFeedWorker(BaseWorker):
    def __init__(self, name: str, url: str, image_queue: 'queue.Queue[CameraImageContainer]'):
        super().__init__()
        self._name = name
        self._url = url
        self._image_queue = image_queue
        self._should_read = threading.Event()
    
    def start(self):
        self.enable_read()
        super().start()
    
    def run_processing(self):
        cap: Optional[cv2.VideoCapture] = None

        try:
            if not self._should_read.wait(timeout=1):
                return

            cap = cv2.VideoCapture(self._url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            time.sleep(1)

            while self._should_read.is_set():
                ret = cap.grab()

                if ret is not True:
                    print('Failed to grab image from camera')
                    time.sleep(5)
                    break

                if not self._image_queue.empty():
                    continue

                ret, raw_image_np = cap.retrieve()
                if ret is not True:
                    print('Failed to retrieve image from camera')
                    time.sleep(5)
                    break
            
                raw_image_np = raw_image_np[...,::-1] #  Convert from BGR to RGB
                raw_image_np = raw_image_np.astype(np.uint8)

                image_container = CameraImageContainer.create(self._name, raw_image_np, get_split_image_dimensions(raw_image_np))

                print('Putting image into queue')
                try:
                    self._image_queue.put_nowait(image_container)
                except queue.Full:
                    pass
        except Exception as error:
            print('Camera feed failed')
            print(error)
            if cap is not None:
                cap.release()
            time.sleep(5)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception as error:
                print('Failed to release camera feed')
                print(error)
    
    def enable_read(self):
        if self._should_exit.is_set():
            raise RuntimeError("Can't enable read on destroyed CameraFeed")
        self._should_read.set()
    
    def disable_read(self):
        self._should_read.clear()
    
    def stop(self):
        super().stop()
        self.disable_read()
