from typing import Optional
import threading
import cv2
import numpy as np
import time
import queue

from image_object_detection.camera.image import CameraImageContainer, get_split_image_dimensions

class CameraFeed(threading.Thread):
    def __init__(self, url: str, image_queue: 'queue.Queue[CameraImageContainer]'):
        super().__init__()
        self._url = url
        self._image_queue = image_queue
        self._should_read = threading.Event()
        self._should_exit = threading.Event()
    
    def run(self):
        while not self._should_exit.is_set():
            cap: Optional[cv2.VideoCapture] = None

            try:
                if not self._should_read.wait(timeout=1):
                    continue

                cap = cv2.VideoCapture(self._url)

                while self._should_read.is_set():
                    ret, raw_image_np = cap.read()

                    if ret is not True:
                        print('Failed to read image from camera')
                        time.sleep(5)
                        break
                
                    raw_image_np = raw_image_np[...,::-1] #  Convert from BGR to RGB
                    raw_image_np = raw_image_np.astype(np.uint8)

                    image_container = CameraImageContainer.create(raw_image_np, get_split_image_dimensions(raw_image_np))

                    print('Putting image into queue')
                    try:
                        self._image_queue.put(image_container, block=True, timeout=1)
                        print('Image put into queue')
                    except queue.Full:
                        print('Image Queue full')
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
        print('Stopping camera feed')
        self._should_read.clear()
        self._should_exit.set()