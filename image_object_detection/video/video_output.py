from image_object_detection.utils.timing import get_current_time_millis
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np


VIDEO_CODEC = 'mp4v'
# VIDEO_CODEC = 'avc1'
# VIDEO_CODED = 'MJPG'
VIDEO_FILE_EXTENSION = 'mp4'
# VIDEO_FILE_EXTENSION = 'avi'


class VideoOutput:
    def __init__(self, prefix: str, fps: float, dimensions: Tuple[int, int]) -> None:
        self._fps = fps
        self._video_writer: cv2.VideoWriter = self.create_video_writer(prefix, fps, dimensions)
        self._last_write: int = 0
        self._frames_written: int = 0
        self._first_write: Optional[int] = None
    
    @staticmethod
    def get_file_name(prefix: str) -> str:
        return f'{prefix}_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.{VIDEO_FILE_EXTENSION}'
    
    @classmethod
    def create_video_writer(cls, prefix: str, fps: float, dimensions: Tuple[int, int]) -> cv2.VideoWriter:
        return cv2.VideoWriter(
            cls.get_file_name(prefix),
            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
            fps,
            dimensions,
        )
    
    @property
    def fps(self) -> float:
        return self._fps
    
    def missing_frames_count(self, time_ref_ms: int) -> int:
        if self._first_write is None:
            return 1
        return int((time_ref_ms - self._first_write) / 1000 * self.fps) - self._frames_written

    def close(self):
        self._video_writer.release()
    
    def write(self, img: np.ndarray):
        self._video_writer.write(img)
        self._frames_written += 1
        self._last_write = get_current_time_millis()
        if self._first_write is None:
            self._first_write = get_current_time_millis()
    