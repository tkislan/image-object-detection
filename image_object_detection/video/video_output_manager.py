from typing import Dict, Tuple

import numpy as np

from image_object_detection.camera.image import CameraImageContainer
from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.utils.timing import get_current_time_millis
from image_object_detection.video.video_output import VideoOutput

VIDEO_LENGTH_WITHOUT_DETECTION = 20 * 1000 # milliseconds


class VideoOutputManager:
    def __init__(self, max_fps: float) -> None:
        self._max_fps = max_fps
        self._video_writers: Dict[str, VideoOutput] = {}
        self._images: Dict[str, CameraImageContainer] = {}
        self._frames: Dict[str, np.ndarray] = {}
        self._last_detections: Dict[str, int] = {}
    
    @property
    def max_fps(self) -> float:
        return self._max_fps
    
    def get_video_writer(self, prefix: str, dimensions: Tuple[int, int]) -> VideoOutput:
        video_writer = self._video_writers.get(prefix)
        if video_writer is None:
            print(f'Creating output: {prefix}, {dimensions}')
            video_writer = VideoOutput(prefix, self._max_fps, dimensions)
            self._video_writers[prefix] = video_writer
            self._last_detections[prefix] = get_current_time_millis()

        return video_writer

    def run_once(self):
        # Iterate through items first, as we might be deleting from map during loop
        for prefix, frame in list(self._frames.items()):
            # TODO - handle failure creating video writer
            # video_writer = self.get_video_writer(prefix, frame.shape[:2])
            video_writer = self.get_video_writer(prefix, (frame.shape[1], frame.shape[0]))
            # print('Writing output', frame.shape)
            video_writer.write(frame)
            
    # def process_detection_result(self, detection_result: DetectionResult):
    #     print('video processing detection result')
    #     prefix = detection_result.image_container.camera_name

    #     if detection_result.has_detection:
    #         self._last_detections[prefix] = get_current_time_millis()
    #     else:
    #         video_writer = self._video_writers.get(prefix)
    #         if video_writer is not None and self._last_detections.get(prefix, 0) < get_current_time_millis() - VIDEO_LENGTH_WITHOUT_DETECTION:
    #             self._close_video_writer(prefix)
    #             return
        
    #     self.set_frame(prefix, detection_result.image_container.raw_image_np)

    def process_detection_result(self, detection_result: DetectionResult):
        print('video processing detection result')
        current_image = detection_result.image_container
        prefix = current_image.camera_name
        shape = current_image.raw_image_np.shape
        dimensions = (shape[1], shape[0])

        if detection_result.has_detection:
            self._last_detections[prefix] = get_current_time_millis()
        
        video_writer = self.get_video_writer(prefix, dimensions)
        previous_image = self._images.get(prefix)

        missing_frames_count = video_writer.missing_frames_count(current_image.created_at)
        print(f'Missing frames count: {missing_frames_count}')

        if previous_image is not None:
            # elapsed = current_image.created_at - previous_image.created_at
            # frames_count = elapsed / video_writer.fps

            frame = previous_image.raw_image_np[...,::-1] #  Convert from BGR to RGB

            if missing_frames_count > 1:
                # TODO - track, and lower FPS for next videos, so we don't need to write same frames multiple times

                for _ in range(missing_frames_count - 1):
                    video_writer.write(frame)
        
        # This can happen if processing FPS is higher than video FPS
        # TODO - track and theoretically increase FPS on video
        if missing_frames_count > 0:
            video_writer.write(current_image.raw_image_np[...,::-1])
        
        self._images[prefix] = current_image

        
    
    def check_videos_output_sizes(self):
        for prefix in self._video_writers.keys():
            if self._last_detections.get(prefix, 0) < get_current_time_millis() - VIDEO_LENGTH_WITHOUT_DETECTION:
                self._close_video_writer(prefix)


    def set_frame(self, prefix: str, frame: np.ndarray):
        frame = frame[...,::-1] #  Convert from BGR to RGB
        self._frames[prefix] = frame
    
    def close_all(self):
        for video_writer in self._video_writers.values():
            video_writer.close()

    def _close_video_writer(self, prefix: str):
        try:
            self._video_writers[prefix].close()
            del self._video_writers[prefix]
        except KeyError:
            print('Trying to close non existing video writer')
            pass
