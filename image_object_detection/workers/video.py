import queue

from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.video.video_output_manager import VideoOutputManager
from image_object_detection.workers.base import BaseWorker
from image_object_detection.detection.image import save_image


class VideoWorker(BaseWorker):
    def __init__(self, video_output_manager: VideoOutputManager, visualized_queue: 'queue.Queue[DetectionResult]'):
        super().__init__()
        self._video_output_manager = video_output_manager
        self._visualized_queue = visualized_queue
    
    def run_processing(self):
        try:
            detection_result: DetectionResult = self._visualized_queue.get(block=True, timeout=1)

            self._video_output_manager.process_detection_result(detection_result)
        except queue.Empty:
            pass

        self._video_output_manager.check_videos_output_sizes()

    # def run_processing(self):
    #     start_time = get_current_time_millis()

    #     try:
    #         while True:
    #             detection_result: DetectionResult = self._visualized_queue.get_nowait()

    #             self._video_output_manager.process_detection_result(detection_result)

    #             # save_image(
    #             #     detection_result.image_container.raw_image_np,
    #             #     f'./camera_output_{int(round(time.time()))}.jpg'
    #             # )
    #     except queue.Empty:
    #         pass
    
    #     self._video_output_manager.run_once()
    
    #     elapsed_ms = get_current_time_millis() - start_time
    #     max_frame_run = 1000 / self._video_output_manager.max_fps

    #     next_run_in = max_frame_run - elapsed_ms

    #     if next_run_in <= 0:
    #         print(f'Video output processing taking too long ({elapsed_ms} ms / {max_frame_run} ms), reduce FPS')
    #     else:
    #         time.sleep(next_run_in / 1000)

    def teardown(self):
        self._video_output_manager.close_all()