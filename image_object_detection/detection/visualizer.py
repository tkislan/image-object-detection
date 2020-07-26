from functools import reduce
from itertools import chain
from typing import Iterable, List

import cv2
import numpy as np

from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.utils.detection_output import DetectionOutput


BOXES_COLOR = (0, 255, 0)
BOXES_THICKNESS = 2
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1


class Visualizer:
    @staticmethod
    def get_unique_classes(detection_result: DetectionResult) -> List[str]:
        initial_value: List[str] = []
        def _add_unique_classes(acc: Iterable[str], image_result: List[DetectionOutput]) -> Iterable[str]:
            return chain(acc, (result.label for result in image_result))
        return list(set(reduce(_add_unique_classes, detection_result.image_results, initial_value)))
    

    @staticmethod
    def draw_detection_output(image_np: np.ndarray, detection_output: DetectionOutput):
        cv2.rectangle(
            image_np,
            (int(detection_output.box[1]), int(detection_output.box[0])),
            (int(detection_output.box[3]), int(detection_output.box[2])),
            BOXES_COLOR,
            BOXES_THICKNESS
        )

        confidence_percentage = "{0:.0%}".format(detection_output.confidence)
        label = "{}: {}".format(detection_output.label, confidence_percentage)

        label_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
        label_width, label_height = label_size[0]

        cv2.rectangle(
            image_np, 
            (
                int(detection_output.box[1]),
                int(detection_output.box[0])
            ),
            (
                int(detection_output.box[1]) + label_width,
                int(detection_output.box[0]) - label_height
            ),
            (0, 255, 0),
            cv2.FILLED
        )
        cv2.putText(
            image_np, 
            label, 
            (
                int(detection_output.box[1]),
                int(detection_output.box[0])
            ),
            FONT_FACE,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS
        )


    @classmethod
    def process_detection_result(cls, detection_result: DetectionResult):
        unique_classes = cls.get_unique_classes(detection_result)
        if not unique_classes:
            print('Nothing detected')
            return None

        for image_result in detection_result.image_results:
            for result in image_result:
                cls.draw_detection_output(detection_result.image_container.raw_image_np, result)
