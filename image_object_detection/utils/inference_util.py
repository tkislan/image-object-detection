import enum
import time
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np

from image_object_detection.camera.image import (CameraImageContainer)
from image_object_detection.detection.boxes import adjust_detection_box
from image_object_detection.detection.detection_result import DetectionResult
from image_object_detection.utils.base_inference import BaseInferenceWrapper


def run_inference_container(
    inference_wrapper: BaseInferenceWrapper,
    image_container: CameraImageContainer,
) -> Tuple[Optional[DetectionResult], List[CameraImageContainer]]:
    start = time.perf_counter()
    detection_outputs_per_image = inference_wrapper.infer_batch(image_container.cropped_images)
    print(f"Inference finished in {time.perf_counter() - start:0.4f} seconds")

    detection_result = DetectionResult(image_container, detection_outputs_per_image)
    denormalize_boxes(detection_result)

    status = analyze_detection(detection_result)

    if status == DetectionStatus.YES:
        cut_off_threshold(detection_result, 0.80)
        return detection_result, []
    elif status == DetectionStatus.MAYBE and not image_container.detailed:
        dimensions_per_size = get_split_detail_image_dimensions(detection_result)

        return None, [
            CameraImageContainer.create(image_container.raw_image_np, dimensions, detailed=True)
            for dimensions in dimensions_per_size.values()
        ]
    else:
        return None, []


class DetectionStatus(int, enum.Enum):
    NO = 0
    MAYBE = 1
    YES = 2


def analyze_detection(result: DetectionResult) -> DetectionStatus:
    maybe = False
    for r in chain(*result.image_results):
        if r.confidence > 0.7:
            return DetectionStatus.YES
        elif r.confidence > 0.15:
            maybe = True
    
    return DetectionStatus.MAYBE if maybe else DetectionStatus.NO

def denormalize_boxes(detection_result: DetectionResult):
    height, width, _ = detection_result.image_container.raw_image_np.shape

    for i, image_result in enumerate(detection_result.image_results):
        for result in image_result:
            box = adjust_detection_box(detection_result.image_container.raw_image_np, result.box, detection_result.image_container.dimensions[i])
            # TODO - adjust for resized image here !!!
            result.box = box[0] * height, box[1] * width, box[2] * height, box[3] * width

def cut_off_threshold(detection_result: DetectionResult, threshold: float):
    detection_result.image_results = [
        [output for output in result if output.confidence >= threshold]
        for result in detection_result.image_results
    ]


def get_image_dimensions_from_box(raw_height: int, raw_width: int, box: np.ndarray) -> Optional[Tuple[int, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    height = box[2] - box[0]
    width = box[3] - box[1]
    size = int(max(height, width) * 1.3)

    height_limit = raw_height * 0.7
    width_limit = raw_width * 0.7
    size_limit = int(min(height_limit, width_limit))

    if size > size_limit:
        return None
    elif size > size_limit * 2/3:
        size = size_limit
    elif size > size_limit * 1/3:
        size = int(size_limit * 2/3)
    else:
        size = int(size_limit * 1/3)
    top = max(int(box[0] - (size - height) / 2), 0)
    left = max(int(box[1] - (size - width) / 2), 0)
    return (size, ((top, top + size), (left, left + size)))

def get_split_detail_image_dimensions(
    detection_result: DetectionResult
) -> Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    dimensions: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = defaultdict(list)
    raw_height, raw_width, _ = detection_result.image_container.raw_image_np.shape
    for image_result in detection_result.image_results:
        for result in image_result:
            size_dimension = get_image_dimensions_from_box(raw_height, raw_width, result.box)
            if size_dimension is None:
                continue
            size, dimension = size_dimension
            dimensions[size].append(dimension)
    
    return dimensions
