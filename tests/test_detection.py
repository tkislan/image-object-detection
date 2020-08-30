import os

import numpy as np
import pytest

from image_object_detection.camera.image import CameraImageContainer, get_split_image_dimensions
from image_object_detection.config.graph import GRAPH_PATH
from image_object_detection.detection.image import load_image, save_image
from image_object_detection.utils.inference_util import run_inference_container

BURGLAR_IMAGES = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'burglar1.png'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'burglar2.png'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'burglar3.png'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'burglar4.png'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'burglar5.png'),
]

@pytest.fixture
def inference_wrapper():
    import image_object_detection.utils.inference as inference_utils # TF inference wrappers
    yield inference_utils.TensorflowInference(GRAPH_PATH)

def test_basic_detection(inference_wrapper):
    raw_image_np = load_image(BURGLAR_IMAGES[3])
    raw_image_np = load_image('__playground/img/cap_1590933177.jpg')
    raw_image_np = raw_image_np.astype(np.uint8)
    image_container = CameraImageContainer.create('some_camera', raw_image_np, get_split_image_dimensions(raw_image_np))

    detection_result, detailed_image_containers = run_inference_container(inference_wrapper, image_container)

    print(detection_result)
    print(detailed_image_containers)

    assert len(detailed_image_containers) == 1

    save_image(detailed_image_containers[0].cropped_images[0], 'cropped_image.jpg')

    detection_result, detailed_image_containers = run_inference_container(inference_wrapper, detailed_image_containers[0])

    print(detection_result)
    print(detailed_image_containers)
