import unittest
import os

from app import DETECTION_MODEL_PATH
from detection.detection import detect
from detection.image import load_image
from detection.session import create_detection_session

IMAGE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street.jpg')


class DetectionTest(unittest.TestCase):

    def setUp(self):
        self.tf_sess = create_detection_session(DETECTION_MODEL_PATH)

    def test_image_load(self):
        image = load_image(IMAGE_PATH)

        detected_image, classes = detect(self.tf_sess, image)

        self.assertEqual((480, 640, 3), detected_image.shape)
        self.assertListEqual(['car', 'person'], sorted(classes))
