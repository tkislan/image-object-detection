import unittest
import os

from image_object_detection.app import DETECTION_MODEL_PATH
from image_object_detection.detection.detection import detect
from image_object_detection.detection.image import load_image
from image_object_detection.detection.session import create_detection_session

IMAGE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street.jpg')
IMAGE_FULL_HD_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street_fullhd.jpg')
IMAGE_4K_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street_4k.jpg')
IMAGE_NO_DETECTION_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'house.jpg')


class DetectionTest(unittest.TestCase):

    def setUp(self):
        self.tf_sess = create_detection_session(DETECTION_MODEL_PATH)

    def test_image_detection(self):
        image = load_image(IMAGE_PATH)

        detected_image, classes = detect(self.tf_sess, image)

        self.assertEqual((480, 640, 3), detected_image.shape)
        self.assertListEqual(['car', 'person'], sorted(classes))
        self.assertIn('person', classes)

    def test_image_fullhd_detection(self):
        image = load_image(IMAGE_FULL_HD_PATH)

        detected_image, classes = detect(self.tf_sess, image)

        self.assertEqual((1280, 1920, 3), detected_image.shape)
        # self.assertListEqual(['car', 'person'], sorted(classes))
        self.assertIn('person', classes)

    def test_image_4k_detection(self):
        image = load_image(IMAGE_4K_PATH)

        detected_image, classes = detect(self.tf_sess, image)

        self.assertEqual((2560, 3840, 3), detected_image.shape)
        # self.assertListEqual(['car', 'person'], sorted(classes))
        self.assertIn('person', classes)

    def test_image_no_detection(self):
        image = load_image(IMAGE_NO_DETECTION_PATH)

        detected_image, classes = detect(self.tf_sess, image)

        self.assertEqual((375, 500, 3), detected_image.shape)
        self.assertListEqual([], sorted(classes))
