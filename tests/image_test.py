import os
import unittest

from image_object_detection.detection.image import load_image

IMAGE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street.jpg')


class ImageTest(unittest.TestCase):

    def test_image_load(self):
        image = load_image(IMAGE_PATH)

        self.assertEqual((480, 640, 3), image.shape)
