import os
import unittest

from PIL import Image

from detection.image import convert_image_to_array, load_image

IMAGE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'img', 'street.jpg')


class ImageTest(unittest.TestCase):

    def test_convert_image_numpy(self):
        image = Image.open(IMAGE_PATH)

        image_np = convert_image_to_array(image)

        self.assertEqual((480, 640, 3), image_np.shape)
        self.assertEqual(921600, image_np.size)
        self.assertEqual(480 * 640 * 3, image_np.size)

    def test_image_load(self):
        image = load_image(IMAGE_PATH)

        self.assertEqual((480, 640, 3), image.shape)
