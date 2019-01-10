import numpy as np
from PIL import Image


def convert_image_to_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()) \
        .reshape((im_height, im_width, 3)) \
        .astype(np.uint8)


def load_image(file_path: str) -> np.array:
    image = Image.open(file_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = convert_image_to_array(image)

    return image_np


def save_image(image: np.array, file_path: str):
    im = Image.fromarray(image)
    im.save(file_path)


