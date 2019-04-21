import numpy as np
from PIL import Image


def load_image(file_path: str) -> np.array:
    print('Opening image')
    image = Image.open(file_path)

    print('Converting image to array')
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(image)

    return image_np


def save_image(image: np.array, file_path: str):
    im = Image.fromarray(image)
    im.save(file_path)
