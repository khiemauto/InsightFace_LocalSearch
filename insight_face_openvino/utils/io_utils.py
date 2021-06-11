import yaml
import os

import cv2
import numpy as np


def read_yaml(path: str):
    with open(path, "r") as fp:
        res = yaml.load(fp, Loader=yaml.FullLoader)
    return res


def read_image(path: str):

    """Reads an image in RGB format."""

    assert os.path.exists(path)
    image = cv2.imread(path)
    assert image is not None, f"{path} returns empty image."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def read_image_from_bytes(buf: bytes):
    """Reads an image in RGB format."""
    image = np.frombuffer(buf, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    assert image is not None, f"buffer returns empty image."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image(image: np.ndarray, path: str):

    """Saves an image in RGB format"""

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
