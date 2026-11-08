#
# File: identifier.py
#
# Description:  Uses OpenCV CNN and ORB to get a confidence
#               match recognizing a brick.
#
###################################################################
import cv2 as cv
import numpy as np
from cv2.typing import MatLike


def bytes_to_image(byte_data: bytes) -> MatLike | None:
    """
    Converts the bytes returned by the UploadFile
    into an image usable by OpenCV

    :param byte_data: The bytes of the image
    :return: A MatLike object OpenCV can use
    """
    nparr = np.frombuffer(byte_data, np.uint8)
    bgr_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if bgr_image is None:
        return None
    return cv.cvtColor(bgr_image, cv.COLOR_RGB2BGR)


def load_models():
    """
    Loads both the detection and identification models.
    """
    return


def cleanup_models():
    """
    Properly deloads the models before program halting
    """
    return


def identify_brick(image: MatLike, bb: tuple[int, int, int, int]) -> tuple[str, float]:
    """
    Figures out how close the given image is to the brick.

    :param image: The image of the brick to check
    :type image: cv.MatLike
    :param bb: The bounding box of the brick to check (x_tl, y_tl, x_br, y_br)
    :return:    A tuple of [
                    The ID of the brick it matches
                    The confidence between 0.0 - 1.0 where 0.0 is no match and 1.0 is match,
                ]
    :rtype: float
    """
    # TODO: Implement this
    return ("WIP", -1.0)
