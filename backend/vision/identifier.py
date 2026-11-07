#
# File: identifier.py
#
# Description:  Uses OpenCV CNN and ORB to get a confidence
#               match recognizing a brick.
#
###################################################################
from cv2.typing import MatLike
import numpy as np
import cv2 as cv
import onnxruntime as ort


def bytes_to_image(byte_data: bytes) -> MatLike | None:
    """
    Converts the bytes returned by the UploadFile
    into an image usable by OpenCV

    :param byte_data: The bytes of the image
    :return: A MatLike object OpenCV can use
    """
    nparr = np.frombuffer(byte_data, np.uint8)
    return cv.cvtColor(cv.imdecode(nparr, cv.IMREAD_COLOR), cv.COLOR_RGB2BGR)


def load_models():
    """
    Loads both the detection and identification models.
    """
    pass


def cleanup_models():
    """
    Properly deloads the models before program halting
    """
    pass


def identify_brick(image: MatLike, bb: tuple[int,int,int,int]) -> tuple[str, float]:
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
    pass


def _identify_orb(image: MatLike) -> float:
    """
    Compares the image against all known bricks to get
    a confidence value.

    :param image: The image to check
    :type image: cv.MatLike
    :return: The confidence between 0.0 - 1.0 where 0.0 is no match and 1.0 is match
    :rtype: float
    """
    pass
    # img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
 
    # # Initiate ORB detector
    # orb = cv.ORB_create()
    
    # # find the keypoints with ORB
    # kp = orb.detect(img,None)
    
    # # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)