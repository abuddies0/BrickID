#
# File: identifier.py
#
# Description:  Uses OpenCV CNN and ORB to get a confidence
#               match recognizing a brick.
#
###################################################################
import numpy as np
import cv2 as cv


def identify_brick(image: cv.MatLike) -> float:
    """
    Figures out how close the given image is to the brick.

    :param image: The image of the brick to check
    :type image: cv.MatLike
    :return: The confidence between 0.0 - 1.0 where 0.0 is no match and 1.0 is match
    :rtype: float
    """
    # TODO: Implement this
    pass


def _identify_orb(image: cv.MatLike) -> float:
    """
    Compares the image against all known bricks to get
    a confidence value.

    :param image: The image to check
    :type image: cv.MatLike
    :return: The confidence between 0.0 - 1.0 where 0.0 is no match and 1.0 is match
    :rtype: float
    """
    img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
 
    # Initiate ORB detector
    orb = cv.ORB_create()
    
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)


def _identify_cnn(image: cv.MatLike) -> float:
    """
    Compares the image against all known bricks to get
    a confidence value.

    :param image: The image to check
    :type image: cv.MatLike
    :return: The confidence between 0.0 - 1.0 where 0.0 is no match and 1.0 is match
    :rtype: float
    """
    # TODO: Implement this
    pass