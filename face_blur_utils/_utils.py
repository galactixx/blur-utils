from cv2.typing import MatLike
import cv2
from typing import (
    Any,
    Union
)

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_blur_utils._exceptions import VideoCaptureError
from face_blur_utils._typing import ImageFile, VideoFile

def convert_BGR(image_array: NDArray[Any]) -> MatLike:
    """
    Converts an nparray instance from RGB format into BGR format
    using open-cv functionality.

    Args:
        image_array (NDArray[Any]): An nparray instance in an RGB format.

    Returns:
        MatLike: A MatLike instance in a BGR format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def convert_RGB(image_array: NDArray[Any]) -> MatLike:
    """
    Converts an nparray instance from BGR format into RGB format
    using open-cv functionality.

    Args:
        image_array (NDArray[Any]): An nparray instance in an BGR format.

    Returns:
        MatLike: A MatLike instance in a RGB format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def load_image(image_file: ImageFile) -> MatLike:
    """
    Given an ImageFile instance, which is either an instance of str,
    Path, Matlike, Image.Image, or NDArray[Any], will load the image and
    return a MatLike instance.
    
    Args:
        image_file (ImageFile): Am ImageFile instance, which is either an
            instance of str, Path, Matlike, Image.Image, or NDArray[Any].

    Returns:
        MatLike: A MatLike instance which is an ndarray representation
            of the image.
    """
    image_array: Union[MatLike, NDArray[Any]]

    if isinstance(image_file, np.ndarray):
        image_array = image_file
    elif isinstance(image_file, Image.Image):
        image_array = np.array(image_file)
    else:
        image_array = cv2.imread(filename=image_file)

    return convert_RGB(image_array=image_array)


def load_video(video_file: VideoFile) -> cv2.VideoCapture:
    """
    Given a VideoFile instance, which is either an instance of str,
    Path, or cv2.VideoCapture, will load the video and return a
    cv2.VideoCapture instance.
    
    Args:
        video_file (VideoFile): A VideoFile instance, which is either an
            instance of str, Path, or cv2.VideoCapture.

    Returns:
        cv2.VideoCapture: A cv2.VideoCapture instance representing the
            loaded video file.
    """
    v_capture: cv2.VideoCapture
    if isinstance(video_file, cv2.VideoCapture):
        v_capture = video_file
    else:
        v_capture = cv2.VideoCapture(video_file)

    if not v_capture.isOpened():
        raise VideoCaptureError(
            'Error with opening video file, capture was not opened'
        )
    return v_capture