from pathlib import Path
import cv2
from typing import (
    TYPE_CHECKING,
    Union
)

import numpy as np
from PIL import Image

from blur_utils._exceptions import (
    ImageReadError,
    VideoCaptureError
)

if TYPE_CHECKING:
    from blur_utils._typing import ImageFile, VideoFile

def _validate_media_file_path(media_path: Union[str, Path]) -> Path:
    """Private generic function to validate file path of image or video."""
    if isinstance(media_path, str):
        media_path = Path(media_path)

    # Check if file exists, otherwise throw a generic error
    if not media_path.exists():
        raise FileNotFoundError(f'No such file {media_path}')
    
    return media_path


def convert_BGR(image_array: np.ndarray) -> np.ndarray:
    """
    Converts an nparray instance from RGB format into BGR format
    using open-cv functionality.

    Args:
        image_array (`np.ndarray`): An nparray instance in an RGB format.

    Returns:
        `np.ndarray`: A `np.ndarray` instance in a BGR format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def convert_RGB(image_array: np.ndarray) -> np.ndarray:
    """
    Converts an nparray instance from BGR format into RGB format
    using open-cv functionality.

    Args:
        image_array (`np.ndarray`): An nparray instance in an BGR format.

    Returns:
        `np.ndarray`: A `np.ndarray` instance in a RGB format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def load_image(image_file: 'ImageFile', load_bgr: bool = True) -> np.ndarray:
    """
    Given an `ImageFile` instance, which is either an instance of str,
    `Path`, `np.ndarray` or `Image.Image`, will load the image and
    return a `np.ndarray` instance.
    
    Args:
        image_file (`ImageFile`): An ImageFile instance, which is either an
            instance of str, `Path`, `np.ndarray`, or `Image.Image`.
        load_bgr (bool): A boolean indicating whether to convert to keep in
            BGR when loading the image. 

    Returns:
        `np.ndarray`: A `np.ndarray` instance which is an ndarray representation
            of the image.
    """
    image_array: np.ndarray

    if isinstance(image_file, (str, Path)):
        # Check if file exists, otherwise throw a generic error
        image_file = _validate_media_file_path(media_path=image_file)

        # Since file exists, read in the image
        image_array = cv2.imread(filename=str(image_file))

        # Ensure the image is not None, which would indicate that an
        # error was encountered when loading
        if image_array is None:
            raise ImageReadError(
                'Error with opening image file, loading was not successful'
            )
    elif isinstance(image_file, np.ndarray):
        image_array = image_file
    elif isinstance(image_file, Image.Image):
        image_array = np.array(image_file)
    else:
        raise TypeError(
            'Expected a string, Path, np.ndarray, or Image.Image object, '
            f'but got {type(image_file).__name__}'
        )

    if not load_bgr:
        image_array = convert_RGB(image_array=image_array)
    
    return image_array


def load_video(video_file: 'VideoFile') -> cv2.VideoCapture:
    """
    Given a `VideoFile` instance, which is either an instance of str,
    Path, or `cv2.VideoCapture`, will load the video and return a
    `cv2.VideoCapture` instance.
    
    Args:
        video_file (`VideoFile`): A `VideoFile` instance, which is either an
            instance of str, Path, or `cv2.VideoCapture`.

    Returns:
        `cv2.VideoCapture`: A `cv2.VideoCapture` instance representing the
            loaded video file.
    """
    v_capture: cv2.VideoCapture
    if isinstance(video_file, (str, Path)):
        # Check if file exists, otherwise throw a generic error
        video_file = _validate_media_file_path(media_path=video_file)
        v_capture = cv2.VideoCapture(filename=str(video_file))
    elif isinstance(video_file, cv2.VideoCapture):
        v_capture = video_file
    else:
        raise TypeError(
            'Expected a string, Path, or cv2.VideoCapture object, '
            f'but got {type(video_file).__name__}'
        )

    if not v_capture.isOpened():
        raise VideoCaptureError(
            'Error with opening video file, capture was not opened'
        )
    return v_capture