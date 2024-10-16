from pathlib import Path
from cv2.typing import MatLike
import cv2
from typing import (
    Any,
    TYPE_CHECKING,
    Union
)

import numpy as np
from numpy.typing import NDArray
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


def convert_BGR(image_array: NDArray[Any]) -> MatLike:
    """
    Converts an nparray instance from RGB format into BGR format
    using open-cv functionality.

    Args:
        image_array (NDArray[Any]): An nparray instance in an RGB format.

    Returns:
        `MatLike`: A `MatLike` instance in a BGR format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def convert_RGB(image_array: NDArray[Any]) -> MatLike:
    """
    Converts an nparray instance from BGR format into RGB format
    using open-cv functionality.

    Args:
        image_array (NDArray[Any]): An nparray instance in an BGR format.

    Returns:
        `MatLike`: A `MatLike` instance in a RGB format.
    """
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def load_image(image_file: 'ImageFile', load_bgr: bool = True) -> MatLike:
    """
    Given an `ImageFile` instance, which is either an instance of str,
    `Path`, `Matlike`, `Image.Image`, or NDArray[Any], will load the image and
    return a `MatLike` instance.
    
    Args:
        image_file (`ImageFile`): An ImageFile instance, which is either an
            instance of str, `Path`, `Matlike`, `Image.Image`, or NDArray[Any].
        load_bgr (bool): A boolean indicating whether to convert to keep in
            BGR when loading the image. 

    Returns:
        `MatLike`: A `MatLike` instance which is an ndarray representation
            of the image.
    """
    image_array: Union[MatLike, NDArray[Any]]

    if isinstance(image_file, np.ndarray):
        image_array = image_file
    elif isinstance(image_file, Image.Image):
        image_array = np.array(image_file)
    else:
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
    if isinstance(video_file, cv2.VideoCapture):
        v_capture = video_file
    else:
        # Check if file exists, otherwise throw a generic error
        video_file = _validate_media_file_path(media_path=video_file)
        v_capture = cv2.VideoCapture(filename=str(video_file))

    if not v_capture.isOpened():
        raise VideoCaptureError(
            'Error with opening video file, capture was not opened'
        )
    return v_capture