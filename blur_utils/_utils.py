from cv2.typing import MatLike
import cv2
from typing import (
    Any,
    Union
)

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from blur_utils._typing import ImageFile

def convert_BGR(image_array: NDArray[Any]) -> MatLike:
    """"""
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def convert_RGB(image_array: NDArray[Any]) -> MatLike:
    """"""
    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)


def load_image(image_file: ImageFile) -> MatLike:
    """"""
    image_array: Union[MatLike, NDArray[Any]]

    if isinstance(image_file, np.ndarray):
        image_array = image_file
    elif isinstance(image_file, Image.Image):
        image_array = np.array(image_file)
    else:
        image_array = cv2.imread(filename=image_file)

    return convert_RGB(image_array=image_array)