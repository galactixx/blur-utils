from __future__ import annotations

from pathlib import Path
from cv2.typing import MatLike
from dataclasses import dataclass
from typing import (
    Any,
    Tuple,
    TypeAlias,
    Union
)

from PIL import Image
from numpy.typing import NDArray

from face_blur_utils._settings import (
    AverageBlurSettings,
    GaussianBlurSettings,
    MedianBlurSettings
)

ImageFile: TypeAlias = Union[
    str,
    Path,
    MatLike,
    Image.Image,
    NDArray[Any]
]

BlurSetting: TypeAlias = Union[
    AverageBlurSettings,
    GaussianBlurSettings,
    MedianBlurSettings
]

@dataclass
class DetectedBBox:
    """"""
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_x_y_w_h(cls, x: int, y: int, w: int, h: int) -> DetectedBBox:
        """"""
        right = x + w
        bottom = y + h
        return cls(left=x, top=y, right=right, bottom=bottom)

    @property
    def width(self) -> int:
        """"""
        return self.right - self.left
    
    @property
    def height(self) -> int:
        """"""
        return self.bottom - self.top
    
    @property
    def x_y_w_h(self) -> Tuple[int, int, int, int]:
        """"""
        return self.left, self.top, self.width, self.height