from __future__ import annotations

from pathlib import Path
import cv2
from cv2.typing import MatLike
from dataclasses import dataclass
from typing import (
    Any,
    Tuple,
    TypeAlias,
    Union
)

from numpy.typing import NDArray
from PIL import Image

from face_blur_utils._utils import load_video
from face_blur_utils._settings import (
    AverageBlurSettings,
    GaussianBlurSettings,
    MedianBlurSettings
)

VideoFile: TypeAlias = Union[
    str,
    Path,
    cv2.VideoCapture
]

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
    def from_x_y_w_h(
        cls,
        x: int,
        y: int,
        w: int,
        h: int
    ) -> DetectedBBox:
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
    

@dataclass
class VideoOutput:
    """"""
    video_path: Path
    frames: int
    fps: int

    @property
    def load_video(self) -> cv2.VideoCapture:
        """"""
        return load_video(video_file=self.video_path)
    
    def display_video(self) -> None:
        """"""
        video_capture: cv2.VideoCapture = self.load_video
        cap_exhausted = False

        while not cap_exhausted:
            cap_exhausted, frame = video_capture.read()

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()