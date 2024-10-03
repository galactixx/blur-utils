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
    """
    Represents a bounding box detected in an image, defined
    by its left, top, right, and bottom coordinates.

    This class provides utility methods for creating a bounding
    box  from (x, y) coordinates and width/height, and it also
    offers properties to access the width, height, and the
    (x, y, w, h) format.

    Args:
        left (int): The minimum x coordinate of thebounding box.
        top (int): The minimum y coordinate of the bounding box.
        right (int): The maximum x coordinate of the bounding box.
        bottom (int): The maximum y coordinate of the bounding box.
    """
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
        """
        Create an instance of DetectedBBox from the minimum x and
        y coordinates of the bounding box, as well as the width and
        height.

        Args:
            x (int): The minimum x coordinate.
            y (int): The minimum y coordinate.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.

        Returns:
            DetectedBBox: A DetectedBBox instance.
        """
        right = x + w
        bottom = y + h
        return cls(left=x, top=y, right=right, bottom=bottom)

    @property
    def width(self) -> int:
        """The width of the bounding box."""
        return self.right - self.left
    
    @property
    def height(self) -> int:
        """The height of the bounding box."""
        return self.bottom - self.top
    
    @property
    def x_y_w_h(self) -> Tuple[int, int, int, int]:
        """A tuple representing the bounding box in (x, y, width, height) format."""
        return self.left, self.top, self.width, self.height
    

@dataclass
class VideoOutput:
    """
    Simple dataclass to store metadata from video output.

    Args:
        video_path (Path): A Path instance representing the video path.
        frames (int): The number of frames in the video.
        fps (int): The frames-per-second of the video.
    """
    video_path: Path
    frames: int
    fps: int

    @property
    def load_video(self) -> cv2.VideoCapture:
        """Loads video and returns a cv2.VideoCapture instance."""
        return load_video(video_file=self.video_path)
    
    def display_video(self) -> None:
        """Loads and displays video from file path."""
        video_capture: cv2.VideoCapture = self.load_video
        cap_exhausted = False

        # Iterate through video until all frames are exhausted
        while not cap_exhausted:
            cap_exhausted, frame = video_capture.read()

            # Show each frame consecutively without break
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # Close all video/open-cv related windows
        video_capture.release()
        cv2.destroyAllWindows()