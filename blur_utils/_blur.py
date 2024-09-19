from __future__ import annotations

from abc import ABC, abstractmethod
import cv2
from cv2.typing import MatLike
from typing import (
    Any,
    Dict,
    Optional,
    Tuple
)

import numpy as np
from pydantic import BaseModel

from blur_utils._typing import BlurSetting, DetectedBBox
from blur_utils._settings import (
    AverageBlurSettings,
    BilateralFilterSettings,
    BoxFilterSettings,
    GaussianBlurSettings,
    MedianBlurSettings,
    MotionBlurSettings
)

class AbstractBlur(ABC):
    """"""
    def __init__(self, image: MatLike, settings: Optional[BaseModel] = None):
        self.image = image
        self._settings = settings

    @abstractmethod
    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def settings(self) -> Dict[str, Any]:
        """"""
        return self._settings.model_dump(by_alias=True)

    def apply_blur_to_face(self, bbox: DetectedBBox) -> None:
        """"""
        x, y, w, h = bbox.x_y_w_h
        image_roi = self.image[y: y+h, x: x+w]
    
        roi_blurred = self.apply_blur(image=image_roi)
        self.image[y: y+h, x: x+w] = roi_blurred


class BilateralFilter(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: BilateralFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.bilateralFilter(image, **self.settings)


class BoxFilter(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: BoxFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.boxFilter(image, **self.settings)


class AverageBlur(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: AverageBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.blur(image, **self.settings)


class GaussianBlur(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: GaussianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.GaussianBlur(image, **self.settings)


class MedianBlur(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: MedianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.medianBlur(image, **self.settings)


class MotionBlur(AbstractBlur):
    """"""
    def __init__(self, image: MatLike, settings: MotionBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        return cv2.filter2D(image, **self.settings)


class MosaicRectangleBlur(AbstractBlur):
    """"""
    def __init__(
        self,
        image: MatLike, 
        num_x_blocks: int,
        num_y_blocks: int,
        blur_method: Optional[BlurSetting] = None
    ):
        super().__init__(image=image)
        self.num_x_blocks = num_x_blocks
        self.num_y_blocks = num_y_blocks
        self._blur_method = self._select_blur_method(blur_method=blur_method)

    def _select_blur_method(self, blur_method: Optional[BlurSetting]) -> AbstractBlur:
        """"""
        class Pixelation(AbstractBlur):
            """"""
            def __init__(self, image: MatLike):
                super().__init__(image=image)
            
            def apply_blur(self, image: MatLike) -> MatLike:
                return np.mean(image, axis=(0, 1))
        
        blur: AbstractBlur
        match blur_method:
            case AverageBlurSettings():
                blur = AverageBlur(image=self.image, settings=blur_method)
            case MedianBlurSettings():
                blur = MedianBlur(image=self.image, settings=blur_method)
            case GaussianBlurSettings():
                blur = GaussianBlur(image=self.image, settings=blur_method)
            case None:
                blur = Pixelation(image=self.image)
            case _:
                raise ValueError(f'Unknown blur settings')
        return blur

    def apply_blur(self, image: MatLike) -> MatLike:
        """"""
        def create_block(block: int, size: int) -> Tuple[int, int]:
            return (block - 1) * size, block * size
            
        shape = image.shape[:2]
        h, w = shape

        y_size = h // self.num_y_blocks
        x_size = w // self.num_x_blocks

        blocks = []
        for y_block in range(1, self.num_y_blocks + 1):
            for x_block in range(1, self.num_x_blocks + 1):
                y_start, y_end = create_block(block=y_block, size=y_size)
                x_start, x_end = create_block(block=x_block, size=x_size)

                block = (y_start, x_start, y_end, x_end)
                blocks.append(block)

        for (y_start, x_start, y_end, x_end) in blocks:
            image[y_start: y_end, x_start: x_end] = self._blur_method.apply_blur(
                image=image[y_start: y_end, x_start: x_end]
            )

        return image
