from __future__ import annotations

import math
from abc import ABC, abstractmethod
import cv2
from cv2.typing import MatLike
from typing import (
    Any,
    Dict,
    Optional,
    overload,
    Set,
    Tuple
)

import numpy as np
from pydantic import BaseModel

from face_blur_utils._exceptions import InvalidSettingsError
from face_blur_utils._typing import BlurSetting, DetectedBBox
from face_blur_utils._settings import (
    AverageBlurSettings,
    BilateralFilterSettings,
    BoxFilterSettings,
    GaussianBlurSettings,
    MedianBlurSettings,
    MotionBlurSettings
)

_STEP_SIZE_PCT = 0.15

@overload
def get_blur(image: MatLike, settings: AverageBlurSettings) -> AverageBlur:
    ...


@overload
def get_blur(image: MatLike, settings: BilateralFilterSettings) -> BilateralFilter:
    ...


@overload
def get_blur(image: MatLike, settings: BoxFilterSettings) -> BoxFilter:
    ...


@overload
def get_blur(image: MatLike, settings: GaussianBlurSettings) -> GaussianBlur:
    ...


@overload
def get_blur(image: MatLike, settings: MedianBlurSettings) -> MedianBlur:
    ...


@overload
def get_blur(image: MatLike, settings: MotionBlurSettings) -> MotionBlur:
    ...


def get_blur(image: MatLike, settings: BlurSetting) -> AbstractBlur:
    """"""
    settings_type = type(settings)
    blur = BLUR_MAPPING.get(settings_type, None)

    if blur is None:
        raise ValueError(f'Unknown blur settings')
    
    return blur(image=image, settings=settings)


class AbstractBlur(ABC):
    """
    A simple abstract class for a variety of facial blur methods. 
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`BaseModel` | None): A pydantic `BaseModel` representing the settings
            for the specific blur method, can be None if implementing `MosaicRectBlur`.
    """
    def __init__(self, image: MatLike, settings: Optional[BaseModel] = None):
        self.image = image
        self._settings = settings

    @abstractmethod
    def apply_blur(self, image: MatLike) -> MatLike:
        """Abstract method for applying a blur directly to an entire image."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def settings(self) -> Dict[str, Any]:
        """Returns the current settings being used for each blur."""
        if self._settings is None:
            raise InvalidSettingsError('Settings for blur has not been loaded')
        
        return self._settings.model_dump(by_alias=True)

    @settings.setter
    def settings(self, settings: BaseModel) -> None:
        """Setter for the blur settings."""
        self._settings = settings

    def apply_blur_to_face(self, bbox: DetectedBBox) -> None:
        """
        Applies a blur directly to the bounding box of a face represented by
        a `DetectedBBox` instance. 

        Using the image attribute of the instance and the `DetectedBBox`, which
        represents a bounding box within the aforementioned image, will blur
        the area of bounding box based on the type of blur class.

        Args:
            bbox (`DetectedBBox`): A `DetectedBBox` instance representing a bounding
                box highlighting a face detected in an image.
        """
        x, y, w, h = bbox.x_y_w_h
        image_roi = self.image[y: y+h, x: x+w]
    
        roi_blurred = self.apply_blur(image=image_roi)
        image_roi[:, :] = roi_blurred


class BilateralFilter(AbstractBlur):
    """
    Implementation of the bilateral filter blur in OpenCV.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image
        settings (`BilateralFilterSettings`): The settings for the bilateral filter blur.
    """
    def __init__(self, image: MatLike, settings: BilateralFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies the bilateral filter blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the bilateral filter blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the bilateral
                filter blur applied.
        """
        return cv2.bilateralFilter(image, **self.settings)


class BoxFilter(AbstractBlur):
    """
    Implementation of the box filter blur in OpenCV.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`BoxFilterSettings`): The settings for the box filter blur.
    """
    def __init__(self, image: MatLike, settings: BoxFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies the box filter blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the box filter blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the box
                filter blur applied.
        """
        return cv2.boxFilter(image, **self.settings)


class AverageBlur(AbstractBlur):
    """
    Implementation of the average blur in OpenCV.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`AverageBlurSettings`): The settings for the average blur.
    """
    def __init__(self, image: MatLike, settings: AverageBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies the average blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the average blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the average
                blur applied.
        """
        return cv2.blur(image, **self.settings)


class GaussianBlur(AbstractBlur):
    """
    Implementation of the gaussian blur in OpenCV.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`GaussianBlurSettings`): The settings for the gaussian blur.
    """
    def __init__(self, image: MatLike, settings: GaussianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies the gaussian blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the gaussian blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the gaussian
                blur applied.
        """
        return cv2.GaussianBlur(image, **self.settings)


class MedianBlur(AbstractBlur):
    """
    Implementation of the median blur in OpenCV.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`MedianBlurSettings`): The settings for the median blur.
    """
    def __init__(self, image: MatLike, settings: MedianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies the median blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the median blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the median
                blur applied.
        """
        return cv2.medianBlur(image, **self.settings)


class MotionBlur(AbstractBlur):
    """
    Implementation of a simple horizontal or vertical motion blur.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        settings (`MotionBlurSettings`): The settings for the motion blur.
    """
    def __init__(self, image: MatLike, settings: MotionBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies a horizontal or vertical motion blur to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the motion blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the motion
                blur applied.
        """
        return cv2.filter2D(image, **self.settings)


class MosaicRectBlur(AbstractBlur):
    """
    Implementation of a mosaic blur where the tesserae are rectangles.
    
    Args:
        image (`MatLike`): A `MatLike` instance, an ndarray representation of the image.
        num_x_tesserae (int): The number of tesserae on the x-axis.
        num_y_tesserae (int): The number of tesserae on the y-axis.
        image (`BlurSetting` | None): The settings for the type of blur to apply to each
            tesserae. Can select from an average blur, gaussian blur, or median blur. If
            None is selected, then defaults to a pixelation blur. Defaults to None.
    """
    def __init__(
        self,
        image: MatLike, 
        num_x_tesserae: int,
        num_y_tesserae: int,
        blur_method: Optional[BlurSetting] = None
    ):
        super().__init__(image=image)
        self.num_x_tesserae = num_x_tesserae
        self.num_y_tesserae = num_y_tesserae

        # Select blur method to apply to each tessera
        self._blur_method = self._select_blur_method(blur_method=blur_method)

    def _select_blur_method(self, blur_method: Optional[BlurSetting]) -> AbstractBlur:
        """A private method to select the blur method to use within tessera."""
        class Pixelation(AbstractBlur):
            """"""
            def __init__(self, image: MatLike):
                super().__init__(image=image)
            
            def apply_blur(self, image: MatLike) -> MatLike:
                return np.mean(image, axis=(0, 1))
        
        # Based on the blur settings passed, determine which blur to select. The blur
        # will be applied to each tessera.
        
        # If no blur settings was specified, then the default is to use a pixelation
        # blur, which simply takes the average of the R, G, B channels
        blur: AbstractBlur
        if blur_method is None:
            blur = Pixelation(image=self.image)
        else:
            blur = get_blur(image=self.image, settings=blur_method)

        return blur

    def apply_blur(self, image: MatLike) -> MatLike:
        """
        Applies a mosaic blur with rectangular tesserae to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the mosaic blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the mosaic
                blur applied.
        """
        class TesseraeInfo:
            def __init__(self, num: int, dim: int):
                self.num = num
                self.size = dim // num
                self.blocks: Dict[int, Tuple[int, int]] = dict()

                self._position = 0
                self._overflow_indices: Set[int] = set()

                size_overflow = dim - num * self.size
                tessera_indices = set(range(self.num))

                # Determine the step-size
                step_size = self._determine_step_size()

                num_chosen = 0
                while num_chosen != size_overflow:
                    # Apply modulo step-size for evenly distributed allocation
                    index = (num_chosen * step_size) % self.num

                    # Adjust the collections of tessera indices
                    self._overflow_indices.add(index)
                    tessera_indices.remove(index)

                    num_chosen += 1
            
            @property
            def position(self) -> int:
                return self._position

            @property
            def overflow_indices(self) -> Set[int]:
                return self._overflow_indices
            
            def _determine_step_size(self) -> int:
                step_gcd = -1
                step_size = round(_STEP_SIZE_PCT * self.num) - 1
                while step_gcd != 1:
                    step_size += 1
                    step_gcd = math.gcd(self.num, step_size)

                return step_size

            def create_block(self, block_index: int) -> Tuple[int, int]:
                if block_index in self.blocks:
                    return self.blocks[block_index]
                else:
                    start = self._position
                    self._position += self.size

                    if block_index in self.overflow_indices:
                        self._position += 1

                    block = (start, self.position)
                    self.blocks.update({block_index: block})

                    return block

        # Retrieve image dimensions
        shape = image.shape[:2]
        h, w = shape

        # Calculate tessera information for x and y
        y_tesserae = TesseraeInfo(num=self.num_y_tesserae, dim=h)
        x_tesserae = TesseraeInfo(num=self.num_x_tesserae, dim=w)

        # Generate all tesserae sections and apply blur
        for y_block in range(y_tesserae.num):
            y_start, y_end = y_tesserae.create_block(block_index=y_block)

            for x_block in range(x_tesserae.num):

                # Calcuate the start and end coordinates in the x direction
                x_start, x_end = x_tesserae.create_block(block_index=x_block)

                # Iterate through each pre-computed tesserae and apply the blur
                tessera_view = image[y_start: y_end, x_start: x_end]
                tessera_view[:, :] = self._blur_method.apply_blur(image=tessera_view)

        return image


BLUR_MAPPING = {
    AverageBlurSettings: AverageBlur,
    BilateralFilterSettings: BilateralFilter,
    BoxFilterSettings: BoxFilter,
    GaussianBlurSettings: GaussianBlur,
    MedianBlurSettings: MedianBlur,
    MotionBlurSettings: MotionBlur
}