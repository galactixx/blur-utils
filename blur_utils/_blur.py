from __future__ import annotations

from abc import ABC, abstractmethod
import cv2
from typing import (
    Any,
    Dict,
    Optional,
    overload
)

import numpy as np
from pydantic import BaseModel

from blur_utils._exceptions import InvalidSettingsError
from blur_utils._typing import BlurSetting, DetectedBBox
from blur_utils._settings import (
    AverageBlurSettings,
    BilateralFilterSettings,
    BoxFilterSettings,
    GaussianBlurSettings,
    MedianBlurSettings,
    MotionBlurSettings
)

@overload
def get_blur(image: np.ndarray, settings: AverageBlurSettings) -> AverageBlur:
    ...


@overload
def get_blur(image: np.ndarray, settings: BilateralFilterSettings) -> BilateralFilter:
    ...


@overload
def get_blur(image: np.ndarray, settings: BoxFilterSettings) -> BoxFilter:
    ...


@overload
def get_blur(image: np.ndarray, settings: GaussianBlurSettings) -> GaussianBlur:
    ...


@overload
def get_blur(image: np.ndarray, settings: MedianBlurSettings) -> MedianBlur:
    ...


@overload
def get_blur(image: np.ndarray, settings: MotionBlurSettings) -> MotionBlur:
    ...


def get_blur(image: np.ndarray, settings: BlurSetting) -> AbstractBlur:
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
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`BaseModel` | None): A pydantic `BaseModel` representing the settings
            for the specific blur method, can be None if implementing `MosaicRectBlur`.
    """
    def __init__(self, image: np.ndarray, settings: Optional[BaseModel] = None):
        self.image = image
        self._settings = settings

    @abstractmethod
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
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
        image (`np.ndarray`): A `np.ndarray` instance, an ndarray representation of the image
        settings (`BilateralFilterSettings`): The settings for the bilateral filter blur.
    """
    def __init__(self, image: np.ndarray, settings: BilateralFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the bilateral filter blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the bilateral filter blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the bilateral
                filter blur applied.
        """
        return cv2.bilateralFilter(image, **self.settings)


class BoxFilter(AbstractBlur):
    """
    Implementation of the box filter blur in OpenCV.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`BoxFilterSettings`): The settings for the box filter blur.
    """
    def __init__(self, image: np.ndarray, settings: BoxFilterSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the box filter blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the box filter blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the box
                filter blur applied.
        """
        return cv2.boxFilter(image, **self.settings)


class AverageBlur(AbstractBlur):
    """
    Implementation of the average blur in OpenCV.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`AverageBlurSettings`): The settings for the average blur.
    """
    def __init__(self, image: np.ndarray, settings: AverageBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the average blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the average blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the average
                blur applied.
        """
        return cv2.blur(image, **self.settings)


class GaussianBlur(AbstractBlur):
    """
    Implementation of the gaussian blur in OpenCV.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`GaussianBlurSettings`): The settings for the gaussian blur.
    """
    def __init__(self, image: np.ndarray, settings: GaussianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the gaussian blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the gaussian blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the gaussian
                blur applied.
        """
        return cv2.GaussianBlur(image, **self.settings)


class MedianBlur(AbstractBlur):
    """
    Implementation of the median blur in OpenCV.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`MedianBlurSettings`): The settings for the median blur.
    """
    def __init__(self, image: np.ndarray, settings: MedianBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the median blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the median blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the median
                blur applied.
        """
        return cv2.medianBlur(image, **self.settings)


class MotionBlur(AbstractBlur):
    """
    Implementation of a simple horizontal or vertical motion blur.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        settings (`MotionBlurSettings`): The settings for the motion blur.
    """
    def __init__(self, image: np.ndarray, settings: MotionBlurSettings):
        super().__init__(image=image, settings=settings)

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a horizontal or vertical motion blur to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the motion blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the motion
                blur applied.
        """
        return cv2.filter2D(image, **self.settings)


class MosaicRectBlur(AbstractBlur):
    """
    Implementation of a mosaic blur where the tesserae are rectangles.
    
    Args:
        image (`np.ndarray`): An ndarray representation of the image.
        num_x_tesserae (int): The number of tesserae on the x-axis.
        num_y_tesserae (int): The number of tesserae on the y-axis.
    """
    def __init__(
        self, image: np.ndarray, num_x_tesserae: int, num_y_tesserae: int
    ):
        super().__init__(image=image)
        self.num_x_tesserae = num_x_tesserae
        self.num_y_tesserae = num_y_tesserae

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a mosaic blur with rectangular tesserae to an image.
        
        Args:
            image (`np.ndarray`): An ndarray representation of the image on which
                to apply the mosaic blur.

        Returns:
            `np.ndarray`: The ndarray representation of the image with the mosaic
                blur applied.
        """
        # Retrieve image dimensions
        shape = image.shape[:2]
        h, w = shape

        # Resize the image down to the tessera grid size
        small_image = cv2.resize(
            image,
            (self.num_x_tesserae, self.num_y_tesserae),
            interpolation=cv2.INTER_AREA
        )

        # Resize the small image back to the original size
        mosaic_image = cv2.resize(
            small_image,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        return mosaic_image


BLUR_MAPPING = {
    AverageBlurSettings: AverageBlur,
    BilateralFilterSettings: BilateralFilter,
    BoxFilterSettings: BoxFilter,
    GaussianBlurSettings: GaussianBlur,
    MedianBlurSettings: MedianBlur,
    MotionBlurSettings: MotionBlur
}