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

from face_blur_utils._typing import BlurSetting, DetectedBBox
from face_blur_utils._settings import (
    AverageBlurSettings,
    BilateralFilterSettings,
    BoxFilterSettings,
    GaussianBlurSettings,
    MedianBlurSettings,
    MotionBlurSettings
)

class AbstractBlur(ABC):
    """
    A  simple abstract class for a variety of facial blur methods. 
    
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
        return self._settings.model_dump(by_alias=True)

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
        self.image[y: y+h, x: x+w] = roi_blurred


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
        self._blur_method = self._select_blur_method(blur_method=blur_method)

    def _select_blur_method(self, blur_method: Optional[BlurSetting]) -> AbstractBlur:
        """A private method to select the blur method to use within each tesserae."""
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
        """
        Applies a mosaic blur with rectangular tesserae to an image.
        
        Args:
            image (`MatLike`): An ndarray representation of the image on which
                to apply the mosaic blur.

        Returns:
            `MatLike`: The ndarray representation of the image with the mosaic
                blur applied.
        """
        def create_block(block: int, size: int) -> Tuple[int, int]:
            return (block - 1) * size, block * size

        # Retrieve image dimensions
        shape = image.shape[:2]
        h, w = shape

        # Calculate the height and width of a tessera
        y_size = h // self.num_y_tesserae
        x_size = w // self.num_x_tesserae

        # Generate all tesserae sections (start and end coordinates)
        blocks = []
        for y_block in range(1, self.num_y_tesserae + 1):
            for x_block in range(1, self.num_x_tesserae + 1):

                # Calcuate the start and end coordinates in both the
                # y and x directions
                y_start, y_end = create_block(block=y_block, size=y_size)
                x_start, x_end = create_block(block=x_block, size=x_size)

                block = (y_start, x_start, y_end, x_end)
                blocks.append(block)

        # Iterate through each pre-generated tesserae and apply the blur
        for (y_start, x_start, y_end, x_end) in blocks:
            image[y_start: y_end, x_start: x_end] = self._blur_method.apply_blur(
                image=image[y_start: y_end, x_start: x_end]
            )

        return image
