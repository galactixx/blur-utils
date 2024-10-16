from __future__ import annotations

import cv2
from typing import Any, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

class AverageBlurSettings(BaseModel):
    """
    Configuration for average blur in OpenCV.

    Attributes:
        kernel (Tuple[int, int]): Kernel size for the blur operation.
        anchor (Tuple[int, int]): Anchor point of the kernel.
        border_type (int): Specifies the border handling.
    """
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    anchor: Tuple[int, int] = Field(default=(-1, -1), serialization_alias='anchor')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class GaussianBlurSettings(BaseModel):
    """
    Configuration for Gaussian blur in OpenCV.

    Attributes:
        kernel (Tuple[int, int]): Kernel size for the blur operation.
        sigma_x (float): Standard deviation in the X direction.
        sigma_y (float): Standard deviation in the Y direction.
        border_type (int): Specifies the border handling.
    """
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    sigma_x: float = Field(default=0, serialization_alias='sigmaX')
    sigma_y: float = Field(default=0, serialization_alias='sigmaY')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class MedianBlurSettings(BaseModel):
    """
    Configuration for median blur in OpenCV.

    Attributes:
        kernel (int): Kernel size for the blur operation.
    """
    kernel: int = Field(..., serialization_alias='ksize')


class BilateralFilterSettings(BaseModel):
    """
    Configuration for bilateral filter in OpenCV.

    Attributes:
        diameter (int): Diameter of each pixel neighborhood.
        sigma_color (float): Filter sigma in the color space.
        sigma_space (float): Filter sigma in the coordinate space.
        border_type (int): Specifies the border handling.
    """
    diameter: int = Field(..., serialization_alias='d')
    sigma_color: float = Field(..., serialization_alias='sigmaColor')
    sigma_space: float = Field(..., serialization_alias='sigmaSpace')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class BoxFilterSettings(BaseModel):
    """
    Configuration for box filter in OpenCV.

    Attributes:
        kernel (Tuple[int, int]): Kernel size for the filter.
        anchor (Tuple[int, int]): Anchor point of the kernel.
        depth (int): Desired depth of the output image.
        normalize (bool): Flag to normalize the filter.
        border_type (int): Specifies the border handling.
    """
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    anchor: Tuple[int, int] = Field(default=(-1, -1), serialization_alias='anchor')
    depth: int = Field(default=-1, serialization_alias='ddepth')
    normalize: bool = Field(default=True, serialization_alias='ddepth')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class MotionBlurSettings(BaseModel):
    """
    Configuration for motion blur in OpenCV.

    Attributes:
        kernel (NDArray[Any]): Kernel to be applied for the motion blur.
        depth (int): Desired depth of the output image.
    """
    kernel: NDArray[Any] = Field(..., serialization_alias='kernel')
    depth: int = Field(default=-1, serialization_alias='ddepth')

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_motion_direction(
        cls, direction: Literal['vertical', 'horizontal'], n: int
    ) -> MotionBlurSettings:
        """
        Create a `MotionBlurSettings` instance with a motion blur kernel
        based on the direction.

        Args:
            direction (Literal['vertical', 'horizontal']): Direction of the motion blur.
            n (int): Size of the kernel.

        Returns:
            `MotionBlurSettings`: A settings object with the generated kernel.
        """
        kernel: NDArray[Any]
        if direction == 'horizontal':
            kernel = np.ones((1, n)) / n
        elif direction == 'vertical':
            kernel = np.ones((n, 1)) / n
        else:
            raise ValueError('Unknown motion direction')
        
        return cls(kernel=kernel)