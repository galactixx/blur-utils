from __future__ import annotations

import cv2
from typing import Any, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

class AverageBlurSettings(BaseModel):
    """"""
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    anchor: Tuple[int, int] = Field(default=(-1, -1), serialization_alias='anchor')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class GaussianBlurSettings(BaseModel):
    """"""
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    sigma_x: float = Field(default=0, serialization_alias='simgaX')
    sigma_y: float = Field(default=0, serialization_alias='sigmaY')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class MedianBlurSettings(BaseModel):
    """"""
    kernel: int = Field(..., serialization_alias='ksize')


class BilateralFilterSettings(BaseModel):
    """"""
    diameter: int = Field(..., serialization_alias='d')
    sigma_color: float = Field(..., serialization_alias='sigmaColor')
    sigma_space: float = Field(..., serialization_alias='sigmaSpace')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class BoxFilterSettings(BaseModel):
    """"""
    kernel: Tuple[int, int] = Field(..., serialization_alias='ksize')
    anchor: Tuple[int, int] = Field(default=(-1, -1), serialization_alias='anchor')
    depth: int = Field(default=-1, serialization_alias='ddepth')
    normalize: bool = Field(default=True, serialization_alias='ddepth')
    border_type: int = Field(
        default=cv2.BORDER_DEFAULT, serialization_alias='borderType'
    )


class MotionBlurSettings(BaseModel):
    """"""
    kernel: NDArray[Any] = Field(..., serialization_alias='kernel')
    depth: int = Field(default=-1, serialization_alias='ddepth')

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_motion_direction(
        cls, direction: Literal['vertical', 'horizontal'], n: int
    ) -> MotionBlurSettings:
        """"""
        kernel: NDArray[Any]
        if direction == 'horizontal':
            kernel = np.array([1 for _ in range(n)]) / n
        elif direction == 'vertical':
            kernel = np.array(n*[[1]]) / n
        else:
            raise ValueError('Unknown motion direction')
        
        return cls(kernel=kernel)