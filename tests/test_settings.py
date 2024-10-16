from dataclasses import dataclass
from typing import (
    Any,
    List,
    Literal
)

import numpy as np
from numpy.typing import NDArray
import pytest
from blur_utils import MotionBlurSettings

@dataclass
class MotionBlurTestCase:
    """Used in testing the `MotionBlurSettings` class."""
    direction: Literal['vertical', 'horizontal']
    n: int
    kernel: NDArray[Any]
    depth: int = -1

MOTION_BLUR_CASES: List[MotionBlurTestCase] = [
    MotionBlurTestCase(direction='horizontal', n=4, kernel=np.array([[0.25, 0.25, 0.25, 0.25]])),
    MotionBlurTestCase(direction='horizontal', n=5, kernel=np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])),
    MotionBlurTestCase(direction='vertical', n=4, kernel=np.array([[0.25], [0.25], [0.25], [0.25]])),
    MotionBlurTestCase(direction='vertical', n=5, kernel=np.array([[0.20], [0.20], [0.20], [0.20], [0.20]])),
]

@pytest.mark.parametrize('motion_case', MOTION_BLUR_CASES)
def test_motion_blur_default(motion_case: MotionBlurTestCase) -> None:
    """Test class method from `MotionBlurSettings` class."""
    motion_settings = MotionBlurSettings.from_motion_direction(
        direction=motion_case.direction, n=motion_case.n
    )
    assert np.array_equal(motion_settings.kernel, motion_case.kernel)
    assert motion_settings.depth == motion_case.depth