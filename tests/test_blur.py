from dataclasses import dataclass
import os
from typing import List, TypeAlias

import numpy as np
import pytest
from blur_utils import (
    AverageBlur,
    AverageBlurSettings,
    DetectedBBox,
    GaussianBlur,
    GaussianBlurSettings,
    load_image,
    MosaicRectBlur
)

from blur_utils._blur import AbstractBlur

@dataclass
class BlurTestCase:
    """Used in testing a variety of blur features."""
    input_image: str
    output_image: str

    def _build_path(self, image_name: str) -> str:
        return os.path.join(EXAMPLES_PATH, image_name)

    @property
    def input_path(self) -> str:
        return self._build_path(image_name=self.input_image)

    @property
    def output_path(self) -> str:
        return self._build_path(image_name=self.output_image)

TestCases: TypeAlias = List[BlurTestCase]

EXAMPLES_PATH = './tests/examples/'

# Photo file names
ELON_MUSK_PHOTO = 'elon.png'
MARK_Z_PHOTO = 'marky_z.png'

# Mosaic blur photo file names
ELON_MUSK_MOSAIC_PHOTO = 'elon_mosaic.png'
MARK_Z_MOSAIC_PHOTO = 'marky_z_mosaic.png'

# Gaussian blur photo file names
ELON_MUSK_GAUSSIAN_PHOTO = 'elon_gaussian.png'
MARK_Z_GAUSSIAN_PHOTO = 'marky_z_gaussian.png'

# Average blur photo file names
ELON_MUSK_AVG_PHOTO = 'elon_average.png'
MARK_Z_AVG_PHOTO = 'marky_z_average.png'

MOSAIC_CASES: TestCases = [
    BlurTestCase(input_image=ELON_MUSK_PHOTO, output_image=ELON_MUSK_MOSAIC_PHOTO),
    BlurTestCase(input_image=MARK_Z_PHOTO, output_image=MARK_Z_MOSAIC_PHOTO)
]

GAUSSIAN_CASES: TestCases = [
    BlurTestCase(input_image=ELON_MUSK_PHOTO, output_image=ELON_MUSK_GAUSSIAN_PHOTO),
    BlurTestCase(input_image=MARK_Z_PHOTO, output_image=MARK_Z_GAUSSIAN_PHOTO)
]

AVERAGE_CASES: TestCases = [
    BlurTestCase(input_image=ELON_MUSK_PHOTO, output_image=ELON_MUSK_AVG_PHOTO),
    BlurTestCase(input_image=MARK_Z_PHOTO, output_image=MARK_Z_AVG_PHOTO)
]

def __apply_blur_and_compare(blur: AbstractBlur, case: BlurTestCase) -> None:
    """
    Private function to run the core of blur tests, and ensure that the
    blur operation is performing as expected.
    """
    # Get dimensions of the image
    shape = blur.image.shape[:2]
    h, w = shape

    # Apply the blur functionality to the entire image
    image_bbox = DetectedBBox.from_x_y_w_h(x=0, y=0, w=w, h=h)
    blur.apply_blur_to_face(bbox=image_bbox)

    output_image = load_image(image_file=case.output_path)

    assert np.array_equal(output_image, blur.image)


@pytest.mark.parametrize('mosaic_case', MOSAIC_CASES)
def test_mosaic_blur(mosaic_case: BlurTestCase) -> None:
    """Test that the mosaic blur is performing as expected."""
    input_image = load_image(image_file=mosaic_case.input_path)

    mosaic_blur = MosaicRectBlur(
        image=input_image, num_x_tesserae=15, num_y_tesserae=15
    )
    __apply_blur_and_compare(blur=mosaic_blur, case=mosaic_case)


@pytest.mark.parametrize('gaussian_case', GAUSSIAN_CASES)
def test_gaussian_blur(gaussian_case: BlurTestCase) -> None:
    """Test that the gaussian blur is performing as expected."""
    input_image = load_image(image_file=gaussian_case.input_path)

    gaussian_blur = GaussianBlur(
        image=input_image, settings=GaussianBlurSettings(kernel=(5, 5))
    )
    __apply_blur_and_compare(blur=gaussian_blur, case=gaussian_case)


@pytest.mark.parametrize('average_case', AVERAGE_CASES)
def test_average_blur(average_case: BlurTestCase) -> None:
    """Test that the average blur is performing as expected."""
    input_image = load_image(image_file=average_case.input_path)

    average_blur = AverageBlur(
        image=input_image, settings=AverageBlurSettings(kernel=(5, 5))
    )
    __apply_blur_and_compare(blur=average_blur, case=average_case)