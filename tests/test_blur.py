from dataclasses import dataclass
import os
from typing import List

import numpy as np
import pytest
from face_blur_utils import (
    DetectedBBox,
    load_image,
    MosaicRectBlur
)

@dataclass
class BlurTestCase:
    """"""
    input_image: str
    output_image: str

    def _build_path(self, image_name: str) -> str:
        return os.path.join(EXAMPLES_PATH, image_name)

    @property
    def input_path(self) -> str:
        """"""
        return self._build_path(image_name=self.input_image)

    @property
    def output_path(self) -> str:
        """"""
        return self._build_path(image_name=self.output_image)

EXAMPLES_PATH = './tests/examples/'

# Photo file names
ELON_MUSK_PHOTO = 'elon.png'
MARK_Z_PHOTO = 'marky_z.png'

# Post-mosaic blur photo file names
ELON_MUSK_MOSAIC_PHOTO = 'elon_mosaic.png'
MARK_Z_MOSAIC_PHOTO = 'marky_z_mosaic.png'

MOSAIC_CASES: List[BlurTestCase] = [
    BlurTestCase(input_image=ELON_MUSK_PHOTO, output_image=ELON_MUSK_MOSAIC_PHOTO),
    BlurTestCase(input_image=MARK_Z_PHOTO, output_image=MARK_Z_MOSAIC_PHOTO)
]

@pytest.mark.parametrize('mosaic_case', MOSAIC_CASES)
def test_mosaic_blur(mosaic_case: BlurTestCase) -> None:
    """"""
    input_image = load_image(image_file=mosaic_case.input_path)

    mosaic_blur = MosaicRectBlur(
        image=input_image, num_x_tesserae=15, num_y_tesserae=15
    )

    # Get dimensions of the image
    shape = input_image.shape[:2]
    h, w = shape

    # Apply the blur functionality to the entire image
    image_bbox = DetectedBBox.from_x_y_w_h(x=0, y=0, w=w, h=h)
    mosaic_blur.apply_blur_to_face(bbox=image_bbox)

    output_image = load_image(image_file=mosaic_case.output_path)

    assert np.array_equal(output_image, mosaic_blur.image)
