from typing import List, Tuple
from dataclasses import dataclass

import pytest
from face_blur_utils import DetectedBBox

@dataclass(frozen=True)
class BBoxTestCase:
    """Used in testing the `DetectedBBox` class."""
    bbox: Tuple[int, int, int, int]
    x_y_w_h: Tuple[int, int, int, int]
    width: int
    height: int

BOUNDING_BOXES: List[BBoxTestCase] = [
    BBoxTestCase(bbox=(10, 20, 100, 150), x_y_w_h=(10, 20, 90, 130), width=90, height=130),
    BBoxTestCase(bbox=(50, 30, 200, 180), x_y_w_h=(50, 30, 150, 150), width=150, height=150),
    BBoxTestCase(bbox=(0, 0, 50, 60), x_y_w_h=(0, 0, 50, 60), width=50, height=60)
]

@pytest.mark.parametrize('bbox_case', BOUNDING_BOXES)
def test_detected_bbox(bbox_case: BBoxTestCase) -> None:
    """Test instantiation and properties of `DetectedBBox` class."""
    detected_bbox = DetectedBBox(*bbox_case.bbox)
    assert detected_bbox.width == bbox_case.width
    assert detected_bbox.height == bbox_case.height
    assert detected_bbox.x_y_w_h == bbox_case.x_y_w_h


@pytest.mark.parametrize('bbox_case', BOUNDING_BOXES)
def test_detected_bbox_from_x_y_w_h(bbox_case: BBoxTestCase) -> None:
    """Test class method and properties of `DetectedBBox` class."""
    detected_bbox = DetectedBBox.from_x_y_w_h(*bbox_case.x_y_w_h)
    assert detected_bbox.width == bbox_case.width
    assert detected_bbox.height == bbox_case.height

    assert (
        detected_bbox.left, detected_bbox.top, detected_bbox.right, detected_bbox.bottom
    ) == bbox_case.bbox