import os

import pytest
from face_blur_utils import (
    load_image, 
    load_video
)

EXAMPLES_PATH = './tests/examples/'

# Valid photo file names
ELON_MUSK_PHOTO = 'elon.jpg'
MARK_Z_PHOTO = 'marky_z.jpg'

# Invalid photo file names
NOT_A_PHOTO = 'not_a_photo.jpg'

# Valid video file name
STOCK_VIDEO = 'video.mp4'

# Invalid video file names
NOT_A_VIDEO = 'not_a_video.mp4'

@pytest.mark.parametrize(
    'image_file_name', [ELON_MUSK_PHOTO, MARK_Z_PHOTO]
)
def test_load_image(image_file_name: str) -> None:
    """"""
    image_file_path = os.path.join(EXAMPLES_PATH, image_file_name)
    _ = load_image(image_file=image_file_path)


def test_load_invalid_image() -> None:
    """"""
    image_file_path = os.path.join(EXAMPLES_PATH, NOT_A_PHOTO)

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = load_image(image_file=image_file_path)

    assert str(exc_info.value)


def test_load_video() -> None:
    """"""
    video_file_path = os.path.join(EXAMPLES_PATH, STOCK_VIDEO)
    _ = load_video(video_file=video_file_path)


def test_load_invalid_video() -> None:
    """"""
    video_file_path = os.path.join(EXAMPLES_PATH, NOT_A_VIDEO)

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = load_video(video_file=video_file_path)

    assert str(exc_info.value)