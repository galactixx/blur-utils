class BaseError(Exception):
    """Base error class."""
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.message


class VideoCaptureError(BaseError):
    """Error occurs when loading a video file into a cv2.VideoCapture."""
    def __init__(self, message: str):
        super().__init__(message=message)


class ImageReadError(BaseError):
    """Error occurs when loading an image file into a MatLike instance."""
    def __init__(self, message: str):
        super().__init__(message=message)