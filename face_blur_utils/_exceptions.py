class BaseError(Exception):
    """Base error class."""
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return self.message


class VideoCaptureError(BaseError):
    """"""
    def __init__(self, message: str):
        super().__init__(message=message)