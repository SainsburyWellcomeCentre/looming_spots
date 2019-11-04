class LoomException(Exception):
    pass


class FileNotPresentError(LoomException):
    pass


class LoomVideosAlreadyExtractedError(LoomException):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def __str__(self):
        return (
            "the directory: {} already contains some extracted videos, "
            "please delete these or set overwrite to True".format(
                self.directory
            )
        )


class DateTimeException(LoomException):
    pass


class SteveIsntHereError(LoomException):
    pass


class CannotFormReferenceFrameError(LoomException):
    pass


class LoomsNotTrackedError(Exception):
    def __init__(self, msg):
        print(
            "no loom folder paths, please check you have tracked this session: {}".format(
                msg
            )
        )


class NoReferenceFrameError(Exception):
    pass


class PdTooShortError(ValueError):
    pass


class LoomNumberError(Exception):
    pass


class NotExtractedError(Exception):
    pass


class MouseNotFoundError(Exception):
    pass
