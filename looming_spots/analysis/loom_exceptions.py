
class LoomException(Exception):
    pass


class FileNotPresentError(LoomException):
    pass


class LoomVideosAlreadyExtractedError(LoomException):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def __str__(self):
        return 'the directory: {} already contains some extracted videos, ' \
               'please delete these or set overwrite to True'.format(self.directory)


class DateTimeException(LoomException):
    pass