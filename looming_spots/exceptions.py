class LoomException(Exception):
    pass


class LoomsNotTrackedError(Exception):
    def __init__(self, msg):
        print(
            "no loom folder paths, please check you have tracked this session: {}".format(
                msg
            )
        )


class PdTooShortError(ValueError):
    pass


class MouseNotFoundError(Exception):
    pass
