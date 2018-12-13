import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session_io

from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY
import warnings


class NotExtractedError(Exception):
    pass


def load_sessions(mouse_id):
    mouse_directory = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    if os.path.isdir(mouse_directory):
        for s in os.listdir(mouse_directory):
            if not looming_spots.util.generic_functions.is_datetime(s):
                continue
            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s = session_io.Session(dt=date, mouse_id=mouse_id)
            session_list.append(s)
        return sorted(session_list)
    warnings.warn('the mouse: {} has not been processed'.format(mouse_id))
