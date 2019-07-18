import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session_io

from looming_spots.db.constants import PROCESSED_DATA_DIRECTORY
import warnings


class NotExtractedError(Exception):
    pass


def load_sessions(mouse_id):
    mouse_directory = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    if os.path.isdir(mouse_directory):
        for s in os.listdir(mouse_directory):
            if 'contrasts.mat' not in s:
                print('no contrasts mat')
                session_directory = os.path.join(mouse_directory, s)
                if not os.path.isdir(session_directory):
                    continue
                file_names = os.listdir(session_directory)
                if not looming_spots.util.generic_functions.is_datetime(s):
                    print('not datetime, skipping')
                    continue
                if not any('.avi' in fname for fname in file_names) and not any('.mp4' in fname for fname in file_names):
                    if not any('dlc_x_tracks.npy' in fname for fname in file_names):
                        print('no video or tracks')
                        continue
                date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
                s = session_io.Session(dt=date, mouse_id=mouse_id)
                session_list.append(s)
        return sorted(session_list)
    warnings.warn('the mouse: {} has not been processed'.format(mouse_id))
