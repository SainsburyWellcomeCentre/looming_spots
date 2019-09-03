import os
from datetime import datetime
import warnings
from pathlib import Path
from shutil import copyfile

import looming_spots.preprocess.extract_videos
import looming_spots.util.generic_functions
from looming_spots.db import session_io
from looming_spots.db.constants import PROCESSED_DATA_DIRECTORY


class NotExtractedError(Exception):
    pass


class MouseNotFoundError(Exception):
    pass


def load_sessions(mouse_id):
    mouse_directory = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    if os.path.isdir(mouse_directory):

        for s in os.listdir(mouse_directory):

            session_directory = os.path.join(mouse_directory, s)
            if not os.path.isdir(session_directory):
                continue

            file_names = os.listdir(session_directory)

            if not contains_analog_input(file_names):
                continue

            if 'contrasts.mat' not in s:
                print('no contrasts mat')

                if not os.path.isdir(session_directory):
                    continue

                if not looming_spots.util.generic_functions.is_datetime(s):
                    print('not datetime, skipping')
                    continue

                if not contains_video(file_names) and not contains_tracks(file_names):
                    print('no video or tracks')
                    if not get_tracks_from_raw(mouse_directory.replace('processed_data', 'raw_data')):
                        continue

            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s = session_io.Session(dt=date, mouse_id=mouse_id)
            session_list.append(s)

        if len(session_list) == 0:
            raise MouseNotFoundError('the mouse: {} has not been processed'.format(mouse_id))

        return sorted(session_list)
    warnings.warn('the mouse: {} has not been copied to the processed data directory'.format(mouse_id))
    raise MouseNotFoundError('the mouse: {} has not been processed'.format(mouse_id))


def contains_analog_input(file_names):
    if 'AI.bin' in file_names or 'AI.tdms' in file_names:
        return True
    return False


def contains_video(file_names):
    return any('.avi' in fname for fname in file_names) or any('.mp4' in fname for fname in file_names)


def contains_tracks(file_names):
    return any('dlc_x_tracks.npy' in fname for fname in file_names)


def get_tracks_from_raw(directory):
    print('getting tracks from {}'.format(directory))
    p = Path(directory)
    track_paths = p.rglob('*tracks.npy')
    if len(list(p.rglob('*tracks.npy'))) == 0:
        print('no track paths found...')
        return False

    for tp in track_paths:
        raw_path = str(tp)
        processed_path = raw_path.replace('raw_data', 'processed_data')
        print('copying {} to {}'.format(raw_path, processed_path))
        copyfile(raw_path, processed_path)
    return True

