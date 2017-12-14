import os
from datetime import datetime

import numpy as np
import skvideo
import skvideo.io

from looming_spots.db.experiment_metadata import load_metadata, save_key_to_metadata
from looming_spots.preprocess.photodiode import get_manual_looms, get_loom_idx_from_raw, get_fpath

METADATA_PATH = './metadata.cfg'


def auto_extract_all(directory, overwrite=False):
    if any('loom' in fname for fname in os.listdir(directory)) and not overwrite:
        print('looms already present')
        return 'looms already extracted.. skipping'
    all_loom_idx = get_loom_idx_from_raw(directory)
    manual_loom_indices = get_manual_looms(all_loom_idx)
    if manual_loom_indices is not None:
        config = load_metadata(directory)
        save_key_to_metadata(config, 'manual_loom_idx', list(manual_loom_indices))
        extract_loom_videos(directory, manual_loom_indices)
    else:
        print('no loom indices')
    print('done')


def extract_loom_videos(directory, manual_loom_indices):
    for loom_number, loom_idx in enumerate(manual_loom_indices):
        extract_loom_video(directory, loom_idx, loom_number)


def extract_loom_video(directory, loom_start, loom_number, n_samples_before=200, n_samples_after=200, shape=(480, 640)):
    loom_video_path = os.path.join(directory, 'loom{}.h264'.format(loom_number))
    if os.path.isfile(loom_video_path):
        return
    video_path = get_fpath(directory, '.mp4')
    rdr = skvideo.io.vreader(video_path)
    loom_video = np.zeros((n_samples_before+n_samples_after+1, shape[0], shape[1], 3))
    a = 0
    for i, frame in enumerate(rdr):
        if (loom_start-n_samples_before) < i < (loom_start + n_samples_after):
            loom_video[a, :, :, :] = frame
            a += 1
    skvideo.io.vwrite(loom_video_path, loom_video)


def check_reference_is_first_frame(video, reference_image):
    return video[0, :, :] == reference_image


def is_datetime(folder_name):
    try:
        date_time = datetime.strptime(folder_name, '%Y%m%d_%H_%M_%S')
        print('string is in date_time format: {}'.format(date_time))
        return True
    except ValueError:  # FIXME: custom exception required
        return False
