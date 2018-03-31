import os

import numpy as np
import skvideo
import skvideo.io

from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import photodiode

METADATA_PATH = './metadata.cfg'
VIDEO_SHAPE = (480, 640)


def auto_extract_all_looms(session_folder, overwrite=False, extract_habituation_looms=False, n_habituation_looms=24):

    if any('loom' in fname for fname in os.listdir(session_folder)) and not overwrite:
        print('looms already present in {}'.format(session_folder))
        return 'looms already extracted.. skipping'

    all_loom_idx = experiment_metadata.get_loom_idx(session_folder)
    manual_loom_indices = photodiode.get_manual_looms(all_loom_idx)

    if extract_habituation_looms:
        looms_idx_to_extract = all_loom_idx[::5]
        habituation_loom_idx = looms_idx_to_extract[:n_habituation_looms]
        save_dir = os.path.join(session_folder, 'habituation')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        extract_loom_videos(session_folder, save_dir, habituation_loom_idx)

    if manual_loom_indices is not None:
        config = experiment_metadata.load_metadata(session_folder)
        experiment_metadata.save_key_to_metadata(config, 'manual_loom_idx', list(manual_loom_indices))
        extract_loom_videos(session_folder, session_folder, manual_loom_indices)


def extract_loom_videos(directory_in, directory_out, extraction_idx):
    for loom_number, loom_idx in enumerate(extraction_idx):
        extract_loom_video(directory_in, directory_out, loom_idx, loom_number)


def extract_loom_video(directory_in, directory_out, loom_start, loom_number, n_samples_before=200, n_samples_after=400):
    loom_start = int(loom_start)
    loom_video_path = os.path.join(directory_out, 'loom{}.h264'.format(loom_number))
    if os.path.isfile(loom_video_path):
        return
    video_path = photodiode.get_fpath(directory_in, '.mp4')
    extract_video(video_path, loom_video_path, loom_start-n_samples_before, loom_start + n_samples_after)


def extract_video(fin_path, fout_path, start, end, shape=VIDEO_SHAPE):
    rdr = skvideo.io.vreader(fin_path)
    video = np.zeros((end - start + 1, shape[0], shape[1], 3))
    a = 0
    for i, frame in enumerate(rdr):
        if start < i < end + 1:
            video[a, :, :, :] = frame
            a += 1
    skvideo.io.vwrite(fout_path, video)


def upsample_video(path):
    vid = skvideo.io.vread(path)
    new_video = []
    for i, frame in enumerate(vid):
        new_video.append(frame)
        new_video.append(frame)
    return np.array(new_video)

