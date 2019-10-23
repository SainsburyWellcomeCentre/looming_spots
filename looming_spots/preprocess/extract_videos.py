import os

import numpy as np
import pims
import skvideo
import skvideo.io

import looming_spots.util.generic_functions


def extract_loom_videos(directory_in, directory_out, extraction_idx):
    for loom_number, loom_idx in enumerate(extraction_idx):
        extract_loom_video(directory_in, directory_out, loom_idx, loom_number)


def extract_loom_video(directory_in, directory_out, loom_start, loom_number, n_samples_before=200, n_samples_after=400):
    loom_start = int(loom_start)
    loom_video_path = os.path.join(directory_out, 'loom{}.h264'.format(loom_number))
    if os.path.isfile(loom_video_path):
        return
    video_path = looming_spots.util.generic_functions.get_fpath(directory_in, '.mp4')
    print(video_path)
    extract_video(video_path, loom_video_path, loom_start-n_samples_before, loom_start + n_samples_after)


def extract_loom_video_trial(path_in, path_out, loom_start, n_samples_before=200, n_samples_after=400, overwrite=False):
    if not os.path.isfile(path_in):
        path_in = path_in.replace('.mp4', '.avi')
        path_in = path_in.replace('processed', 'raw')
    loom_start = int(loom_start)
    if not overwrite:
        if os.path.isfile(path_out):
            return
    print(path_in)
    extract_video(path_in, path_out, loom_start-n_samples_before, loom_start + n_samples_after)


def extract_video(fin_path, fout_path, start, end):
    v = pims.Video(fin_path)
    out_video = v[start:end]
    skvideo.io.vwrite(fout_path, out_video)


def upsample_video(path):
    vid = skvideo.io.vread(path)
    new_video = []
    for i, frame in enumerate(vid):
        new_video.append(frame)
        new_video.append(frame)
    return np.array(new_video)

