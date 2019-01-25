import errno
import os
import shutil
import subprocess
import sys

import numpy as np
import pims

from looming_spots.db import load
from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import photodiode
from looming_spots.db.paths import RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY


def process_all_mids():  # TODO: look for recent folders first instead of looping over all
    for mouse_directory in os.listdir(RAW_DATA_DIRECTORY):
        try:
            apply_all_preprocessing_to_mouse_id(mouse_directory)
        except Exception as e:
            print(e)
            continue


def apply_all_preprocessing_to_mouse_id(mouse_id):
    raw_mouse_dir = get_raw_path(mouse_id)
    if os.path.isdir(raw_mouse_dir):
        mouse_dir = get_processed_mouse_directory(mouse_id)
        copy_mouse_directory_to_processed(mouse_id)
        apply_all_preprocessing(mouse_dir)


def copy_mouse_directory_to_processed(mouse_id):
    path_to_mouse_raw = get_raw_path(mouse_id)
    path_to_mouse_processed = get_processed_mouse_directory(mouse_id)

    if 'test' in mouse_id:
        return 'test data... skipping'

    if not os.path.isdir(path_to_mouse_processed):
        copy_directory(path_to_mouse_raw, path_to_mouse_processed)
    else:
        for fname in os.listdir(path_to_mouse_raw):
            path_to_session_raw = os.path.join(path_to_mouse_raw, fname)
            path_to_session_processed = os.path.join(path_to_mouse_processed, fname)
            if os.path.isdir(path_to_session_processed):
                print('{} has already been copied'.format(fname))
                continue
            else:
                print('copying {} to {}'.format(path_to_session_raw, path_to_session_processed))
                copy_directory(path_to_session_raw, path_to_session_processed)
    return '{} has already been copied'.format(path_to_mouse_raw)


def get_processed_mouse_directory(mouse_id):
    return os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)


def get_raw_path(mouse_id):
    return os.path.join(RAW_DATA_DIRECTORY, mouse_id)


def compare_pd_and_video(directory):
    pd_trace = photodiode.load_pd_on_clock_ups(directory)
    n_samples_pd = len(pd_trace)
    video_path = os.path.join(directory, 'camera.mp4')
    n_samples_video = len(pims.Video(video_path))

    print('pd found {} samples, there are {} frames in the video'.format(n_samples_pd, n_samples_video))

    if n_samples_pd != n_samples_video:
        n_samples_ratio = round(n_samples_pd/n_samples_video, 2)
        if n_samples_ratio.is_integer():
            print('downsampling by factor {}'.format(n_samples_ratio))
            print(n_samples_pd, n_samples_video, n_samples_ratio)
            downsampled_ai = pd_trace[::int(n_samples_ratio)]
            save_path = os.path.join(directory, 'AI_corrected')
            np.save(save_path, downsampled_ai)


def apply_all_preprocessing(mouse_id, video_name='camera'):
    raw_video_name = video_name + '.avi'
    processed_video_name = video_name + '.mp4'

    sessions = load.load_sessions(mouse_id)
    for s in sessions:
        raw_video_path = os.path.join(s.path, raw_video_name)
        processed_video_path = os.path.join(s.path, processed_video_name)
        if not os.path.isfile(processed_video_path) and os.path.isfile(raw_video_path):
            convert_to_mp4(raw_video_name, s.path, remove_avi=True)
            initialise_metadata(s.path, remove_txt=True)
        if not os.path.isfile(processed_video_path):
            raise NoProcessedVideoError

        s.extract_trials()


def convert_to_mp4(name, directory, remove_avi=False):  # TODO: remove duplication
    mp4_path = os.path.join(directory, name[:-4] + '.mp4')
    avi_path = os.path.join(directory, name)
    if os.path.isfile(mp4_path):
        print("{} already exists".format(mp4_path))
        if remove_avi:  # TEST this
            if os.path.isfile(avi_path):
                print("{} present in processed data, deleting...".format(avi_path))
                os.remove(avi_path)
    else:
        print("Creating: " + mp4_path)
        convert_avi_to_mp4(avi_path)


def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path[:-4] + '.mp4'
    print('avi: {} mp4: {}'.format(avi_path, mp4_path))

    supported_platforms = ['linux', 'windows']

    if sys.platform == 'linux':
        cmd = 'ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}'.format(avi_path, mp4_path)

    elif sys.platform == 'windows':  # TEST: on windows
        cmd = 'ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}'.format(avi_path, mp4_path).split(' ')

    else:
        raise(OSError('platform {} not recognised, expected one of {}'.format(sys.platform, supported_platforms)))

    subprocess.check_call([cmd], shell=True)


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns('*.imec*'))
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def initialise_metadata(session_folder, remove_txt=False):
    metadata_cfg_path = os.path.join(session_folder, 'metadata.cfg')
    metadata_txt_path = os.path.join(session_folder, 'metadata.txt')
    if not os.path.isfile(metadata_cfg_path):
        experiment_metadata.initialise_metadata(session_folder)
    if remove_txt:
        if os.path.isfile(metadata_txt_path):
            os.remove(metadata_txt_path)
            print('deleting: {}'.format(metadata_txt_path))


class NoProcessedVideoError(Exception):
    def __str__(self):
        print('there is no mp4 video')
