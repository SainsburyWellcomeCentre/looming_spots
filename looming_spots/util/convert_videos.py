import os
import subprocess
import sys
import warnings

import numpy as np
import pims

import looming_spots.preprocess.io
from looming_spots.db import load
from looming_spots.db.constants import get_processed_mouse_directory, get_raw_path, RAW_DATA_DIRECTORY
from looming_spots.db.metadata import experiment_metadata
from looming_spots.deprecated.deprecated import copy_mouse_directory_to_processed


def process_all_mids():
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
        #apply_all_preprocessing(mouse_dir)


class NoPdError(Exception):
    pass


def compare_pd_and_video(directory):
    pd_trace = looming_spots.preprocess.io.load_pd_on_clock_ups(directory)
    n_samples_pd = len(pd_trace)
    video_path = os.path.join(directory, 'camera.mp4')
    n_samples_video = len(pims.Video(video_path))

    print('pd found {} samples, there are {} frames in the video'.format(n_samples_pd, n_samples_video))

    if n_samples_pd != n_samples_video:
        if n_samples_pd == 0:
            raise NoPdError
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
        print(s.path)
        raw_video_path = os.path.join(s.path, raw_video_name)
        processed_video_path = os.path.join(s.path, processed_video_name)
        if not os.path.isfile(processed_video_path) and os.path.isfile(raw_video_path):
            try:
                convert_to_mp4(raw_video_name, s.path, remove_avi=True)
                initialise_metadata(s.path, remove_txt=True)
            except FileNotFoundError as e:
                print(e)
                continue
        if not os.path.isfile(processed_video_path):

            warnings.warn('no video file in this session')
            #NoProcessedVideoError
            continue

        #s.extract_trials()
        #s.track_trials()


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
