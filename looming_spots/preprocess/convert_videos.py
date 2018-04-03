import errno
import os
import shutil
import subprocess

import numpy as np

from looming_spots.db.metadata import experiment_metadata
from looming_spots.preprocess import extract_looms
from looming_spots.preprocess import photodiode
from looming_spots.db.paths import RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY


def process_all_data():  # TODO: look for recent folders first instead of looping over all
    for mouse_id in os.listdir(RAW_DATA_DIRECTORY):
        try:
            mouse_dir = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
            copy_mouse_directory_to_processed(mouse_id)
            apply_all_preprocessing(mouse_dir)
        except Exception as e:
            print(e)
            continue


def get_frame_number(video_path):
    from ffprobe3 import FFProbe
    metadata = FFProbe(video_path)
    for stream in metadata.streams:
        if stream.is_video():
                return stream.frames()


def compare_pd_and_video(directory):
    pd_trace = photodiode.load_pd_on_clock_ups(directory)
    n_samples_pd = len(pd_trace)
    video_path = os.path.join(directory, 'camera.mp4')
    n_samples_video = get_frame_number(video_path)
    print('pd found {} samples, there are {} frames in the video'.format(n_samples_pd, n_samples_video))
    if n_samples_pd != n_samples_video:
        n_samples_ratio = round(n_samples_pd/n_samples_video, 2)
        if n_samples_ratio.is_integer():
            print('downsampling by factor {}'.format(n_samples_ratio))
            print(n_samples_pd, n_samples_video, n_samples_ratio)
            downsampled_ai = pd_trace[::int(n_samples_ratio)]
            save_path = os.path.join(directory, 'AI_corrected')
            np.save(save_path, downsampled_ai)


def get_processed_path(mouse_id):
    return os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)


def get_raw_path(mouse_id):
    return os.path.join(RAW_DATA_DIRECTORY, mouse_id)


def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path[:-4] + '.mp4'
    print('avi: {} mp4: {}'.format(avi_path, mp4_path))
    #subprocess.check_call(['ffmpeg -i {} -c:v libx264 -preset fast -crf 18 {}'.format(avi_path, mp4_path)], shell=True)
    subprocess.check_call(['ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}'.format(avi_path,
                                                                                             mp4_path)], shell=True)
    os.remove(avi_path)


def copy_mouse_directory_to_processed(mouse_id):
    path_to_mouse_raw = get_raw_path(mouse_id)
    path_to_mouse_processed = get_processed_path(mouse_id)
    if 'test' in mouse_id:
        return 'test data... skipping'
    if 'probe.txt' in path_to_mouse_raw:
        return 'this is a probe experiment... skipping all sessions'
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


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def apply_all_preprocessing(mouse_dir):
    for session_folder, dirs, files in os.walk(mouse_dir, topdown=False):
        print(session_folder)
        for name in files:
            if ".avi" in name:
                convert_to_mp4(name, session_folder, remove_avi=True)
                initialise_metadata(session_folder, remove_txt=True)
                extract_looms.auto_extract_all_looms(session_folder)
            if ".mp4" in name:
                extract_looms.auto_extract_all_looms(session_folder)


def apply_all_preprocessing_to_mouse_id(mouse_id):
    mouse_dir = get_processed_path(mouse_id)
    copy_mouse_directory_to_processed(mouse_id)
    apply_all_preprocessing(mouse_dir)


def initialise_metadata(session_folder, remove_txt=False):
    metadata_cfg_path = os.path.join(session_folder, 'metadata.cfg')
    metadata_txt_path = os.path.join(session_folder, 'metadata.txt')
    if not os.path.isfile(metadata_cfg_path):
        experiment_metadata.initialise_metadata(session_folder)
    if remove_txt:
        if os.path.isfile(metadata_txt_path):
            os.remove(metadata_txt_path)
            print('deleting: {}'.format(metadata_txt_path))


def convert_to_mp4(name, directory, remove_avi=False):  # TODO: remove duplication
    mp4_path = os.path.join(directory, name[:-4] + '.mp4')
    avi_path = os.path.join(directory, name)
    if os.path.isfile(mp4_path):
        print("{} already exists".format(mp4_path))
        if remove_avi:
            if os.path.isfile(avi_path):
                print("{} present in processed data, deleting...".format(avi_path))
                os.remove(avi_path)
    else:
        print("Creating: " + mp4_path)
        convert_avi_to_mp4(avi_path)
