import os
import subprocess
import errno
import shutil
from looming_spots.db import experiment_metadata
from looming_spots.scripts.extract_and_get_reference_frame import pyper_cli_track
from looming_spots.preprocess import extract_looms
from looming_spots.ref_builder import reference_frames

HEAD_DIRECTORY = "/home/slenzi/spine_shares/loomer/data_working_copy/"
RAW_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, 'test')
PROCESSED_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, 'test1')


def main():
    for mouse_id in os.listdir(RAW_DATA_DIRECTORY):
        if mouse_id in os.listdir(PROCESSED_DATA_DIRECTORY):
            continue
        copy_mouse_directory_to_processed(mouse_id)
        mouse_dir = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
        apply_all_preprocessing(mouse_dir)


def get_processed_path(mouse_id):
    return os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)


def get_raw_path(mouse_id):
    return os.path.join(RAW_DATA_DIRECTORY, mouse_id)


def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path[:-4] + '.mp4'
    print('avi: {} mp4: {}'.format(avi_path, mp4_path))
    subprocess.check_call(['ffmpeg -i {} -c:v libx264 -preset fast -crf 18 {}'.format(avi_path, mp4_path)], shell=True)
    os.remove(avi_path)


def copy_mouse_directory_to_processed(mouse_id):
    path_to_mouse_raw = get_raw_path(mouse_id)
    path_to_mouse_processed = get_processed_path(mouse_id)
    if os.path.isdir(path_to_mouse_processed):
        print('{} has already been copied'.format(path_to_mouse_raw))
        return

    copy_directory(path_to_mouse_raw, path_to_mouse_processed)


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
    for root, dirs, files in os.walk(mouse_dir, topdown=False):
        for name in files:

            if ".avi" in name:
                mp4_path = os.path.join(root, name[:-4] + '.mp4')

                if os.path.isfile(mp4_path):
                    print("Already exists: " + mp4_path)
                    print(root)
                else:
                    print("Creating: " + mp4_path)
                    avi_path = os.path.join(root, name)
                    convert_avi_to_mp4(avi_path)

                if not os.path.isfile(os.path.join(root, 'metadata.cfg')):
                    experiment_metadata.initialise_metadata(root)

                extract_looms.auto_extract_all(root)
                reference_frames.add_ref_to_all_loom_videos(root)
                pyper_cli_track(root)
