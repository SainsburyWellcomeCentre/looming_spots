import os
import subprocess

from configobj import ConfigObj

import looming_spots.db.experiment_metadata
import looming_spots.exceptions
import looming_spots.exceptions
import looming_spots.preprocess.extract_looms
import looming_spots.preprocess.photodiode
import looming_spots.ref_builder.reference_frames
from looming_spots.ref_builder import viewer
from looming_spots.preprocess import extract_looms
from pyper.config import conf

METADATA_PATH = './metadata.cfg'


def pyper_cli_track(directory):
    for fname in os.listdir(directory):
        if 'loom' in fname and '.h264' in fname:
            video_path = os.path.join(directory, fname)
            print(video_path)
            metadata = ConfigObj(os.path.join(directory, METADATA_PATH), encoding="UTF8", indent_type='    ',
                                 unrepr=False, create_empty=True, write_empty_values=True)
            config = conf.config

            if metadata['context'] == 'A':
                config['tracker']['roi']['restriction']['rectangle']['top_left'] = (0, 0)  # TODO: rm hard code
            elif metadata['context'] == 'B':
                config['tracker']['roi']['restriction']['rectangle']['top_left'] = (200, 0)
            print(config['tracker']['detection']['threshold'])
            config.write()
            print(config['tracker']['detection']['threshold'])
            subprocess.check_call('cd /home/slenzi/code/python/repositories/pyper/;'
                                  'python2 pyper/cli/tracking_cli.py --overwrite {}'.format(video_path), shell=True)


def main(directory, video_fname='camera.mp4'):
    for dirName, subdirList, fileList in os.walk(directory):
        for subdir in subdirList:
            if looming_spots.preprocess.extract_looms.is_datetime(subdir):
                target_dir = os.path.join(directory, dirName, subdir)
                if any(x == 'loom0' for x in os.listdir(target_dir)):
                    print('folder already analysed... skipping...')
                    continue
                looming_spots.db.experiment_metadata.create_metadata_from_experiment(target_dir)
                vid_path = os.path.join(target_dir, video_fname)
                if os.path.isfile(vid_path):
                    print(target_dir)
                    try:
                        extract_looms.auto_extract_all(target_dir)
                        looming_spots.ref_builder.reference_frames.add_ref_to_all_loom_videos(target_dir)
                        pyper_cli_track(target_dir)
                    except looming_spots.exceptions.CannotFormReferenceFrameError:
                        print('there is no reference frame for this video, skipping...')
                        continue
                    except looming_spots.exceptions.SteveIsntHereError:
                        'No user present to do manual reference frame selection... skipping...'
                        continue


def process_experiments(directory, video_fname='camera.mp4'):
    for dirName, subdirList, fileList in os.walk(directory):
        for subdir in subdirList:

            if looming_spots.preprocess.extract_looms.is_datetime(subdir):
                mouse_id = os.path.split(dirName)[-1]
                if 'CA105' in mouse_id:
                    print(mouse_id)

                    target_dir = os.path.join(directory, dirName, subdir)

                    vid_path = os.path.join(target_dir, video_fname)
                    if os.path.isfile(vid_path):
                        looming_spots.ref_builder.reference_frames.add_ref_to_all_loom_videos(target_dir)
                        pyper_cli_track(target_dir)


def rerun_all(directory, video_fname='camera.mp4'):
    #looming_spots.db.experiment_metadata.create_metadata_from_experiment(directory)
    vid_path = os.path.join(directory, video_fname)
    if os.path.isfile(vid_path):
        print(directory)
        #extract_looms.auto_extract_all(directory, overwrite=True)
        # try:
        #     looming_spots.ref_builder.reference_frames.add_ref_to_all_loom_videos(directory)
        # except looming_spots.exceptions.LoomException:
        #     viewer.Viewer(directory)
        pyper_cli_track(directory)


if __name__ == '__main__':
    main('/home/slenzi/spine_shares/loomer/data_working_copy/')
