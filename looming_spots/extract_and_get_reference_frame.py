import os
import subprocess
from looming_spots.analysis import extract_looms
from configobj import ConfigObj
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
            config.write()
            subprocess.check_call('cd /home/slenzi/code/python/repositories/pyper/;'
                                  'python2 pyper/cli/tracking_cli.py {}'.format(video_path), shell=True)


def main(directory, video_fname='camera.mp4'):
    for dirName, subdirList, fileList in os.walk(directory):
        for subdir in subdirList:
            if extract_looms.is_datetime(subdir):
                target_dir = os.path.join(directory, dirName, subdir)
                if any(x == 'loom0' for x in os.listdir(target_dir)):
                    print('folder already analysed... skipping...')
                    continue
                extract_looms.create_metadata_from_experiment(target_dir)
                vid_path = os.path.join(target_dir, video_fname)
                if os.path.isfile(vid_path):
                    print(target_dir)
                    try:
                        extract_looms.auto_extract_all(target_dir)
                        extract_looms.add_ref_to_all_loom_videos(target_dir)
                        pyper_cli_track(target_dir)
                    except extract_looms.CannotFormReferenceFrameError:
                        print('there is no reference frame for this video, skipping...')
                        continue
                    except extract_looms.SteveIsntHereError:
                        'No user present to do manual reference frame selection... skipping...'
                        continue


if __name__ == '__main__':
    main('/home/slenzi/spine_shares/loomer/data_working_copy/')
