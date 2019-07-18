import os
from shutil import copyfile
from pathlib import Path

import nptdms
from matplotlib import pyplot as plt

from looming_spots.db import session_group, load, experimental_log
from looming_spots.db.loom_trial_group import MouseLoomTrialGroup
from looming_spots.preprocess import convert_videos
from looming_spots.analysis.plotting import plot_looms
from looming_spots.preprocess.convert_videos import copy_mouse_directory_to_processed
from looming_spots.db.constants import get_raw_path, RAW_DATA_DIRECTORY
from looming_spots.preprocess.photodiode import LoomNumberError


def copy_to_processed(mid):
    pass


def main():
    #experimental_group = 'spot_contrast_cossel_curve'
    #nmda_lesion_group = ['CA439_1', 'CA439_4', '276585A', '276585B']
    #nmda_control_group = ['CA452_1', 'CA439_5', '276585C']

    mouse_ids_in_experiment = ['CA467_1', 'CA467_3', 'CA467_4', 'CA467_5']  #['CA457_1', 'CA457_2', 'CA457_3'] #nmda_control_group #['276585B']  # ['CA436_2', 'CA436_3', 'CA436_4']  # ['CA403_1', 'CA403_2', 'CA366_5', 'BY184_1', 'CA428_1']
    #mouse_ids_in_experiment = experimental_log.get_mouse_ids_in_experiment(experimental_group)
    # print(mouse_ids_in_experiment)
    for mid in mouse_ids_in_experiment:
        print(mid)
        copy_video_to_winstor_tmp(mid)
        copy_mouse_directory_to_processed(mid)


    #extract_and_track_all(mouse_ids_in_experiment)
    #quick_plot(mouse_ids_in_experiment, label=experimental_group)


def extract_and_track_all(mouse_ids):

    for mid in mouse_ids:
        raw_mouse_dir = get_raw_path(mid)
        print(raw_mouse_dir)
        if os.path.isdir(raw_mouse_dir):
            print('attempting conversion... wololo')
            try:
                convert_videos.apply_all_preprocessing_to_mouse_id(mid)
            except LoomNumberError as e:
                continue

    # for mid in mouse_ids:
    #     raw_mouse_dir = get_raw_path(mid)
    #     if os.path.isdir(raw_mouse_dir):
    #         sg = load.load_sessions(mid)
    #         for s in sg:
    #             for t in s.trials:
    #                 t.make_reference_frames()

    # for mid in mouse_ids:
    #     raw_mouse_dir = get_raw_path(mid)
    #     if os.path.isdir(raw_mouse_dir):
    #         sg = load.load_sessions(mid)
    #         for s in sg:
    #             for t in s.trials:
    #                 t.extract_track(overwrite=False)


def quick_plot_as_group(mouse_ids_group, other_mouse_ids_group, labels):
    fig, axes = plt.subplots(2, 1)

    for i, (grp, label) in enumerate(zip([mouse_ids_group, other_mouse_ids_group], labels)):
        plt.sca(axes[i])
        plt.title(label)

        for i, mid in enumerate(grp):
            sg = session_group.MouseSessionGroup(mid)
            [print(s.path) for s in sg.sessions]

            for t in sg.pre_trials:
                t.plot_track()


def process_mouse_dlc(mouse_id):
    pass


def copy_video_to_winstor_tmp(mouse_id):

    raw_data_mouse_directory = Path(get_raw_path(mouse_id))
    tmp_dir = Path('/home/slenzi/winstor/margrie/slenzi/tmp/')

    videos = raw_data_mouse_directory.rglob('*.avi')
    for v in videos:

        relative_path = v.relative_to(RAW_DATA_DIRECTORY)
        new_path = tmp_dir.joinpath(relative_path)
        if not os.path.exists(str(new_path)):
            os.makedirs(str(new_path.parent))
            copyfile(str(v), str(new_path))
            make_finished_file(str(new_path.parent))


def make_finished_file(directory_path):
    directory_path = os.path.join(directory_path, 'finished')
    os.mkdir(directory_path)


def quick_plot(mouse_ids, label=None):

    for i, mid in enumerate(mouse_ids):
        raw_mouse_dir = get_raw_path(mid)
        if os.path.isdir(raw_mouse_dir):
            print(mid)
            fig, axes = plt.subplots(2, 1)
            plot_title_pre = '{}_{}_{}'.format(label, mid, 'pre_habituation_protocol')
            plot_title_post = '{}_{}_{}'.format(label, mid, 'post_habituation_protocol')

            mtg = MouseLoomTrialGroup(mid)

            plt.sca(axes[0])
            plt.title(plot_title_pre)
            for t in mtg.all_trials:#mtg.get_trials_of_type('pre_test'):
                t.plot_track()
            plt.ylim([-0.15, 1.0])

            plt.sca(axes[1])
            plt.title(plot_title_post)
            for t in mtg.get_trials_of_type('post_test', limit=6):
                t.plot_track()
            plt.ylim([-0.15, 1.0])

        plot_looms(fig)
        plt.tight_layout()
    plt.show()


def get_all_tracks(tmp_directory='/home/slenzi/winstor/margrie/slenzi/tmp/'):
    p = Path(tmp_directory)
    track_paths = p.rglob('*.npy')
    for track_path in track_paths:
        old_path = str(track_path)
        mouse_id = old_path.split('/')[-3]
        date = old_path.split('/')[-2]
        fname = old_path.split('/')[-1]
        new_path = '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/{}/{}/{}'.format(
            mouse_id, date, fname)

        # os.makedirs(os.path.split(new_path)[0])
        if not os.path.isfile(new_path) and ('CA435_2' not in new_path):
            print(old_path)
            print(new_path)
            copyfile(old_path, new_path)


if __name__ == '__main__':
    main()

