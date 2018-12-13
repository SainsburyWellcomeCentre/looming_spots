from matplotlib import pyplot as plt

from looming_spots.db import session_group, load
from looming_spots.db.loom_trial_group import MouseLoomTrialGroup
from looming_spots.preprocess import convert_videos
from looming_spots.analysis.plotting import plot_looms
from looming_spots.db import experimental_log


def main():
    experimental_group = ''
    mouse_ids_in_experiment = experimental_log.get_mouse_ids_in_experiment(experimental_group)

    print(mouse_ids_in_experiment)

    extract_and_track_all(mouse_ids_in_experiment)

    quick_plot(mouse_ids_in_experiment, label=experimental_group)


def extract_and_track_all(mouse_ids):

    for mid in mouse_ids:
        convert_videos.apply_all_preprocessing_to_mouse_id(mid)

    for mid in mouse_ids:
        sg = load.load_sessions(mid)
        for s in sg:
            for t in s.trials:
                t.make_reference_frames()

    for mid in mouse_ids:
        sg = load.load_sessions(mid)
        for s in sg:
            for t in s.trials:
                t.extract_track(overwrite=False)


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


def quick_plot(mouse_ids, label=None):

    for i, mid in enumerate(mouse_ids):
        print(mid)
        fig, axes = plt.subplots(2, 1)
        plot_title_pre = '{}_{}_{}'.format(label, mid, 'pre_habituation_protocol')
        plot_title_post = '{}_{}_{}'.format(label, mid, 'post_habituation_protocol')

        mtg = MouseLoomTrialGroup(mid)

        plt.sca(axes[0])
        plt.title(plot_title_pre)
        for t in mtg.get_trials_of_type('pre_test'):
            t.plot_track()
        plt.ylim([-0.15, 1.0])

        plt.sca(axes[1])
        plt.title(plot_title_post)
        for t in mtg.get_trials_of_type('post_test'):
            t.plot_track()
        plt.ylim([-0.15, 1.0])

        plot_looms(fig)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

