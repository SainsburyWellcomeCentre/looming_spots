import os

from matplotlib import pyplot as plt
import numpy as np

from looming_spots.db import session_group
from looming_spots.preprocess import convert_videos
from looming_spots.reference_frames import session_getter
from looming_spots.analysis.plotting import plot_looms, plot_line_with_color_variable
from looming_spots.db import experimental_log
from looming_spots.db.loom_trial_group import LoomTrialGroup


def main():

    experiment_labels = ['social_housing_age_litter_matched_isolation_control', 'social_housed_in_20',
                'social_house_single_mouse', 'social_house_single_mouse_control']

    for i,exp_label in enumerate(experiment_labels):
        #exp_label = 'social_house_single_mouse_control'
        #control_label = 'social_house_single_mouse_control'

        log_df = experimental_log.load_df()
        mouse_ids_in_experiment = experimental_log.get_mouse_ids_in_experiment(log_df, exp_label)
        mouse_ids_in_experiment = [x.replace('.', '_') for x in mouse_ids_in_experiment]


        #quick_plot(mouse_ids_in_experiment, ('pre',), exp_label, save_dir='/home/slenzi/loomer/looming_analysis/figures/social_house_single_mouse_IVC_litter_match_controls/')
        #plot_all(mouse_ids_in_experiment, exp_label, save_dir='/home/slenzi/loomer/looming_analysis/figures/social_summary/')
        #plot_all_as_heatmap(mouse_ids_in_experiment, exp_label)
        all_trials = get_trial_group(mouse_ids_in_experiment)
        ltg = LoomTrialGroup(all_trials, exp_label)
        ltg.plot_latencies(i)
        # fig = ltg.plot_hm(ltg.latencies())
        # save_dir = '/home/slenzi/loomer/looming_analysis/figures/social_summary/'
        # fmt = '.png'
        # title_str = '{}_{}{}'.format(exp_label, 'heatmap', fmt)
        # save_path = os.path.join(save_dir, title_str)
        # fig.savefig(save_path, format=fmt[1:])
    plt.xticks([0, 1, 2, 3], experiment_labels)
    plt.ylabel('latency (s)')
    plt.show()


def get_trial_group(mids):
    all_trials = []
    for i, mid in enumerate(mids):

        sg = session_group.MouseSessionGroup(mid)
        [print(s.path) for s in sg.sessions]

        for t in sg.pre_trials[:3]:
            all_trials.append(t)
    return all_trials


def plot_all(mids, exp_label, save_dir=None, format='png'):

    fig = plt.figure(figsize=(10, 3))

    n_trials = 0
    n_flees = 0
    for i, mid in enumerate(mids):

        sg = session_group.MouseSessionGroup(mid)
        [print(s.path) for s in sg.sessions]

        for t in sg.pre_trials[:3]:
            n_trials += 1
            t.plot_track()
            if t.is_flee():
                n_flees += 1

    title = '{}: {} flees out of {} trials, n={} mice'.format(exp_label, n_flees, n_trials,len(mids))
    plt.title(title)

    plt.ylim([-0.15, 1.0])
    plot_looms(fig)
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, exp_label + '.' + format)
        fig.savefig(save_path, format=format)
    else:
        plt.show()


def plot_all_as_heatmap(mids, exp_label, save_dir=None, format='png'):
    fig = plt.figure(figsize=(7, 5))
    all_speeds = []
    n_trials = 0
    n_flees = 0

    for i, mid in enumerate(mids):

        sg = session_group.MouseSessionGroup(mid)
        [print(s.path) for s in sg.sessions]

        for t in sg.pre_trials[:3]:
            n_trials += 1
            all_speeds.append(t.normalised_x_speed)
            if t.is_flee():
                n_flees += 1

    plt.imshow(all_speeds, vmax=0.1, vmin=-0.1, aspect='auto', cmap='coolwarm_r')
    title = '{}: {} flees out of {} trials, n={} mice'.format(exp_label, n_flees, n_trials, len(mids))
    plt.title(title)
    plt.colorbar()
    if save_dir:
        save_path = os.path.join(save_dir, exp_label + '_heatmap.' + format)
        fig.savefig(save_path, format=format)
    else:
        plt.show()


def quick_plot_as_group(mouse_ids_exp, other_mouse_ids_control, label_exp, label_control):
    fig, axes = plt.subplots(2, 1)

    for i, (grp, label) in enumerate(zip([mouse_ids_exp, other_mouse_ids_control], [label_exp, label_control])):
        plt.sca(axes[i])
        plt.title(label)

        for i, mid in enumerate(grp):
            sg = session_group.MouseSessionGroup(mid)
            [print(s.path) for s in sg.sessions]

            for t in sg.pre_trials:
                t.plot_track()

    plt.ylim([-0.15, 1.0])
    plot_looms(fig)
    plt.tight_layout()
    plt.show()


def get_tests(session_group, key, limit=3):
    if key == 'pre':
        return session_group.pre_trials[:limit]
    elif key == 'post':
        return session_group.post_trials[:limit]


def quick_plot(mouse_ids, test_types, label=None, save_dir=None):
    for i, mid in enumerate(mouse_ids):
        fig, axes = plt.subplots(len(test_types), 1, figsize=(10, 3*len(test_types)))

        sg = session_group.MouseSessionGroup(mid)
        [print(s.path) for s in sg.sessions]

        for j, test_type in enumerate(test_types):
            plot_title = '{}_{}_{}'.format(label, mid, test_type)
            if not isinstance(axes, list):
                plt.sca(axes)
            else:
                plt.sca(axes[j])

            plt.title(plot_title)

            for t in get_tests(sg, test_type, limit=3):
                t.plot_track()
                #plot_track_with_x_speed_on_image(0.09, t, title=None)
            plt.ylim([-0.15, 1.0])

        plot_looms(fig)
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, '{}_{}.png'.format(label, mid))
            fig.savefig(save_path)

        else:
            plt.show()


def plot_track_with_x_speed_on_image(max_speed, t, title=None):

    plt.title(title)
    plt.imshow(t.get_reference_frame()[0:275, :], cmap='Greys_r')
    x, y = t.raw_track
    plot_line_with_color_variable(x, y, np.abs(t.normalised_x_speed), start=200, normalising_factor=max_speed)
    t.plot_mouse_location_at_stimulus_onset()
    plt.axis('off')

if __name__ == '__main__':
    main()


