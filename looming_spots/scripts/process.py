from matplotlib import pyplot as plt

from looming_spots.db import session_group
from looming_spots.preprocess import convert_videos
from looming_spots.reference_frames import session_getter
from looming_spots.analysis.plotting import plot_looms
from looming_spots.db import experimental_log


def main():
    log_df = experimental_log.load_df()
    mouse_ids_in_experiment = experimental_log.get_mouse_ids_in_experiment(log_df, 'DREADDs_V2')
    mouse_ids_in_experiment = [x.replace('.', '_') for x in mouse_ids_in_experiment]

    #failed_experiments_df = log_df[log_df['exclude']]
    # # failed_experiments_mids = experimental_log.get_mouse_ids(failed_experiments_df)
    # # failed_experiments_mids = [x.replace('.', '_') for x in failed_experiments_mids]
    #
    # for mid in failed_experiments_mids:
    #     if mid in mouse_ids_in_experiment:
    #         mouse_ids_in_experiment.remove(mid)
    #mouse_ids_in_experiment = ['CA188_2']
    #extract_and_track_all(['CA301_2'])
    extract_and_track_all(mouse_ids_in_experiment)
    #quick_plot(mouse_ids_in_experiment)
    quick_plot(mouse_ids_in_experiment)


def extract_and_track_all(mouse_ids):

    for mid in mouse_ids:
        convert_videos.apply_all_preprocessing_to_mouse_id(mid)

    for mid in mouse_ids:
        sg = session_getter.load_sessions(mid)
        for s in sg:
            for t in s.trials:
                t.track_trial(overwrite=False)


def quick_plot_as_group(mouse_ids_group, other_mouse_ids_group, labels):
    fig, axes = plt.subplots(2, 1)

    for i, (grp, label) in enumerate(zip([mouse_ids_group, other_mouse_ids_group]), labels):
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

        #sessions = session_getter.load_sessions(mid)
        sg = session_group.MouseSessionGroup(mid)
        [print(s.path) for s in sg.sessions]
        plt.sca(axes[0])
        plt.title(plot_title_pre)
        for t in sg.pre_trials:
            t.plot_track()
        plt.ylim([-0.15, 1.0])
        plt.sca(axes[1])
        plt.title(plot_title_post)
        for t in sg.post_trials:
            t.plot_track()
        plt.ylim([-0.15, 1.0])

        # for j, s in enumerate(sg):
        #
        #     #plt.subplot(4, 3, (i*(j+1)) + j + 1)
        #     plt.sca(axes[i][j])
        #     plt.title(plot_title)
        #     s.plot_trials()
        plot_looms(fig)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


