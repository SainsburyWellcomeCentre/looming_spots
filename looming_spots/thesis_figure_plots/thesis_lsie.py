from looming_spots.db import trial_group, experimental_log
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from datetime import datetime
import seaborn as sns

from looming_spots.constants import LOOM_ONSETS
from looming_spots.util.plotting import plot_looms_ax

sns.set_style("whitegrid", {'axes.grid':False})
from looming_spots.trial_group_analysis import escape_metric_dataframes
from looming_spots.util import generic_functions

HEADBAR_REMOVED_DATE = datetime(2018, 2, 23)
flatui = ["#34495e", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71"]


def get_all_directories(mid, root=r'Z:\margrie\glusterfs\imaging\l\loomer\processed_data'):
    p = pathlib.Path(root) / mid
    directories = p.glob('*201*')
    return [str(d) for d in directories if os.path.isdir(str(d))]


def separated_by_headbar(mids):

    post_headbar = []
    pre_headbar = []
    for mid in mids:
        session_paths = get_all_directories(mid)
        for session_path in session_paths:
            session_path = pathlib.Path(session_path)
            s = session_path.stem
            try:
                date = datetime.strptime(s, "%Y%m%d_%H_%M_%S")
                if date > HEADBAR_REMOVED_DATE:
                    post_headbar.append(mid)
                else:
                    pre_headbar.append(mid)
            except Exception as e:
                print(e)
    pre_headbar = set(pre_headbar)
    post_headbar = set(post_headbar)
    return pre_headbar, post_headbar


def get_lsie_mice():
    mids=[]

    for k in ['control_habituation',
              'habituation_protocol_params_standard_ptcl',
              'long_term_context_confirmatory_A_A_0day',
              'long_term_context_confirmatory_A_A_1day',
              'long_term_context_confirmatory_A_A_3day',
              'long_term_context_confirmatory_A_A_7day',
              'long_term_context_confirmatory_A_A_14day',
              'habituation_no_pre_test_A9_0day',
              'habituation_no_pre_test_A9_3day',
              'preliminary_long_term_context_no_exploration_A_A_1day',
              'preliminary_long_term_context_no_exploration_A_A_3day',
              'preliminary_long_term_context_no_exploration_A_A_7day',
              'preliminary_long_term_context_no_exploration_A_A_8day',
              'preliminary_long_term_context_A_A_0day PICAM',
              'preliminary_long_term_context_A_A_0day',
              'context_variable_delays_A_A_0day',
               # 'pre_hab_post_24hr'
              ]:

        mouse_ids = experimental_log.get_mouse_ids_in_experiment(k)
        print(k, mouse_ids)
        mids.extend(mouse_ids)
    unprocessed=['CA39_3', 'CA39_2', 'CA39_4', 'CA40_3', 'CA39_1', 'CA40_4', 'CA40_5', 'CA41_1', 'CA51_1']
    for mid in unprocessed:
        if mid in mids:
            mids.remove(mid)

    return mids


def get_control_escapes():
    all_mids=[]
    keys = ['pre_test_control_same_day', 'pre_test_two_weeks_apart', 'auditory_white_noise_with_pre_test_-++',
            'auditory_white_noise_with_pre_test_---', 'auditory_white_noise_with_pre_test_and_delay---+-',
            'naive_escape_control', 'baseline_flee_rates_in_different_contexts', 'naive_escape_in_A',
            'molly_mouse_pre_test',
            'pre_test_effect_on_gradient_protocol', 'control_escape_adapted_arena_photometry',
            'control_3+weeks_isolated_pre_hab_post',
            'spot_contrast_cossell_curve_ct', 'pre_hab_post_24hr', 'pre_hab_post_immediate'
            ]
    for k in keys:
        mouse_ids = experimental_log.get_mouse_ids_in_experiment(k)
        print(k, mouse_ids)
        all_mids.extend(mouse_ids)
    return all_mids
    #additional_mids = ['CA50_3', 'CA40_1', 'CA41_4', 'CA435_1', 'CA435_2']


def plot_lsie_metric_distributions():
    from looming_spots.trial_group_analysis import escape_metric_dataframes
    import matplotlib.pyplot as plt
    from looming_spots.db import trial_group

    plt.close('all')

    mids = get_lsie_mice()

    pre, post = separated_by_headbar(mids)

    metrics = ['speed', 'latency peak detect']
    for metric in metrics:
        bins = metric_distribution_bins()[metric]
        mtgs_lsie = [trial_group.MouseLoomTrialGroup(mid) for mid in mids]
        df_lsie = escape_metric_dataframes.get_behaviour_metric_dataframe(mtgs_lsie, metric, 'post_test')

        mids_control = get_control_escapes()
        mtgs_control = [trial_group.MouseLoomTrialGroup(mid) for mid in mids_control]
        df_control = escape_metric_dataframes.get_behaviour_metric_dataframe(mtgs_control, metric, 'pre_test')
        plt.figure()
        sns.distplot(df_control['metric value'], bins=bins)
        sns.distplot(df_lsie['metric value'], bins=bins)


def lsie_control_escape_plots():
    from looming_spots.trial_group_analysis import escape_metric_dataframes
    import matplotlib.pyplot as plt
    from looming_spots.db import trial_group
    plt.close('all')
    mids = get_control_escapes()
    mids_lsie = get_lsie_mice()
    mids_spot_var = experimental_log.get_mouse_ids_in_experiment('spot_contrast_cossel_curve')

    metrics = ['speed', 'latency peak detect', 'time to reach shelter stimulus onset']
    mtgs_lsie = [trial_group.MouseLoomTrialGroup(mid) for mid in mids_lsie]
    df_lsie = escape_metric_dataframes.get_behaviour_metrics_dataframe(mtgs_lsie, metrics, 'post_test', 'LSIE')

    mtgs_ctrl = [trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    df_ctrl = escape_metric_dataframes.get_behaviour_metrics_dataframe(mtgs_ctrl, metrics, 'pre_test', 'CONTROL')

    #mtgs_var_spot = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids_spot_var + GROUPS['CONTROL']]
    #df_var_spot = escape_metric_dataframes.get_behaviour_metrics_dataframe(mtgs_var_spot, metrics, 'variable_contrast', 'spot_var')

    plt.figure()
    sns.scatterplot(x='time to reach shelter stimulus onset', y='speed', data=df_ctrl, s=50, marker='+',color='r')
    sns.scatterplot(x='time to reach shelter stimulus onset', y='speed', data=df_lsie,  s=50, marker='+', color='k')
    plot_looms_ax(plt.gca(), vertical=True, height=150, loom_n_samples=14, relative=True,upsample_factor=1/30)
    #sns.scatterplot(x='speed', y='time to reach shelter stimulus onset', data=df_var_spot)

    g=sns.jointplot(x='time to reach shelter stimulus onset', y='speed', data=df_ctrl, kind='kde', color='tab:blue')
    g.plot_joint(plt.scatter, c='tab:blue', linewidth=1, marker='+')
    g.ax_joint.collections[0].set_alpha(0)
    plot_looms_ax(plt.gca(), vertical=True, height=150, loom_n_samples=14, relative=True,upsample_factor=1/30)
    plt.ylim([0, 140])
    plt.xlim([0, 70])

    g=sns.jointplot(x='time to reach shelter stimulus onset', y='speed', data=df_lsie, kind='kde', color='tab:orange')
    g.plot_joint(plt.scatter, c='tab:orange', linewidth=1, marker='+')
    g.ax_joint.collections[0].set_alpha(0)
    plot_looms_ax(plt.gca(), vertical=True, height=150, loom_n_samples=14, relative=True,upsample_factor=1/30)
    plt.ylim([0, 140])
    plt.xlim([0, 70])

    for metric in metrics:
        plt.figure()
        speed_bins=np.arange(0,140,2)
        time_bins=np.arange(0,70,0.5)
        if metric =='speed':
            bins=speed_bins
        else:
            bins=time_bins
        sns.distplot(df_ctrl[metric],bins=bins)
        sns.distplot(df_lsie[metric], bins=bins)


def plot_heatmap(trials, metric='latency peak detect', stimulus_onset=200, limit=150):

    metrics = []
    speeds = []
    for t in trials:
        speeds.append(-t.smoothed_x_speed[stimulus_onset:stimulus_onset+limit])
        metric_value = t.metric_functions[metric]()
        if metric_value is None:
            metric_value = 100
        metrics.append(metric_value)
    speeds_sorted, metrics = generic_functions.sort_by(speeds, metrics)
    fig=plt.figure()
    sns.heatmap(speeds_sorted[::-1], vmin=0, vmax=0.05)
    return fig


def plot_heatmap_mids(mids, test_type):
    trials=[]
    mtgs = [trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    for mtg in mtgs:
        trials.extend(escape_metric_dataframes.get_trials(mtg, test_type))
    fig=plot_heatmap(trials)
    return fig


def control_vs_lsie_heatmap():
    mids = get_control_escapes()
    mids_lsie = get_lsie_mice()
    fig1 = plot_heatmap_mids(mids,'pre_test')
    fig2 = plot_heatmap_mids(mids_lsie, 'post_test')


def metric_distribution_bins():
    return {'speed': np.arange(0, 140, 2),
            'latency peak detect': np.arange(0, 70, 0.5)}


def which_loom_triggered_escape():
    mids = get_control_escapes()
    mtgs = [trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    loom_numbers = []
    for mtg in mtgs:
        for t in mtg.loom_trials()[:3]:
            latency = t.latency_peak_detect()
            if latency is not None:
                diffs = [x - latency for x in LOOM_ONSETS + [LOOM_ONSETS[-1] +28]]
                positive_diffs = np.array(diffs)>0
                arg = np.where(positive_diffs)[0]
                if len(arg) >0:

                    loom_numbers.append(min(arg))
    return loom_numbers


if __name__ == '__main__':
    lsie_control_escape_plots()