import warnings
import os

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd

from looming_spots.util import plotting
from looming_spots.db import loom_trial_group

from looming_spots.db.experimental_log import get_mouse_ids_in_experiment


def get_flee_probabilities_from_trials(trials):
    all_mouse_ids = set([t.mouse_id for t in trials])
    flee_probabilities = []
    n_flees_all = []
    n_non_flees_all = []

    for id in all_mouse_ids:
        mouse_trial_results = [t.is_flee() for t in trials if t.mouse_id == id]
        n_flees = np.count_nonzero(mouse_trial_results)
        n_non_flees = len(mouse_trial_results) - n_flees
        flee_probability = n_flees / len(mouse_trial_results)
        flee_probabilities.append(flee_probability)
        n_flees_all.append(n_flees)
        n_non_flees_all.append(n_non_flees)
    return n_flees_all, n_non_flees_all, flee_probabilities


def plot_flee_probabilities(
    experimental_labels, trial_type="pre_test", ax=None
):
    df = pd.DataFrame()
    for experimental_group_label in experimental_labels:
        df_dict = {}
        trials = load_trials_from_label(experimental_group_label, trial_type)
        n_flees, n_non_flees, flee_probabilities = get_flee_probabilities_from_trials(
            trials
        )

        df_dict.setdefault(
            "experimental group",
            [experimental_group_label] * len(flee_probabilities),
        )
        df_dict.setdefault("n flees", n_flees)
        df_dict.setdefault("n non flees", n_non_flees)
        df_dict.setdefault("flee probability", flee_probabilities)

        group_df = pd.DataFrame.from_dict(df_dict)
        df = df.append(group_df, ignore_index=True)
    df.boxplot(by="experimental group", ax=ax, rot=90, grid=False)
    plotting.format_plots(ax)
    return df


def plot_all_metrics_mouse_avgs(
    experimental_labels, trial_type="pre_test", ax=None
):
    all_df = pd.DataFrame()
    for experimental_group_label in experimental_labels:
        trials = load_trials_from_label(experimental_group_label, trial_type)
        ltg = MouseLoomTrialGroup(trials, experimental_group_label)
        df = ltg.get_mouse_avg_df(trials, experimental_group_label)
        all_df = all_df.append(df)
    all_df.boxplot(by="experimental group", ax=ax, rot=90, grid=False)
    plotting.format_plots(ax)
    return all_df


def plot_all_metrics_trials(experimental_group_labels, trial_type="pre_test"):
    all_dfs = pd.DataFrame()
    for experimental_group_label in experimental_group_labels:
        trials = load_trials_from_label(experimental_group_label, trial_type)
        mtg = ExperimentalConditionGroup()

        df = mtg.to_df()
        all_dfs = all_dfs.append(df, ignore_index=True)

    fig, axes = plt.subplots(1, len(MouseLoomTrialGroup.analysed_metrics()))

    all_dfs.boxplot(
        column=MouseLoomTrialGroup.analysed_metrics(),
        by="condition",
        ax=axes,
        grid=False,
        rot=90,
    )
    plotting.format_plots(axes)
    return all_dfs


def get_trials(mouse_ids, trial_type):
    trials = []
    for mid in mouse_ids:
        msg = MouseSessionGroup(mid)
        pre, post = msg.get_pre_and_post_test_trials()
        if trial_type == "pre_test":
            trials.extend(pre)
        elif trial_type == "post_test":
            trials.extend(post)
    return trials


def load_trials_from_label(experimental_group_label, trial_type="pre_test"):
    trials = []
    mouse_ids = get_mouse_ids_in_experiment(experimental_group_label)
    return get_trials(mouse_ids, trial_type)


def plot_trials_with_habituation(mtg, trial_type="pre_test", habit_limit=5):
    """
    
    :param MouseTrialGroup mtg:  
    :param trial_type: 
    :param habit_limit: 
    :return: 
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 1))
    for t in mtg.get_trials_of_type(trial_type):
        plt.sca(axes[1])
        color = "r" if t.is_flee() else "k"
        plt.plot(t.normalised_x_track, color=color)
    for t in mtg.get_trials_of_type("habituation")[0:habit_limit]:
        plt.sca(axes[0])
        t.plot_track_on_image(180, -200)

        plt.sca(axes[2])

    plotting.plot_looms_ax(axes[1])


def plot_habituation_heatmap(mtg, n_trials_to_show=24):
    ref = mtg.get_reference_frame("habituation")
    hm = mtg.habituation_heatmap(n_trials_to_show)
    alphas = hm > 0
    cmap = plt.cm.RdYlBu

    colors = plt.Normalize(0, hm.max())(hm)
    colors = cmap(colors)
    colors[..., -1] = alphas

    fig, ax = plt.subplots()
    ax.imshow(ref)
    ax.imshow(colors)


def plot_photometry_with_thresholds(pre_trials, post_trials, rescale_factor=1):
    if not all([os.path.split(t.video_path)] for t in pre_trials):
        warnings.warn("this trial group consists of multiple sessions")

    if not all([os.path.split(t.video_path)] for t in post_trials):
        warnings.warn("this trial group consists of multiple sessions")

    scale_factor = 30
    # cumsum_sf = 0.5 #2
    thresholds = []
    fig, axes = plt.subplots(2, max(len(pre_trials), len(post_trials)))
    if pre_trials[0].stimulus_type == "loom":
        plotting.plot_looms(fig)
    else:
        plotting.plot_stimulus(fig)

    # get_rescale_factors_to_sessions_max()

    for j, tg in enumerate([pre_trials, post_trials]):
        for i, t in enumerate(tg):
            cumsum = t.get_cumsum(rescale_factor)
            ax = axes[j][i]
            plt.sca(ax)
            plt.ylim([-0.1, 1])
            plt.xlim([0, 600])

            color = "r" if t.is_flee() else "k"
            t.plot_delta_f_with_track(color, scale_factor / rescale_factor)
            # ax.plot(t.events_trace * scale_factor, color='y', linestyle='--')
            ax.plot(cumsum, color="k", alpha=0.3, linewidth=3)

            if j == 0:
                latency = t.estimate_latency(False)
                plt.plot(latency, t.normalised_x_track[latency], "o")
                thresholds.append(cumsum[latency])
            else:
                plt.sca(ax)
                [plt.axhline(t) for t in thresholds]


def plot_habituation(habituation_trials, max_df=1):
    all_trials = []
    summary = []

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    for i, t in enumerate(habituation_trials):
        plt.sca(axes[0])
        plt.plot(t.delta_f() + i / 100 * max_df, color="k")
        plt.sca(axes[1])
        plt.plot(
            t.cumulative_sum_raw / max_df,
            linewidth=3,
            color=str((len(habituation_trials) + 1) / (i + 1)),
        )
        plt.xlim([100, 400])
        ax = plt.gca()
        plotting.plot_looms_ax(ax)
        all_trials.append(t.delta_f())
        summary.append(max(t.cumulative_sum_raw[200:350]))
    plt.sca(axes[2])
    plt.imshow(all_trials, aspect="auto", origin=0, vmin=0, vmax=max_df / 50)

    plt.sca(axes[3])
    summary = np.array(summary)
    plt.plot(np.arange(len(summary)), summary / max_df, "o", color="k")
    plt.plot(np.arange(len(summary)), summary / max_df, color="k")
    plt.ylim([0, 1])


def plot_all(trials, max_df=1):
    greys = cm.get_cmap("Greys")

    all_trials = []
    summary = []

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

    color_range = np.linspace(0.1, 1, len(trials[3:27]))
    crange = [greys(c) for c in color_range]

    for i, t in enumerate(trials):
        if i < 3:
            color = "r"
        elif 3 <= i < 27:
            color = crange[i - 3]
        elif i > 26:
            color = "b"

        plt.sca(axes[0])
        plt.plot(t.delta_f() + i / 100 * max_df, color=color)
        plt.sca(axes[1])
        plt.plot(t.cumulative_sum_raw / max_df, linewidth=3, color=color)
        plt.xlim([100, 400])
        ax = plt.gca()
        plotting.plot_looms_ax(ax)
        all_trials.append(t.delta_f())
        summary.append(max(t.cumulative_sum_raw[200:350]))
    plt.sca(axes[2])
    plt.imshow(all_trials, aspect="auto", origin=0, vmin=0, vmax=max_df / 50)

    plt.sca(axes[3])
    summary = np.array(summary)
    x = np.arange(len(summary))
    plt.plot(x[:3], summary[:3] / max_df, "o", color="r")
    plt.scatter(x[3:27] + 1, summary[3:27] / max_df, c=crange)
    plt.plot(x[27:30] + 2, summary[27:30] / max_df, "o", color="b")

    plt.plot(x[:3], summary[:3] / max_df, color="r")
    plt.plot(x[3:27] + 1, summary[3:27] / max_df, color="k")
    plt.plot(x[27:30] + 2, summary[27:30] / max_df, color="b")

    plt.ylim([0, 1])
    return fig


def get_all(trials, max_df=1):

    summary = []

    for t in trials:
        summary.append(max(t.cumulative_sum_raw[200:350]) / max_df)

    return np.array(summary)


def get_max_stimulus_response(trials):
    return get_max_integral(trials)
    # return max(max(t.cumulative_sum_raw) for t in all_trials)


def get_max_integral(trials):
    return max(np.nanmax(t.integral_downsampled()) for t in trials)


def get_normalising_factor_from_mouse(trial):
    """

    :param looming_spots.db.loomtrial.LoomTrial trial:
    :return:
    """
    mtg = loom_trial_group.MouseLoomTrialGroup(trial.mouse_id)
    return get_max_stimulus_response(mtg.all_trials)


def get_normalised_pre_post_cumsums(pre_trials, post_trials):
    """
    must be used with single mouse because normalisation

    :param pre_trials:
    :param post_trials:
    :return:
    """

    normalising_factor = get_normalising_factor_from_mouse(pre_trials[0])
    pre_cumsum = [
        t.cumulative_sum_raw / normalising_factor for t in pre_trials
    ]
    post_cumsum = [
        t.cumulative_sum_raw / normalising_factor for t in post_trials
    ]

    return pre_cumsum, post_cumsum


def plot_all_normalised_pre_post_cumsums(mouse_ids):
    all_pre_cumsums = []
    all_post_cumsums = []
    ax = plt.subplot(111)
    for mid in mouse_ids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        pre_cumsum, post_cumsum = get_normalised_pre_post_cumsums(
            mtg.loom_trials()[:3], mtg.loom_trials()[27:30]
        )
        all_pre_cumsums.extend(pre_cumsum)
        all_post_cumsums.extend(post_cumsum)

    for all_cumsums, color in zip(
        [all_pre_cumsums, all_post_cumsums], ["r", "k"]
    ):
        mean_cumsums = np.mean(all_cumsums, axis=0)
        std_cumsums = np.std(all_cumsums, axis=0)
        t = np.arange(len(mean_cumsums))
        ax.plot(mean_cumsums, color=color, linewidth=3)
        ax.fill_between(
            t,
            mean_cumsums + std_cumsums,
            mean_cumsums - std_cumsums,
            facecolor=color,
            alpha=0.5,
        )


def plot_all_with_integral(mtg):
    pre_test_trials = mtg.pre_test_trials()[:3]
    habituation_trials = mtg.habituation_trials()[:24]
    post_test_trials = mtg.post_test_trials()[:3]
    all_trials = [pre_test_trials, habituation_trials, post_test_trials]

    norm_factor = get_normalising_factor_from_mouse(mtg.pre_test_trials[0])

    greys = cm.get_cmap("Greys")

    all_trials = []
    summary = []

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))

    habituation_color_range = np.linspace(0.1, 1, len(habituation_trials))
    crange = [greys(c) for c in habituation_color_range]

    for i, t in enumerate(all_trials):
        if i < 3:
            color = "r"
        elif 3 <= i < 27:
            color = crange[i - 3]
        elif i > 26:
            color = "b"

        plt.sca(axes[0])
        plt.plot(t.delta_f() + i / 5 * norm_factor, color=color)

        plt.sca(axes[1])
        plt.plot(t.integral, linewidth=3, color=color)
        ax = plt.gca()
        plotting.plot_upsampled_looms_ax(ax)
        plt.ylim([-0.01, norm_factor])

        plt.xlim([100 * 10000 / 30, 400 * 10000 / 30])
        ax = plt.gca()
        plotting.plot_looms_ax(ax)
        all_trials.append(t.delta_f())
        summary.append(t.integral_at_end())
    plt.sca(axes[2])
    plt.imshow(all_trials, aspect="auto", origin=0, vmin=0, vmax=norm_factor)

    plt.sca(axes[3])
    summary = np.array(summary)
    x = np.arange(len(summary))
    plt.plot(x[:3], summary[:3] / norm_factor, "o", color="r")
    plt.scatter(x[3:27] + 1, summary[3:27] / norm_factor, c=crange)
    plt.plot(x[27:30] + 2, summary[27:30] / norm_factor, "o", color="b")

    plt.plot(x[:3], summary[:3] / norm_factor, color="r")
    plt.plot(x[3:27] + 1, summary[3:27] / norm_factor, color="k")
    plt.plot(x[27:30] + 2, summary[27:30] / norm_factor, color="b")

    plt.ylim([0, 1])
    return fig


def plot_integrals_latencies_tracks_and_df(
    trials, cmap_key="Greys", fig=None, axes=None
):
    """
    plot the integrals, average delta F and tracks for given list of trials

    >>> mtg = loom_trial_group.MouseLoomTrialGroup('898990')
    >>> fig, axes = plot_integrals_latencies_tracks_and_df(mtg.habituation_trials()[:24],'Greys')
    >>> plot_integrals_latencies_tracks_and_df(mtg.pre_test_trials()[:3],'Reds', fig, axes)
    >>> plot_integrals_latencies_tracks_and_df.plot_group(mtg.post_test_trials()[:3],'Blues', fig, axes)

    :param trials:
    :param cmap_key:
    :param fig:
    :param axes:
    :return:
    """
    cmap = cm.get_cmap(cmap_key)
    trial_range = np.linspace(0.1, 1, len(trials))
    crange = [cmap(c) for c in trial_range]

    norm_factor = get_normalising_factor_from_mouse(trials[0])
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    for i, t in enumerate(trials):
        color = crange[i]
        plt.sca(axes[0])
        plt.plot(t.integral_downsampled(), linewidth=3, color=color)
        plt.ylim([-0.01, norm_factor])
        plt.xlim([100, 400])
        plt.sca(axes[1])
        t.plot_track(color=color)

    ax = plt.sca(axes[2])
    avg_df = np.mean([t.delta_f() for t in trials], axis=0)
    plt.plot(avg_df, color=color)
    plt.ylim([-0.01, max(avg_df) + 0.02])
    plt.xlim([180, 350])
    plotting.plot_looms(fig)
    return fig, axes


def plot_all_LSIE(mouse_id):
    mtg = loom_trial_group.MouseLoomTrialGroup(mouse_id)
    fig, axes = plot_integrals_latencies_tracks_and_df(
        mtg.habituation_trials()[:24], "Greys"
    )
    plot_integrals_latencies_tracks_and_df(
        mtg.pre_test_trials()[:3], "Reds", fig, axes
    )
    plot_integrals_latencies_tracks_and_df(
        mtg.post_test_trials()[:3], "Blues", fig, axes
    )

    plt.sca(axes[0])
    plt.axvline(
        get_avg_latency(mtg.pre_test_trials()[:3]), linestyle="--", color="k"
    )

    plt.sca(axes[1])
    plt.axvline(
        get_avg_latency(mtg.pre_test_trials()[:3]), linestyle="--", color="k"
    )

    plt.sca(axes[2])
    plt.axvline(
        get_avg_latency(mtg.pre_test_trials()[:3]), linestyle="--", color="k"
    )

    return fig, axes


def get_avg_latency(trials):
    return np.mean([t.estimate_latency(False) for t in trials])


def plot_trials(trials):
    data = np.array([t.delta_f() for t in trials])
    mean = np.mean(data, axis=0)
    plt.plot(data.T, color="b", linewidth=0.5)
    plt.plot(mean, linewidth=3, color="k")
    plt.ylim([-0.01, 0.1])


def plot_by_classification(trials):
    flees = []
    flees_integrals = []
    non_flees = []
    non_flees_integrals = []
    for t in trials:
        flees.append(t.delta_f()) if t.is_flee() else non_flees.append(
            t.delta_f()
        )
        flees_integrals.append(
            t.integral
        ) if t.is_flee() else non_flees_integrals.append(t.integral)

    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(np.array(flees).T, color="r", linewidth=0.5)
    plt.plot(np.mean(flees, axis=0), color="r", linewidth=4)
    plt.ylim([-0.01, 0.1])

    ax2 = plt.subplot(312)
    plt.plot(np.array(non_flees).T, color="k")
    plt.plot(np.mean(non_flees, axis=0), color="k", linewidth=4)
    plt.ylim([-0.01, 0.1])

    ax3 = plt.subplot(313)
    plt.plot(np.mean(flees_integrals, axis=0), color="b")
    plt.plot(np.mean(non_flees_integrals, axis=0), color="y")
    plt.ylim([-0.01, 0.1])
    plotting.plot_looms_ax(ax1)
    plotting.plot_looms_ax(ax2)
    plotting.plot_upsampled_looms_ax(ax3)

    return fig
