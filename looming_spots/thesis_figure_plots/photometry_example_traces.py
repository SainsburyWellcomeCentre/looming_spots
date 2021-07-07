import numpy as np
from looming_spots.db.constants import FIGURE_DIRECTORY, FRAME_RATE, N_SAMPLES_TO_SHOW
from looming_spots.db import loom_trial_group, experimental_log
from looming_spots.thesis_figure_plots.randomised_contrast_escape_curves_lesions import flatui
from looming_spots.trial_group_analysis import photometry_habituations
from pingouin import ancova, partial_corr

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

pre_test_24hr = ["074743",  "074746", "895773", "953828", "953829"] #"074744",
pre_test_sameday = ["898989", "898990", "987657", "977659"]

contrast_curves = ["898992", "916063", "921000", "907822"]

any_escape = ["074743", "895773", "953828", "898989", "898990"]
no_escape = ["074746", "957073"]
d1_pre_test_24hr = ['1012998', '1034952', '1034953', '1111674']  # '1012996', '1111675' DNE
d2_pre_test_24hr = ['1004807', '1016719', '1029104', '1029105', '1100840', '1100845']


d1_suppressed = ['1012998', '1034953', '1111674']
d2_suppressed = ['1004807', '1016719', '1029105', '1100840', '1100845']

d1_d2_non_suppressed = ['1034952', '1029104']
d2_escape_curve = ['1095775', '1095777','1095779']


def plot_pre_post_photometry_trials_lsie(mtg):
    for t in mtg.pre_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig(
            f"/{FIGURE_DIRECTORY}/loom_{t.loom_number}_{mtg.mouse_id}_pre_test.eps",
            format="eps",
        )
        plt.close("all")
    for t in mtg.post_test_trials()[:3]:
        fig = t.plot_track_and_delta_f()
        fig.savefig(
            f"/{FIGURE_DIRECTORY}/loom_{t.loom_number}_{mtg.mouse_id}_post_test.eps",
            format="eps",
        )
        plt.close("all")


def plot_pre_post_photometry_trials_lsie_all_mtgs(mtgs):
    fig, axes = plt.subplots(2,1)
    avg_pre_test_signal=[]
    avg_post_test_signal=[]
    for mtg in mtgs:
        for t in mtg.pre_test_trials()[:3]:
            t.plot_track_and_delta_f(axes=axes)
            avg_pre_test_signal.append(t.delta_f()[:600])
    axes[1].plot(np.mean(avg_pre_test_signal, axis=0), color='k', linewidth=3)
    fig2, axes2 = plt.subplots(2,1)
    for mtg in mtgs:
        for t in mtg.post_test_trials()[:3]:
            t.plot_track_and_delta_f(axes=axes2)
            avg_post_test_signal.append(t.delta_f()[:600])
    axes2[1].plot(np.mean(avg_post_test_signal, axis=0), color='k', linewidth=3)
    fig.savefig(
        f"/{FIGURE_DIRECTORY}/loom_{t.loom_number}_{mtg.mouse_id}_pre_test.eps",
        format="eps",
    )

    fig2.savefig(
        f"/{FIGURE_DIRECTORY}/loom_{t.loom_number}_{mtg.mouse_id}_post_test.eps",
        format="eps",
    )
    #plt.close("all")


def plot_pre_post_photometry_lsie(mtg):

    fig1 = plt.figure()
    ax1 = plt.subplot(211)
    avg_df = []

    for t in mtg.pre_test_trials()[:3]:
        t.plot_track()
    mtg.pre_test_trials()[0].plot_stimulus()
    ax2 = plt.subplot(212)
    for t in mtg.pre_test_trials()[:3]:
        plt.plot(t.delta_f()[:N_SAMPLES_TO_SHOW])
        avg_df.append(t.delta_f()[:N_SAMPLES_TO_SHOW])
    mtg.pre_test_trials()[0].plot_stimulus()
    plt.plot(np.mean(avg_df, axis=0), linewidth=4)
    plt.ylim([-0.01, 0.15])

    avg_df = []
    fig2 = plt.figure()
    ax3 = plt.subplot(211)
    for t in mtg.post_test_trials()[:3]:
        t.plot_track()
    mtg.pre_test_trials()[0].plot_stimulus()

    ax4 = plt.subplot(212)
    for t in mtg.post_test_trials()[:3]:
        avg_df.append(t.delta_f()[:N_SAMPLES_TO_SHOW])
        plt.plot(t.delta_f()[:N_SAMPLES_TO_SHOW])
    plt.plot(np.mean(avg_df, axis=0), linewidth=4)

    plt.ylim([-0.01, 0.15])
    mtg.pre_test_trials()[0].plot_stimulus()

    fig1.savefig(
        f"/home/slenzi/pre_post_LSIE_photometry/{mtg.mouse_id}_pre_test.eps",
        format="eps",
    )
    fig2.savefig(
        f"/home/slenzi/pre_post_LSIE_photometry/{mtg.mouse_id}_post_test.eps",
        format="eps",
    )


def plot_pre_test_max_integral():
    mids = [
        "074744",
        "074746",
        "074743",
        "898989",
        "898990",
        "895773",
        "898992",
        "916063",
        "921000",
        "907822",
    ]
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    all_max_integrals = []
    for mtg in mtgs:
        norm_factor = max([np.nanmax(t.integral) for t in mtg.loom_trials()])
        if len(mtg.contrasts()) == 0:
            pre_trials = mtg.loom_trials()[:3]
        else:
            pre_trials = [t for t in mtg.all_trials if t.contrast == 0][:3]
        max_integrals = [
            np.nanmax(t.integral) for t in pre_trials
        ] / norm_factor
        all_max_integrals.append(max_integrals)

    plt.bar(1, np.mean(all_max_integrals), alpha=0.2)
    plt.scatter(
        np.ones_like(np.array(all_max_integrals).flatten()),
        np.array(all_max_integrals).flatten(),
        color="w",
        edgecolor="k",
    )


def plot_all_pre_tests(mids):
    mids = [
        "074744",
        "074746",
        "074743",
        "898989",
        "898990",
        "895773",
        "898992",
        "916063",
        "921000",
        "907822",
    ]

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    all_tracks = []
    all_delta_f = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        norm_factor = max([np.nanmax(t.delta_f()) for t in mtg.loom_trials()])
        if len(mtg.contrasts()) == 0:
            pre_trials = mtg.loom_trials()[:3]
        else:
            pre_trials = [t for t in mtg.all_trials if t.contrast == 0][:3]

        for t in pre_trials:
            norm_df = t.delta_f() / norm_factor

            t.plot_track(ax1)
            ax2.plot(norm_df, color="grey", linewidth=0.5)
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(norm_df)
    ax1.plot(np.mean(all_tracks, axis=0), color="k", linewidth=3)
    ax2.plot(np.mean(all_delta_f, axis=0), color="k", linewidth=3)


def plot_all_post_tests(mids):
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    all_tracks = []
    all_delta_f = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        norm_factor = max([np.nanmax(t.delta_f()) for t in mtg.loom_trials()])
        for t in mtg.post_test_trials()[:3]:
            norm_df = t.delta_f() / norm_factor

            t.plot_track(ax1)
            ax2.plot(norm_df, color="grey", linewidth=0.5)
            all_tracks.append(t.normalised_x_track)
            all_delta_f.append(norm_df)

    ax1.plot(np.mean(all_tracks, axis=0), color="k", linewidth=3)
    ax2.plot(np.mean(all_delta_f, axis=0), color="k", linewidth=3)
    ax2.set_ylim([-0.1, 1.1])
    return fig


def plot_photometry_by_contrast(mtg):
    for contrast in np.unique(mtg.contrasts()):  # TODO: extract to mtg?
        tracks = []
        avg_df = []
        fig, axes = plt.subplots(2, 1)

        for t in mtg.all_trials:
            if t.contrast == contrast:
                tracks.append(t.normalised_x_track)
                avg_df.append(t.delta_f())
                axes[0].plot(t.normalised_x_track, color="grey", linewidth=0.5)
                axes[1].plot(t.delta_f(), color="grey", linewidth=0.5)

        axes[0].plot(np.mean(tracks, axis=0), color="r")
        axes[1].plot(np.mean(avg_df, axis=0), color="k")
        t.plot_stimulus()


def plot_delta_f_max_integral_against_contrast(mtgs):
    plt.figure()

    for mtg in mtgs:
        trials = mtg.all_trials
        normalising_factor = max([np.nanmax(t.integral) for t in trials])
        for t in trials:
            ca_response = np.nanmax(t.integral) / normalising_factor
            color = "r" if t.is_flee() else "k"
            plt.plot(t.contrast, ca_response, "o", color=color)


def plot_delta_f_at_latency_against_contrast(mtgs):
    plt.figure()

    for mtg in mtgs:
        test_contrast_trials = [
            t for t in mtg.loom_trials() if t.contrast == 0
        ]
        pre_test_latency = np.nanmean(
            [t.estimate_latency(False) for t in test_contrast_trials]
        )
        normalising_factor = max(
            [
                np.nanmax(
                    [
                        t.integral_escape_metric(int(pre_test_latency))
                        for t in mtg.loom_trials()
                    ]
                )
            ]
        )

        for i, t in zip(np.arange(18), mtg.all_trials):
            ca_response = (
                np.nanmax(t.integral_escape_metric(int(pre_test_latency)))
                / normalising_factor
            )
            color = "r" if t.is_flee() else "k"
            plt.plot(t.contrast, ca_response, "o", color=color)


def plot_lsie_evoked_signals_binned_by_contrast(
    groups=(pre_test_sameday, pre_test_24hr),
    colors=flatui[:2][::-1],
    labels=("same day pre-test (n={})", "24 hr pre-test n={}"), ax=None
):
    if ax is None:
        plt.figure()
        ax=plt.subplot(111)
    plt.sca(ax)
    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]

        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
    plt.legend(
        [labels[0].format(len(groups[0])), labels[1].format(len(groups[1]))]
    )
    plt.title("loom evoked DA signal in ToS during LSIE protocol")
    plt.xlabel("contrast (binned 3 trials each)")
    plt.ylabel("integral of dF/F at avg. pre-test escape latency")

    plt.ylim([0.1, 0.6])


def plot_LSIE_evoked_signals_all_mice(
    groups=(pre_test_sameday, pre_test_24hr),
    colors=("b", "k"),
    labels=("same day pre-test (n={})", "24 hr pre-test n={}"),
):
    if "074744" in pre_test_24hr:
        pre_test_24hr.remove("074744")
    plt.figure()
    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
    plt.legend(
        [labels[0].format(len(groups[0])), labels[1].format(len(groups[1]))]
    )  # two loops because legend

    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        for mtg in mtgs:
            photometry_habituations.plot_habituation_curve_with_sem(
                [mtg], color, linewidth=0.5, plot_dots=False
            )

    plt.title("loom evoked DA signal in ToS during LSIE protocol")
    plt.xlabel("contrast (binned 3 trials each)")
    plt.ylabel("integral of dF/F at avg. pre-test escape latency")
    plt.show()


def plot_LSIE_bars(groups=(pre_test_sameday, pre_test_24hr)):
    if "074744" in pre_test_24hr:
        pre_test_24hr.remove("074744")
    mtgs_sup = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in groups[1]]
    mtgs_non_sup = [
        loom_trial_group.MouseLoomTrialGroup(mid) for mid in groups[0]
    ]

    fig = plt.figure(figsize=(3, 5))
    plt.title("ToS DA signal before and after LSIE protocol")
    plt.subplot(121)
    plt.ylabel("integral dF/F at avg. escape latency in pre-test")
    photometry_habituations.plot_integral_at_latency_bars(
        mtgs_sup, (flatui[0], flatui[0])
    )
    plt.subplot(122)
    photometry_habituations.plot_integral_at_latency_bars(
        mtgs_non_sup, (flatui[1], flatui[1])
    )
    for ax in fig.axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylim([0, 1.2])
        plt.sca(ax)
        plt.subplots_adjust(bottom=0.35, left=0.3, right=0.8)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_escape_metrics_variable_contrast_experiments(
    metrics=(
        "latency to escape",
        "speed",
        "acceleration",
        "time in safety zone",
        "classified as flee",
    ),mouse_ids=("898992", "916063", "921000", "907822"),
):

    all_dfs = []
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]

    for metric in metrics:
        df = photometry_habituations.get_signal_metric_dataframe_variable_contrasts(
            mtgs, metric
        )
        all_dfs.append(df)

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(
            metric,
            "ca signal",
            df,
            fit_reg=True,
            palette=sns.cubehelix_palette(8)[::-1],
            legend=False,
        )

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(
            metric, "ca signal", df, hue="escape", fit_reg=False, legend=True
        )

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        fig = sns.lmplot(
            metric,
            "ca signal",
            df,
            hue="contrast",
            fit_reg=False,
            palette=sns.cubehelix_palette(8)[::-1],
            legend=False,
        )
    plt.figure()
    sns.pointplot('contrast', 'escape', data=df, err_style='bars', scatter_kws={'s': 150, 'linewidth': 3})
    plt.ylim([-0.1, 1.1])
    plt.figure()
    sns.pointplot('contrast', 'ca signal', data=df, err_style='bars', scatter_kws={'s': 150, 'linewidth': 3})
    plt.ylim([-0.1, 1.1])

    return all_dfs


def plot_ca_vs_metric_signal_before_after_lsie(
    metrics=(
        "latency to escape",
        "speed",
        "acceleration",
        "time in safety zone",
    )
):
    all_dfs = []
    mouse_ids = [
        "074743",
        "074746",
        "895773",
        "953828",
        "953829",
        "898989",
        "898990",
    ]

    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in metrics:
        df = photometry_habituations.get_signal_metric_dataframe(mtgs, metric)
        all_dfs.append(df)

    for i, (df, metric) in enumerate(zip(all_dfs, metrics)):
        g = sns.lmplot(
            metric, "ca signal", data=df, hue="escape", fit_reg=False
        )
        sns.regplot(
            x=metric,
            y="ca signal",
            data=df,
            scatter=False,
            ax=g.axes[0, 0],
            color="k",
        )
    return all_dfs


def plot_lsie_suppression_over_variable_contrast(hue="test type"):
    all_dfs = []
    mouse_ids = ["898992", "916063", "921000", "907822"]
    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in [
        "latency to escape",
        "speed",
        "acceleration",
        "time in safety zone",
    ]:
        df = photometry_habituations.get_signal_metric_dataframe_variable_contrasts(
            mtgs, metric
        )
        all_dfs.append(df)
    all_dfs_pre_post = []

    mouse_ids = ["074743", "074746", "895773", "953828", "953829"]

    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mouse_ids]
    for metric in [
        "latency to escape",
        "speed",
        "acceleration",
        "time in safety zone",
    ]:
        df = photometry_habituations.get_signal_metric_dataframe(mtgs, metric)
        all_dfs_pre_post.append(df)

    metrics = [
        "latency to escape",
        "speed",
        "acceleration",
        "time in safety zone",
    ]
    for dfa, dfb, metric in zip(all_dfs, all_dfs_pre_post, metrics):
        joined = dfa.append(dfb)
        g = sns.lmplot(
            metric, "ca signal", data=joined, hue=hue, fit_reg=False
        )
        sns.regplot(
            x=metric,
            y="ca signal",
            data=joined,
            scatter=False,
            ax=g.axes[0, 0],
            color="k",
        )
        g.savefig(
            f"/home/slenzi/figures/photometry_contrasts/pre_post_24hr_group_on_contrast_ca_curve_{metric}_hue_{hue.replace(' ', '_')}.eps",
            format="eps",
        )
        g.savefig(
            f"/home/slenzi/figures/photometry_contrasts/pre_post_24hr_group_on_contrast_ca_curve_{metric}_hue_{hue.replace(' ', '_')}.png",
            format="png",
        )


def plot_habituation_trialwise_with_lowess_fit(groups=(pre_test_24hr, pre_test_sameday),
                                               group_labels=("24 hr", "same day"), x_measure="trial number",
                                               y_measure="ca signal"):

    plt.figure(figsize=(10, 6))
    if "074744" in pre_test_24hr:
        pre_test_24hr.remove("074744")

    mtgs_24 = [
        loom_trial_group.MouseLoomTrialGroup(mid) for mid in (groups[0])
    ]
    mtgs_imm = [
        loom_trial_group.MouseLoomTrialGroup(mid) for mid in (groups[1])
    ]

    df = photometry_habituations.habituation_df(
        [mtgs_24, mtgs_imm], list(group_labels)
    )

    plt.subplot(141)
    plot_group_by_test_type(df, group_labels[0], x_measure, y_measure)

    plt.subplot(142)
    plot_group_by_test_type(df, group_labels[1], x_measure, y_measure)

    plt.subplot(143)
    plot_habituation_by_group(df, group_labels, x_measure, y_measure)

    ax = plt.subplot(144)
    plot_lsie_evoked_signals_binned_by_contrast(groups=groups, labels=group_labels, ax=ax)
    return df


def plot_habituation_by_group(df, group_labels, x_measure, y_measure):
    data = df[df["test type"] == "habituation"]
    # sns.scatterplot(
    #     x_measure,
    #     y_measure,
    #     hue="group label",
    #     data=data,
    #     palette=flatui[:len(group_labels)],
    #     edgecolor='k',
    #     legend=False
    # )

    sns.lineplot(
        x_measure,
        y_measure,
        hue="group label",
        data=data,
        palette=flatui[:len(group_labels)],
        legend=False
    )
    plt.ylim([-0.1, 1.1])


def plot_group_by_test_type(df, group_label, x_measure="trial number", y_measure='ca signal'):
    palette = ['k', 'Grey', 'r']
    data = df[df["group label"] == group_label]
    sns.scatterplot(
        x_measure,
        y_measure,
        hue="test type",
        data=data,
        palette=palette[:3],
        edgecolor='k',
        legend=False
    )

    sns.lineplot(
        x_measure,
        y_measure,
        hue="test type",
        data=data,
        palette=palette[:3],
        legend=False
    )

    plt.ylim([-0.1, 1.1])


def get_first_loom_response_by_contrast(contrast_curve_mids=contrast_curves, n_samples=30, start=200):
    mtgs = [
        loom_trial_group.MouseLoomTrialGroup(mid)
        for mid in contrast_curve_mids
    ]
    df_all = pd.DataFrame()
    for contrast in np.unique(mtgs[0].contrasts()):
        contrast_response_dict = {}
        print(contrast)
        avg_df = []
        pooled_trials_at_contrast = get_trials_of_contrast(mtgs, contrast, 4)

        for t in pooled_trials_at_contrast:
            avg_df.append(t.delta_f()[start:start+n_samples])
        # normalised_signals_at_contrast = get_trials_of_contrast_normalised(mtgs, contrast, 4)

        avg_response_at_contrast = np.mean(avg_df, axis=0)
        contrast_response_dict.setdefault("signal", avg_response_at_contrast)
        contrast_response_dict.setdefault(
            "contrast", [contrast] * len(avg_response_at_contrast)
        )
        contrast_response_dict.setdefault(
            "timepoint", np.arange(len(avg_response_at_contrast))
        )
        df = pd.DataFrame.from_dict(contrast_response_dict)
        df_all = df_all.append(df)
    return df_all


def get_trials_of_contrast(mtgs, contrast, n_trials_to_take):
    pooled_trials_at_contrast = []

    for mtg in mtgs:
        trials_at_contrast = [
            t for t in mtg.all_trials[:18] if t.contrast == contrast
        ]
        pooled_trials_at_contrast.extend(trials_at_contrast[:n_trials_to_take])

    return pooled_trials_at_contrast


def get_trials_of_contrast_normalised(mtgs, contrast, n_trials_to_take):
    pooled_signal_at_contrast = []

    for mtg in mtgs:
        normalising_factor = max(
            [max(t.delta_f()[200:230]) for t in mtg.all_trials]
        )
        trials_at_contrast = [
            t for t in mtg.all_trials[:18] if t.contrast == contrast
        ]
        normalised_signals_at_contrast = [
            t.delta_f()[200:230] / normalising_factor
            for t in trials_at_contrast
        ]
        pooled_signal_at_contrast.extend(
            normalised_signals_at_contrast[:n_trials_to_take]
        )

    return pooled_signal_at_contrast


def get_post_lsie_signal_df(groups, metric):
    all_df = pd.DataFrame()
    for group in groups:
        mtgs = experimental_log.get_mtgs_in_experiment(group)
        mtgs = [m for m in mtgs if m.mouse_id != "074744"]

        for mtg in mtgs:
            mtg_dict = {}
            pre_vals = []
            post_vals = []
            post_has_escaped_vals = []
            post_metric_vals = []

            pre_test_latency = np.nanmean(
                [t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]]
            )
            normalising_factor = max(
                [
                    np.nanmax(
                        [
                            t.integral_escape_metric(int(pre_test_latency))
                            for t in mtg.loom_trials()[:30]
                        ]
                    )
                ]
            )

            for t in mtg.pre_test_trials()[:3]:
                val = (
                    t.integral_escape_metric(int(pre_test_latency))
                    / normalising_factor
                )
                pre_vals.append(val)

            for t in mtg.post_test_trials()[:3]:
                val = (
                    t.integral_escape_metric(int(pre_test_latency))
                    / normalising_factor
                )
                post_vals.append(val)
                post_has_escaped_vals.append(
                    t.has_escaped_by(int(pre_test_latency))
                )
                post_metric_vals.append(t.metric_functions[metric]())

            mtg_dict.setdefault("group", [group] * len(pre_vals))
            mtg_dict.setdefault("pre test values", pre_vals)
            mtg_dict.setdefault("post test values", post_vals)
            mtg_dict.setdefault(f"post test {metric} values", post_metric_vals)
            mtg_dict.setdefault(
                "suppression value",
                np.array([np.mean(pre_vals)] * len(post_vals)) - post_vals,
            )
            mtg_dict.setdefault(
                "escape by pretest latency", post_has_escaped_vals
            )
            mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(pre_vals))
            mtg_df = pd.DataFrame.from_dict(mtg_dict)
            all_df = all_df.append(mtg_df)
    return all_df


def get_pre_lsie_signal_df(mtgs):
    all_df = pd.DataFrame()

    for mtg in mtgs:
        mtg_dict = {}
        pre_vals = []
        pre_metrics = []
        loom_numbers = []

        pre_test_latency = np.nanmean(
            [t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]]
        )
        normalising_factor = max(
            [
                np.nanmax(
                    [
                        t.integral_escape_metric(int(pre_test_latency))
                        for t in mtg.loom_trials()[:30]
                    ]
                )
            ]
        )

        for t in mtg.pre_test_trials()[:3]:
            val = (
                t.integral_escape_metric(int(pre_test_latency))
                / normalising_factor
            )
            pre_vals.append(val)
            pre_metrics.append((t.estimate_latency(False)-200)/FRAME_RATE)
            loom_numbers.append(t.loom_number)
        mtg_dict.setdefault("pre test values", pre_vals)
        mtg_dict.setdefault("pre test metrics", pre_metrics)
        mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(pre_vals))
        mtg_dict.setdefault("loom number", loom_numbers)

        mtg_df = pd.DataFrame.from_dict(mtg_dict)
        all_df = all_df.append(mtg_df)
    return all_df


def get_trace_normalising_factor_during_stimulus(trials):
    return max([max(t.delta_f()[200:450]) for t in trials])

def get_normalising_factor_latency_metric(trials, pre_test_latency):
    normalising_factor = max(
        [
            np.nanmax(
                [
                    t.integral_escape_metric(int(pre_test_latency))
                    for t in trials
                ]
            )
        ]
    )
    return normalising_factor


def plot_d1_d2_gcamp_pre_post_normalised():

    groups = [d1_suppressed, d2_suppressed, d1_d2_non_suppressed]
    labels = ['d1-cre-flexGCaMP', 'd2-cre-flexGCaMP']

    for group, label in zip(groups, labels):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        fig_pre, axes_pre = plt.subplots(3, 1)
        fig_post, axes_post = plt.subplots(3, 1)

        for mtg in mtgs:
            norm_factor = get_trace_normalising_factor_during_stimulus(mtg.loom_trials())
            for t, ax in zip(mtg.all_trials[:3], axes_pre):
                color = 'r' if t.is_flee() else 'k'
                ax.plot(t.delta_f()[:600] / norm_factor, color=color)

            for t, ax in zip(mtg.loom_trials()[-3:], axes_post):
                color = 'r' if t.is_flee() else 'k'
                ax.plot(t.delta_f()[:600] / norm_factor, color=color)

        for i, ax in enumerate(axes_pre):
            plt.sca(ax)
            plt.ylim([-0.3, 1.2])
            plt.title(f'pre-test {label}, trial {i}')
            plt.xlabel('samples (30 hz)')
            plt.ylabel('normalised df/f')
            t.plot_stimulus()
        for i, ax in enumerate(axes_post):
            plt.sca(ax)
            plt.ylim([-0.3, 1.2])
            plt.title(f'post-test {label}, trial {i}')
            plt.xlabel('samples (30 hz)')
            plt.ylabel('normalised df/f')
            t.plot_stimulus()

        fig_pre.subplots_adjust(hspace=1)
        fig_post.subplots_adjust(hspace=1)

        fig_pre.savefig(f'{FIGURE_DIRECTORY}pre-test{label}')
        fig_post.savefig(f'{FIGURE_DIRECTORY}post-test{label}')


def plot_d1_d2_gcamp_pre_post_normalised_one_plot():

    groups = [d1_suppressed, d2_suppressed, d1_d2_non_suppressed]
    labels = ['d1-cre-flexGCaMP', 'd2-cre-flexGCaMP']

    for group, label in zip(groups, labels):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]
        fig, axes = plt.subplots(2, 1)
        avg_pre = []
        avg_post = []
        for mtg in mtgs:
            norm_factor = get_trace_normalising_factor_during_stimulus(mtg.loom_trials())
            for t in mtg.all_trials[:3]:
                color = 'r' if t.is_flee() else 'k'
                axes[0].plot(t.delta_f()[:600] / norm_factor, color=color)
                avg_pre.append(t.delta_f()[:600]/norm_factor)
            for t in mtg.loom_trials()[-3:]:
                color = 'r' if t.is_flee() else 'k'
                axes[1].plot(t.delta_f()[:600] / norm_factor, color=color)
                avg_post.append(t.delta_f()[:600]/norm_factor)
        axes[0].plot(np.mean(avg_pre, axis=0), linewidth=3)
        axes[1].plot(np.mean(avg_post, axis=0), linewidth=3)
        for i, ax in enumerate(axes):
            plt.sca(ax)
            plt.ylim([-0.3, 1.2])
            plt.title(f'{label}')
            plt.xlabel('samples (30 hz)')
            plt.ylabel('normalised df/f')
            t.plot_stimulus()

        fig.subplots_adjust(hspace=1)


def plot_LSIE_bars_all_groups(groups=(pre_test_sameday, pre_test_24hr)):
    fig, axes = plt.subplots(1, len(groups), figsize=(5, 5))
    for i,group in enumerate(groups):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]

        plt.ylabel("integral dF/F at avg. escape latency in pre-test")
        ax = plt.sca(axes[i])
        photometry_habituations.plot_integral_at_latency_bars(
            mtgs, (flatui[i], flatui[i])
        )

        for ax in fig.axes:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_ylim([0, 1.2])
            plt.sca(ax)
            plt.subplots_adjust(bottom=0.35, left=0.3, right=0.8)
        ax.spines["left"].set_visible(False)
        ax.get_yaxis().set_visible(False)
    return fig

def analyse_signal_by_escape_latency():
    groups = [pre_test_24hr + pre_test_sameday]
    labels = ['pre_test_snl_all']

    for group, label in zip(groups, labels):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]

        for mtg in mtgs:

            normalising_factor = get_normalisation_factor(mtg, 220)
            for t, c in zip(mtg.all_trials[:3], ['k', 'b', 'g']):
                lat=int(t.estimate_latency(False))
                val = t.integral_escape_metric(220) / normalising_factor

                plt.scatter(lat, val, color=c, s=100)
        t.plot_stimulus()


def get_normalisation_factor(mtg, timepoint=215, escape_only=False):

    normalising_factor = max(
        [
            np.nanmax(
                [
                    t.integral_escape_metric(int(timepoint))
                    for t in mtg.loom_trials()[:30]
                ]
            )
        ])
    return normalising_factor


def get_signal_df(groups, timepoint=300):
    all_df = pd.DataFrame()
    for group in groups:
        mtgs = experimental_log.get_mtgs_in_experiment(group)
        mtgs = [m for m in mtgs if m.mouse_id not in ["074744"]]

        for mtg in mtgs:
            mtg_dict = {}
            vals = []
            escapes = []
            contrasts = []
            trials= mtg.loom_trials() #[:16]
            if timepoint is None:
                timepoint = np.mean([t.latency_peak_detect() for t in mtg.pre_test_trials()[:3]])
            normalising_factor = get_normalisation_factor(mtg, timepoint)

            for t in trials:
                val = (
                        t.integral_escape_metric(int(timepoint))
                        / normalising_factor
                )
                vals.append(val)
                escapes.append(t.is_flee())
                contrasts.append(t.contrast)

            for metric in mtg.analysed_metrics():
                metric_vals = []
                for t in trials:
                    metric_vals.append(t.metric_functions[metric]())
                mtg_dict.setdefault(metric, metric_vals)

            mtg_dict.setdefault("group", [group] * len(trials))
            mtg_dict.setdefault("deltaf metric", vals)
            mtg_dict.setdefault("contrast", contrasts)
            mtg_dict.setdefault("escape", escapes)
            mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(trials))
            mtg_df = pd.DataFrame.from_dict(mtg_dict)
            all_df = all_df.append(mtg_df)
    return all_df


def significance_at_each_tp(group_label):
    ps_escape=[]
    ps_ctst=[]
    for tp in range(210,300,15):
        groups = [group_label]
        df = get_signal_df(groups, timepoint=tp)
        #result=ancova(df, dv='deltaf metric', covar='contrast', between='escape')
        result=partial_corr(df, y='deltaf metric', x='speed', covar='contrast', method='spearman')

        ps_escape.append(result['p-val'][0])
        #ps_ctst.append(result['p-val'][1])
    return ps_escape, ps_ctst


def partial_correlation_all(group_label, timepoint=220):
    df = get_signal_df([group_label], timepoint=timepoint)
    stats_summary = pd.DataFrame()
    for metric in df.keys():
        if metric not in ['time of loom', 'loom number', 'deltaf metric', 'contrast', 'escape', 'mouse id', 'group']:
            result = partial_corr(df, y='deltaf metric', x=metric, covar='contrast', method='spearman')
            result['metric'] = metric
            stats_summary = stats_summary.append(result, ignore_index=True)
    return df, stats_summary


def get_signal_df_pre_post(groups, timepoint = 300, trial_type='variable', escapes_only=False):
    all_df = pd.DataFrame()
    for group in groups:
        mtgs = experimental_log.get_mtgs_in_experiment(group)
        mtgs = [m for m in mtgs if m.mouse_id not in ["074744"]]

        for mtg in mtgs:
            mtg_dict = {}
            vals = []
            escapes = []
            contrasts = []
            if trial_type == 'variable':
                trials = mtg.loom_trials()[:19]
                if escapes_only:
                    trials = [t for t in trials if t.is_flee()]
                trial_types = ['variable']*len(trials)
            else:
                trials = mtg.pre_test_trials()[:3] + mtg.post_test_trials()[:3] + mtg.auditory_lsie_trials()

                latency = np.nanmean([t.latency_peak_detect() for t in mtg.pre_test_trials()[:3]])
                #timepoint = 230
                if np.isnan(latency):
                    continue
                timepoint=latency
                trial_types = ['pre_test'] * len(mtg.pre_test_trials()[:3]) + ['post_test']*len(mtg.post_test_trials()[:3]) + ['auditory']*len(mtg.auditory_lsie_trials())

            print(timepoint)
            normalising_factor = get_normalisation_factor(mtg, timepoint)
            for t in trials:
                if escapes_only:
                    timepoint = t.latency_peak_detect()
                val = (
                        t.integral_escape_metric(int(timepoint))
                        / normalising_factor
                )
                vals.append(val)
                escapes.append(t.is_flee())
                contrasts.append(t.contrast)
            for metric in mtg.analysed_metrics():
                metric_vals = []
                for t in trials:
                    metric_vals.append(t.metric_functions[metric]())
                mtg_dict.setdefault(metric, metric_vals)
            mtg_dict.setdefault("group", [group] * len(trials))
            mtg_dict.setdefault("deltaf metric", vals)
            mtg_dict.setdefault("contrast", contrasts)
            mtg_dict.setdefault("escape", escapes)
            mtg_dict.setdefault("trial type", trial_types)
            mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(trials))
            mtg_dict.setdefault("experimental group", [group] * len(trials))

            mtg_df = pd.DataFrame.from_dict(mtg_dict)
            all_df = all_df.append(mtg_df)
    return all_df


def get_normalised_signals_mtg(mtg, n_samples_before=10):
    trials= mtg.loom_trials()[:18]
    vals=[]
    escapes = []
    contrasts = []
    trials_subset = []
    for t in trials:
        if t.contrast ==0 and t.is_flee():
            escape_latency = int(t.latency_peak_detect())
            s, e = escape_latency-n_samples_before, escape_latency
            val = np.mean(t.delta_f()[s:e])
            vals.append(val)
            escapes.append(t.is_flee())
            contrasts.append(t.contrast)
            trials_subset.append(t)
    vals = np.array(vals) / np.nanmax(vals)

    return vals, escapes, contrasts, trials_subset


def analyse_pre_trials_latency(mids):
    mtgs = [loom_trial_group.MouseLoomTrialGroup(m) for m in mids if m not in ["074744"]]
    trials = [mtg.pre_test_trials()[:3] for mtg in mtgs]
    shortest_latency_trial = trials[np.argmin([t.latency_peak_detect() for t in trials])]
    threshold = shortest_latency_trial.integral_downsampled()[shortest_latency_trial.latency_peak_detect()]


def get_signal_df_pre_post(groups, timepoint = 300, trial_type='variable', escapes_only=False):
    all_df = pd.DataFrame()
    for group in groups:
        mtgs = experimental_log.get_mtgs_in_experiment(group)
        mtgs = [m for m in mtgs if m.mouse_id not in ["074744"]]

        for mtg in mtgs:
            mtg_dict = {}
            vals, escapes, contrasts, trials = get_normalised_signals_mtg(mtg, 10)

            for metric in mtg.analysed_metrics():
                metric_vals = []
                for t in trials:
                    metric_vals.append(t.metric_functions[metric]())
                mtg_dict.setdefault(metric, metric_vals)
            mtg_dict.setdefault("group", [group] * len(trials))
            mtg_dict.setdefault("deltaf metric", vals)
            mtg_dict.setdefault("contrast", contrasts)
            mtg_dict.setdefault("escape", escapes)
            mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(trials))
            mtg_dict.setdefault("experimental group", [group] * len(trials))

            mtg_df = pd.DataFrame.from_dict(mtg_dict)
            all_df = all_df.append(mtg_df)
    return all_df


def get_mouse_signal_df(mtg, group):
    mtg_dict = {}
    vals, escapes, contrasts, trials = get_normalised_signals_mtg(mtg, 10)
    for metric in mtg.analysed_metrics():
        metric_vals = []
        for t in trials:
            metric_vals.append(t.metric_functions[metric]())
        mtg_dict.setdefault(metric, metric_vals)
    mtg_dict.setdefault("group", [group] * len(trials))
    mtg_dict.setdefault("deltaf metric", vals)
    mtg_dict.setdefault("contrast", contrasts)
    mtg_dict.setdefault("escape", escapes)
    mtg_dict.setdefault("mouse id", [mtg.mouse_id] * len(trials))
    mtg_dict.setdefault("experimental group", [group] * len(trials))
    mtg_df = pd.DataFrame.from_dict(mtg_dict)
    return mtg_df


def rescale_lines(ax1, ax2):
    val1 = ax1.lines[0].get_data()[1][0]
    val2 = ax2.lines[0].get_data()[1][0]
    scaling_factor = val1/val2
    plt.plot(ax2.lines[0].get_data()[1] * scaling_factor, linewidth=3)
    sns.scatterplot(ax2.lines[0].get_data()[0], ax2.lines[0].get_data()[1] * scaling_factor, s=150)
    for i, line in enumerate(ax2.lines[1:8]):
        plt.plot(line.get_data()[0],
                 line.get_data()[1] + (scaling_factor * ax2.lines[0].get_data()[1][i]) - ax2.lines[0].get_data()[1][i],
                 color='b', linewidth=3)


def get_peak_locations(trials):
    locs = []
    for t in trials:
        val, loc = np.argmax(t.delta_f()[:600])
        locs.append(loc)
    return locs


def compare_peaks(mtgs):
    LOOM_ONSETS = [200, 228, 256, 284, 312]
    all_vals = {1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                }
    LOOM_OFFSETS = [x+14 for x in LOOM_ONSETS]
    for mtg in mtgs:
        for t in mtg.loom_trials()[:3]:
            peaks = [np.max(t.delta_f()[s:e]/np.max(t.delta_f()[200:350])) for (s, e) in zip(LOOM_ONSETS, LOOM_OFFSETS)]
            for i, p in enumerate(peaks):
                all_vals[i+1].append(p)
                print(i)
    return all_vals


def compute_stats_peaks(groups=('photometry_habituation_tre-GCaMP_24hr_pre', 'photometry_habituation_tre-GCaMP_same_day_pre')):
    from looming_spots.db import loom_trial_group, experimental_log
    from looming_spots.thesis_figure_plots import photometry_example_traces
    mids = []
    for group in groups:
        mids.extend(experimental_log.get_mouse_ids_in_experiment(group))

    mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
    a = photometry_example_traces.compare_peaks(mtgs)
    keys = []
    vals = []
    new_dict = {}
    for k, v in a.items():
        keys.extend([k] * len(v))
        vals.extend(v)
    new_dict['values'] = vals
    new_dict['loom number'] = keys
    df =pd.DataFrame.from_dict(new_dict)
    return df