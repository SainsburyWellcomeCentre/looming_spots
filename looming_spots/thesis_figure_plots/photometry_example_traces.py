import numpy as np
from looming_spots.db.constants import FIGURE_DIRECTORY
from looming_spots.db import loom_trial_group, experimental_log
from looming_spots.track_analysis import photometry_habituations
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

pre_test_24hr = ["074743", "074744", "074746", "895773", "953828", "953829"]
pre_test_sameday = ["898989", "898990"]
contrast_curves = ["898992", "916063", "921000", "907822"]

any_escape = ["074743", "895773", "953828", "898989", "898990"]
no_escape = ["074746", "957073"]


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


def plot_pre_post_photometry_lsie(mtg):

    fig1 = plt.figure()
    ax1 = plt.subplot(211)
    avg_df = []

    for t in mtg.pre_test_trials()[:3]:
        t.plot_track()
    mtg.pre_test_trials()[0].plot_stimulus()
    ax2 = plt.subplot(212)
    for t in mtg.pre_test_trials()[:3]:
        plt.plot(t.delta_f())
        avg_df.append(t.delta_f())
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
        avg_df.append(t.delta_f())
        plt.plot(t.delta_f())
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


def plot_LSIE_evoked_signals(
    groups=(pre_test_sameday, pre_test_24hr),
    colors=("b", "k"),
    labels=("same day pre-test (n={})", "24 hr pre-test n={}"),
):
    plt.figure()
    for group, color in zip(groups, colors):
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in group]

        photometry_habituations.plot_habituation_curve_with_sem(mtgs, color)
    plt.legend(
        [labels[0].format(len(groups[0])), labels[1].format(len(groups[1]))]
    )
    plt.title("loom evoked DA signal in ToS during LSIE protocol")
    plt.xlabel("contrast (binned 3 trials each)")
    plt.ylabel("integral of dF/F at avg. pre-test escape latency")
    plt.show()


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
        mtgs_sup, ("Grey", "Grey")
    )
    plt.subplot(122)
    photometry_habituations.plot_integral_at_latency_bars(
        mtgs_non_sup, ("b", "b")
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
    )
):

    all_dfs = []
    mouse_ids = ["898992", "916063", "921000", "907822"]
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

    # fig = sns.lineplot('contrast', 'escape', data=df, err_style='bars')
    # sns.lineplot('contrast', 'ca signal', data=df, err_style='bars')
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


def plot_habituation_trialwise_with_lowess_fit():
    pre_test_24hr = ["074743", "074746", "895773", "953828", "953829"]
    pre_test_sameday = ["898989", "898990"]
    mtgs_24 = [
        loom_trial_group.MouseLoomTrialGroup(mid) for mid in (pre_test_24hr)
    ]
    mtgs_imm = [
        loom_trial_group.MouseLoomTrialGroup(mid) for mid in (pre_test_sameday)
    ]

    df = photometry_habituations.habituation_df(
        [mtgs_24, mtgs_imm], ["24 hr", "same day"]
    )
    a = sns.lmplot(
        "trial number",
        "ca signal",
        hue="test type",
        data=df[df["group label"] == "24 hr"],
        lowess=True,
    )
    b = sns.lmplot(
        "trial number",
        "ca signal",
        hue="test type",
        data=df[df["group label"] == "same day"],
        lowess=True,
    )
    c = sns.lmplot(
        "trial number",
        "ca signal",
        hue="group label",
        data=df[df["test type"] == "habituation"],
        fit_reg=False,
    )
    sns.lineplot(
        "trial number",
        "ca signal",
        hue="group label",
        data=df[df["test type"] == "habituation"],
        err_style="bars",
    )
    return df


def get_first_loom_response_by_contrast(contrast_curve_mids=contrast_curves):
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
            avg_df.append(t.delta_f()[200:230])
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
