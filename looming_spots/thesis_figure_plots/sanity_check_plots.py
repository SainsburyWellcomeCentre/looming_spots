import matplotlib
import pathlib

import pims

from looming_spots.db import loom_trial_group
import matplotlib.pyplot as plt
import numpy as np

mouse_ids = ["1012998", "1034952", "1034953", "1034954_b", "1034956"]


def plot_escape_and_delta_f_with_latencies():
    for mid in mouse_ids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        fig, axes = plt.subplots(3, 1)
        plt.title(mid)

        pre_latency = int(
            np.mean(
                [t.estimate_latency(False) for t in mtg.pre_test_trials()[:3]]
            )
        )
        for t in mtg.pre_test_trials()[:3]:
            axes[0].plot(t.normalised_x_track[:600])
            latency = t.estimate_latency(False)
            axes[0].plot(latency, t.normalised_x_track[latency], "o")

            axes[1].plot(t.delta_f()[:600])
            axes[1].plot(pre_latency, t.delta_f()[pre_latency], "o", color="r")
            axes[2].plot(t.integral_downsampled(), color="r")

        for t in mtg.post_test_trials()[:3]:
            color = "r" if t.classify_escape() else "k"
            axes[0].plot(t.normalised_x_track[:600], color=color)
            axes[1].plot(t.delta_f()[:600], color="k")
            axes[1].plot(pre_latency, t.delta_f()[pre_latency], "o", color="k")
            axes[2].plot(t.integral_downsampled(), color="k")
        plt.sca(axes[0])
        t.plot_stimulus()
        plt.sca(axes[1])
        t.plot_stimulus()


def plot_all_metrics_sanity(mid):
    mtg = loom_trial_group.MouseLoomTrialGroup(mid)
    for t in mtg.loom_trials()[:3]:
        plt.figure()
        plt.title(f"{mtg.mouse_id}_{t.loom_number}")
        t.plot_stimulus()
        rt = t.reaction_time() + 200
        speed, arg_peak_speed = t.peak_speed(True)
        latency = t.estimate_latency(True)
        ttrs = t.n_samples_to_reach_shelter()
        ttls = t.samples_to_leave_shelter()
        acc_arg = t.peak_x_acc_idx()
        x = t.normalised_x_track
        latency_pd = t.latency_peak_detect()
        plt.plot(x)
        plt.plot(-t.smoothed_x_acceleration * 200)
        plt.plot(-t.smoothed_x_speed * 20)
        plt.xlim([0, 600])
        plt.ylim([0, 1])
        for point, c in zip(
            [acc_arg, latency, arg_peak_speed, ttrs, ttls, rt, latency_pd],
            ["k", "r", "b", "y", "g", "c", "m"],
        ):
            if point is not None and not np.isnan(point):
                print(point)
                plt.plot(point, x[int(point)], "o", color=c)


def plot_trajectory_metrics_sanity(mid, test_type, start=195, end=350):
    matplotlib.rcParams["figure.figsize"] = [3.2, 8]
    mtg = loom_trial_group.MouseLoomTrialGroup(mid)
    trials = (
        mtg.pre_test_trials()[:3]
        if test_type == "pre_test"
        else mtg.post_test_trials()[:3]
    )
    for t in trials:
        plt.title(f"{mtg.mouse_id}_{t.loom_number}")
        x = t.smoothed_x_track[start:end]
        y = t.smoothed_y_track[start:end]
        color = "r" if t.classify_escape() else "k"
        # if x[0] < 0.65:
        plt.plot(y, x, color=color)
        # plt.plot(x[0], y[0], 'o', color='g')
        # plt.plot(x[200-start], y[200-start], 'o', color='k')
        plt.plot(y[-1], x[-1], "o", color="b")
        plt.xlim([-0.05, 0.35])
        plt.ylim([0, 1])
