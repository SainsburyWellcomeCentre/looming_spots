import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt, patches as patches

from matplotlib.collections import LineCollection

import looming_spots.preprocess.normalisation
from looming_spots.db.constants import STIMULUS_ONSETS


def plot_looms(fig):
    for ax in fig.axes:
        for loom in [create_loom_patch(stim) for stim in STIMULUS_ONSETS]:
            ax.add_patch(loom)
    return fig


def plot_looms_upsampled(fig):
    for ax in fig.axes:
        for loom in [
            create_loom_patch(
                stim * 10000 / 30, upsample_factor=int(10000 / 30)
            )
            for stim in STIMULUS_ONSETS
        ]:
            ax.add_patch(loom)
    return fig


def plot_upsampled_looms_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    looms = [
        create_loom_patch(stim * 10000 / 30, upsample_factor=int(10000 / 30))
        for stim in STIMULUS_ONSETS
    ]
    for loom in looms:
        ax.add_patch(loom)


def plot_stimulus(fig, onset=200, n_samples=90):
    for ax in fig.axes:
        patch = patches.Rectangle(
            (onset, -0.2), n_samples, 1.3, alpha=0.1, color="b", linewidth=0
        )
        ax.add_patch(patch)
    return fig


def plot_home(fig, context):
    for ax in fig.axes:
        plt.sca(ax)
        house_front = looming_spots.preprocess.normalisation.normalised_home_front(
            context
        )
        plt.axhline(house_front, 0, 400, ls="--")


def plot_looms_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    looms = [create_loom_patch(stim) for stim in STIMULUS_ONSETS]
    for loom in looms:
        ax.add_patch(loom)


def create_loom_patch(start, upsample_factor=1):
    return patches.Rectangle(
        (start, -0.2),
        int(14 * upsample_factor),
        1.3,
        alpha=0.1,
        color="k",
        linewidth=0,
    )


def plot_line_with_color_variable(
    x, y, color_variable_array, start=None, normalising_factor=None
):

    """
    modified from: https://stackoverflow.com/questions/10252412/matplotlib-varying-color-of-line-to-capture-natural-time-parameterization-in-da

    meant for plotting speeds on tracks

    :param start:
    :param normalising_factor:
    :param x:
    :param y:
    :param color_variable_array:
    :return:
    """

    points = np.array([x, y]).transpose().reshape(-1, 1, 2)
    color_variable_array = np.array(
        [
            val
            for (p, val) in zip(points, color_variable_array)
            if not np.isnan(p).any()
        ]
    )
    points = np.array([p for p in points if not np.isnan(p).any()])

    if start is None:
        norm_factor = 1
    else:
        norm_factor = max(color_variable_array)
        points = points[start:]
        color_variable_array = color_variable_array[start:]

    points = np.array(
        [
            p
            for (p, val) in zip(points, color_variable_array)
            if not np.isnan(val)
        ]
    )
    color_variable_array = np.array(
        [val for val in color_variable_array if not np.isnan(val)]
    )

    if normalising_factor is not None:
        norm_factor = normalising_factor

    color_variable_array /= norm_factor

    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segs,
        cmap=plt.get_cmap("inferno"),
        norm=matplotlib.colors.Normalize(vmin=0, vmax=0.8),
    )
    lc.set_array(color_variable_array)

    plt.gca().add_collection(lc)

    plt.plot(color_variable_array)


def plot_trials_separate_plots(trials):
    fig, axes = plt.subplots(len(trials))
    for t, ax in zip(trials, axes):
        plt.axes(ax)
        t.plot_track()


def format_plots(axes):
    plt.subplots_adjust(wspace=1, hspace=1)
    for i, ax in enumerate(axes):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


def format_ax(xlabel, ylabel, ax=None):
    if ax is None:
        ax = plt.gca()

    plt.ylabel(
        "escape probability", fontsize=15, fontweight="black", color="#333F4B"
    )
    plt.xlabel("contrast", fontsize=15, fontweight="black", color="#333F4B")

    ax.spines["left"].set_smart_bounds(True)
    ax.spines["bottom"].set_smart_bounds(True)

    # set the spines position
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position(("axes", -0.04))
    ax.spines["left"].set_position(("axes", 0.015))
