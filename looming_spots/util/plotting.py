import matplotlib.colors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, patches as patches

from matplotlib.collections import LineCollection
from looming_spots.constants import LOOM_ONSETS, SHELTER_FRONT


def plot_shelter_location(fig, context):
    for ax in fig.axes:
        plt.sca(ax)
        house_front = SHELTER_FRONT
        plt.axhline(house_front, 0, 400, ls="--")


def plot_looms_ax(
    ax=None,
    vertical=True,
    height=1.3,
    loom_n_samples=14,
    relative=False,
    upsample_factor=1,
):
    if ax is None:
        ax = plt.gca()
    loom_onsets = LOOM_ONSETS
    if relative:
        loom_onsets = [x - 200 for x in loom_onsets]
    looms = [
        create_loom_patch(
            stim,
            vertical=vertical,
            height=height,
            loom_n_samples=loom_n_samples,
            upsample_factor=upsample_factor,
        )
        for stim in loom_onsets
    ]
    for loom in looms:
        ax.add_patch(loom)


def plot_auditory_stimulus(n_samples_before, n_samples=90):
    patch = patches.Rectangle(
        (n_samples_before, -0.2), n_samples, 1.3, alpha=0.1, color="r"
    )
    ax.add_patch(patch)


def create_loom_patch(
    start,
    upsample_factor=1,
    vertical=True,
    height=1.3,
    loom_n_samples=14,
    y=-0.2,
):
    width = loom_n_samples * upsample_factor
    x = start * upsample_factor
    if not vertical:
        width, height = height, width
        x, y = y, x
    return patches.Rectangle(
        (x, y),
        width,
        height,
        alpha=0.1,
        color="k",
        linewidth=0,
    )


def convert_x_axis(n_samples, n_steps, frame_rate):
    plt.xticks(
        np.linspace(0, n_samples - 1, n_steps),
        np.linspace(0, n_samples / frame_rate, n_steps),
    )


def convert_y_axis(old_min, old_max, new_min, new_max, n_steps):
    plt.yticks(
        np.linspace(old_min, old_max, n_steps),
        np.linspace(new_min, new_max, n_steps),
    )


def get_x_length(ax=None):
    if ax is None:
        ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    return len(xdata)
