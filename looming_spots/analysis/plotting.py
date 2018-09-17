from matplotlib import pyplot as plt, patches as patches

from looming_spots.db.constants import STIMULUS_ONSETS, NORM_FRONT_OF_HOUSE_B, NORM_FRONT_OF_HOUSE_A

def plot_looms(fig):
    for ax in fig.axes:
        for loom in [create_loom_patch(stim) for stim in STIMULUS_ONSETS]:
            ax.add_patch(loom)
    return fig


def plot_home(context):
    ax = plt.gca()
    home_front = NORM_FRONT_OF_HOUSE_B if context == 'B' else NORM_FRONT_OF_HOUSE_A
    ax.hlines(home_front, 0, 400, linestyles='dashed')


def plot_looms_ax(ax):
    looms = [create_loom_patch(stim) for stim in STIMULUS_ONSETS]
    for loom in looms:
        ax.add_patch(loom)


def create_loom_patch(start):
    return patches.Rectangle((start, -0.2), 14, 1.3, alpha=0.1, color='k')

