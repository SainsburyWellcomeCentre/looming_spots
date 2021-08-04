import os

import looming_spots.io.load
import numpy as np
import matplotlib.pyplot as plt

from looming_spots.io.photodiode import filter_pd
from looming_spots.io.load import load_pd_on_clock_ups


def get_calibration_curve(pd_directory):
    pd = looming_spots.io.load.load_pd_on_clock_ups(pd_directory)
    starts, ends = get_calibration_starts_ends(pd_directory)
    pd_vals = []
    for start, end in zip(starts, ends):
        pd_val = np.median(pd[start:end])
        pd_vals.append(pd_val)

    return pd, starts, ends, pd_vals


def get_calibration_starts_ends(directory):
    pd = load_pd_on_clock_ups(directory)
    starts, ends = find_pd_calibration_crossings(pd, 0.9)
    return starts, ends


def find_pd_calibration_crossings(ai, threshold=0.4):

    filtered_pd = filter_pd(ai)

    print("threshold: {}".format(threshold))
    loom_on = (filtered_pd < threshold).astype(int)
    loom_ups = np.diff(loom_on) == 1
    loom_starts = np.where(loom_ups)[0]
    loom_downs = np.diff(loom_on) == -1
    loom_ends = np.where(loom_downs)[0]
    return loom_starts, loom_ends


psychopy_directory = "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/raw_data/calibration_psychtoolbox_-1_1_contrast_steps/20190613_09_09_04/"
psychtoolbox_directory = "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/raw_data/calibration_psychtoolbox_-1_1_contrast_steps/20190613_11_02_06/"

fig, axes = plt.subplots(2, 1)
for i, directory in enumerate([psychopy_directory, psychtoolbox_directory]):

    pd, starts, ends, pd_vals = get_calibration_curve(directory)

    axes[0].plot(pd)
    contrasts = np.load(os.path.join(directory, "steps.npy"))
    if i == 1:
        contrasts -= 1
    axes[0].plot(starts[: len(contrasts)], contrasts, "o")
    axes[1].plot(pd_vals[: len(contrasts)], contrasts, "o")


plt.show()
