import os

import matplotlib.pyplot as plt
import numpy as np
from configobj import ConfigObj

from looming_spots.analysis import tracks
from looming_spots.util.generic_functions import chunks

N_MIN = 10
FRAME_RATE = 30
N_SAMPLES_TO_ANALYSE = N_MIN*60*FRAME_RATE

root = '/home/slenzi/spine_shares/loomer/processed_data/CA178_1/20180318_19_17_58/'
mtd_path = os.path.join(root, 'metadata.cfg')
config = ConfigObj(mtd_path, unrepr=False)


def plot_binned_preference(path, start, grid_location, n_bins=10):

    normalised_track = tracks.load_normalised_track(path, context='split')

    if start is None:
        start = int(config['track_start'])

    end = start + N_SAMPLES_TO_ANALYSE
    track_sample = normalised_track[start:end]
    print('length_of_track to consider= {}'.format(len(track_sample)))
    binned_preference = []

    for i, chunk in enumerate(chunks(track_sample, int(N_SAMPLES_TO_ANALYSE/n_bins))):
        if np.count_nonzero(np.isnan(chunk)) > len(chunk)/2:
            binned_preference.append(np.nan)
            continue

        left_pref = np.count_nonzero(chunk > 0.5)
        right_pref = np.count_nonzero(chunk < 0.5)

        if grid_location == 'left':
            grid_pref = left_pref
        elif grid_location == 'right':
            grid_pref = right_pref

        total_tracked_frames = len(chunk) - np.count_nonzero(np.isnan(chunk))
        binned_preference.append(grid_pref/total_tracked_frames)
    return binned_preference


def plot_sides(path, start=None):
    if start is None:
        start = int(config['track_start'])
    end = start + N_SAMPLES_TO_ANALYSE
    track = tracks.load_normalised_track(path + '/camera', context='split')[start:end]
    x, y = tracks.load_raw_track(path + '/camera')
    x = x[start:end]
    y = y[start:end]
    above = track > 0.5
    below = track < 0.5

    where_above = np.where(above)[0]
    where_below = np.where(below)[0]

    ref_path = os.path.join(path, 'ref.npy')
    ref = np.load(ref_path)
    plt.imshow(ref, cmap='Greys')
    plt.plot(x[where_above], y[where_above], 'o', color='k')
    plt.plot(x[where_below], y[where_below], 'o', color='r')
