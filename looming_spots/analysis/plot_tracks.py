import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

STIMULUS_ONSETS = [200, 214, 228, 242, 256]


def plot_tracks(directory, fig, color='k'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            track = load_track(loom_folder)
            plt.plot(track, color=color)
    return fig


def plot_distances(directory, fig, color='b'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            distances = load_distances(loom_folder)
            plt.plot(distances, color=color)
    return fig


def plot_looms(fig):
    looms = [loom_patch(stim) for stim in STIMULUS_ONSETS]
    for loom in looms:
        fig.axes[0].add_patch(loom)
    return fig


def load_track(path, name='data.dat'):
    path = os.path.join(path, name)
    df = pd.read_csv(path, sep='\t')
    x_pos = np.array(df['frame_num'])  # FIXME: pyper saving issue?
    return x_pos


def load_distances(path, name='distances.dat'):
    path = os.path.join(path, name)
    print(path)
    df = pd.read_csv(path, sep='\t')
    return np.array(df)


def get_flee_rate(directory):
    n_flees = 0
    results = []
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            distances = load_distances(loom_folder)
            position_x = load_track(loom_folder)
            results.append(classify_flee(position_x, distances))
            n_flees += 1
    return np.count_nonzero(results)/n_flees


def loom_patch(start):
    return patches.Rectangle((start, 0), 7, 800, alpha=0.1, color='k')


def classify_flee(track, speed):
    if any([x > 15 for x in speed]) and ((track[300] - track[200]) > 150):
        print(True)
        return True
    print(False)


def plot_all(path, fig):
    track = load_track(path)
    distances = load_distances(path)
    if classify_flee(track, distances):
        color = 'r'
    else:
        color = 'k'
    plt.plot(track, color=color)
    return fig
