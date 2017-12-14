import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.misc
STIMULUS_ONSETS = [197, 225, 253, 281, 309] # FIXME: use the filtfilt


def plot_tracks(directory, fig, color='k'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            track, _ = load_track(loom_folder)
            plt.plot(track, color=color)
    return fig


def plot_track_and_loom_position(directory, fig, color='k'):
    for name in os.listdir(directory):
        loom_folder = os.path.join(directory, name)
        if os.path.isdir(loom_folder):
            color='r'
            x, y = load_track(loom_folder)
            # if x[200] < 100 or y[200] < 200:
            #     print('TRACK ERROR. SKIPPING...')
            #     color = 'g'
            #     continue
            img = np.load(os.path.join(directory, 'ref.npy'))
            plt.imshow(img)
            plt.plot(x[200], y[200], 'o', markersize=10, color=color)
            plt.plot(x[170:], y[170:], color='k')
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
    y_pos = np.array(df['centre_x'])
    return x_pos, y_pos


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
            position_x, _ = load_track(loom_folder)
            results.append(classify_flee(position_x, distances))
            n_flees += 1
    return np.count_nonzero(results)/n_flees


def loom_patch(start):
    return patches.Rectangle((start, 0), 14, 800, alpha=0.1, color='k')


def classify_flee(track, speed):
    if any([x > 15 for x in speed]) and ((track[300] - track[200]) > 150):
        print(True)
        return True
    print(False)


def plot_all(directory, fig):
    for name in os.listdir(directory):
        if os.path.isdir(loom_folder):
            loom_folder = os.path.join(directory, name)
            track, _ = load_track(loom_folder)
            distances = load_distances(loom_folder)
            if classify_flee(track, distances):
                color = 'r'
            else:
                color = 'k'
            plt.plot(track, color=color)
    return fig


# fig=plt.figure()
# for m in all_m:
#     first_session = min(m.sessions)
#     plot_distances(first_session.path, fig)
#     for name in os.listdir(first_session.path):
#         loom_folder = os.path.join(first_session.path, name)
#         if os.path.isdir(loom_folder):
#             distances = load_distances(loom_folder)
#             peak = max(distances[150:300])
#             arg_peak = np.argmax(distances[150:300])
#             plt.plot(arg_peak+150,peak, 'o')