import matplotlib.pyplot as plt
import numpy as np

from looming_spots.analysis import plotting
from looming_spots.db import session
from looming_spots.util import generic_functions

ARENA_SIZE_CM = 50
FRAME_RATE = 30


class SessionGroup(object):
    def __init__(self, sessions=None, group_key=None, fig=None):
        self.sessions = sessions
        self.group_key = group_key
        self.fig = fig
        self.title = '{} (n = {} mice)'.format(self.group_key, len(self.sessions))

    @property
    def heatmap_data(self):
        all_speeds = []

        for s in self.sessions:
            for t in s.trials:
                try:
                    all_speeds.append(t.smoothed_speed)
                except session.LoomsNotTrackedError:
                    continue
        return all_speeds

    @property
    def figure(self):
        return plt.figure()

    def plot_all_sessions(self, fig=None):  # TODO: pass functions around instead of copying code
        if fig is None:
            fig = plt.figure()

        for s in self.sessions:
            try:
                s.plot_trials()
            except session.LoomsNotTrackedError:
                continue

        plotting.plot_looms(fig)
        plt.title(self.title)
        plt.ylabel('x position in box (cm)')
        plt.xlabel('time (s)')
        track_length = get_x_length(fig)
        self.convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        self.convert_x_axis(track_length, n_steps=11)
        generic_functions.neaten_plots(fig.axes)

    def convert_x_axis(self, track_length, n_steps):
        plt.xticks(np.linspace(0, track_length - 1, n_steps), np.linspace(0, (track_length - 1) / FRAME_RATE, n_steps))

    @staticmethod
    def convert_y_axis(old_min, old_max, new_min, new_max, n_steps):
        plt.yticks(np.linspace(old_min, old_max, n_steps), np.linspace(new_min, new_max, n_steps))

    def plot_all_sessions_heatmaps(self, fig=None):

        if fig is None:
            fig = plt.figure()

        plt.imshow(self.heatmap_data, cmap='Greys', aspect='auto', vmin=-0.03, vmax=0.03)
        self.convert_x_axis(track_length=len(self.heatmap_data[0])+1, n_steps=11)
        plt.title(self.title)
        plt.ylabel('trial')
        plt.xlabel('time (s)')
        generic_functions.neaten_plots(fig.axes)

    def plot_acc_heatmaps(self, fig=None):

        if fig is None:
            fig = plt.figure()

        all_accs = []

        for s in self.sessions:
            for t in s.trials:
                try:
                    all_accs.append(t.smoothed_acceleration)
                except session.LoomsNotTrackedError:
                    continue
        plt.imshow(all_accs, cmap='Greys', aspect='auto', vmin=-0.0075, vmax=0.0075)
        plt.title(self.title)
        generic_functions.neaten_plots(fig.axes)


def get_x_length(fig):
    line = fig.axes[0].lines[0]
    xdata = line.get_xdata()
    return len(xdata)

