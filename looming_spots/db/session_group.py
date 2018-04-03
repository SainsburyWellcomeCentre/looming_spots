import matplotlib.pyplot as plt
import numpy as np

from looming_spots.analysis import plotting
from looming_spots.db import session
from looming_spots.util import generic_functions
from looming_spots.db import load


ARENA_SIZE_CM = 50
FRAME_RATE = 30


class MouseSessionGroup(object):
    """this class is for selecting the right session from a mouse that has had multiple sessions"""

    def __init__(self, mouse_id):
        self.mouse_id = mouse_id
        self.sessions = np.array(load.load_sessions(mouse_id))
        self.protocols = np.array([s.protocol for s in self.sessions])

    @property
    def habituation_session(self):
        return self.sessions[self.habituation_index]

    @property
    def habituation_index(self):
        is_habituation_only = np.array(['habituation_only' in p for p in self.protocols])
        is_habituation_and_test = np.array(['habituation_and_test' in p for p in self.protocols])
        if not any(is_habituation_only) and not any(is_habituation_and_test):
            print('no habituations detected')
            return None
        elif any(is_habituation_and_test):
            return np.where(is_habituation_and_test)[0][0] - 1
        else:
            print('habituation detected {}'.format(np.where(is_habituation_only)[0][0]))
            return np.where(is_habituation_only)[0][0]

    @property
    def pre_tests(self):
        return self.sessions[0:self.habituation_index]

    @property
    def post_tests(self):
        if self.habituation_index is None:
            print('no habituation {}'.format(self.mouse_id))
            return self.sessions
        return self.sessions[self.habituation_index+1:]

    def nth_pre_test(self, n):
        return self.pre_tests[n]

    def nth_post_test(self, n):
        return self.post_tests[n]


class SessionGroup(object):
    """this class if for experimentally similar groups of sessions. a single session group can be considered
    to be a group of sessions that belong in the same plot together"""

    def __init__(self, sessions=None, group_key=None, fig=None):
        self.sessions = sessions
        self.group_key = group_key
        self.fig = fig
        self.n_trials = sum([len(s.trials) for s in self.sessions])
        self.title = '{} (n = {} trials, {} mice)'.format(self.group_key, self.n_trials, len(self.sessions))

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

