import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        self.pre_trials, self.post_trials = self.get_pre_and_post_test_trials()

    @property
    def habituation_session(self):  # TODO: remove all session references
        return self.sessions[self.habituation_idx]

    @property
    def habituation_type(self):  # TODO: phase this out obsolete

        habituation_only = np.array(['habituation_only' in p for p in self.protocols])
        habituation_and_test = np.array(['habituation_and_test' in p for p in self.protocols])
        habituation = np.array(['habituation' in p for p in self.protocols])

        if not any(habituation):
            return None
        elif all([any(habituation_only), any(habituation_and_test)]):
            return 'multiple habituations of different types'
        elif any(habituation_and_test):
            return 'habituation_and_test'
        elif any(habituation_only):
            return 'habituation_only'

    @property
    def habituation_idx(self):
        print(self.protocols)
        habituations = np.array(['habituation' in p for p in self.protocols])
        return np.where(habituations)[0][0]

    @property
    def pre_tests(self):  # FIXME: pre test in habituation recording
        if self.habituation_type is None:
            return self.sessions
        return self.sessions[0:self.habituation_idx]

    @property
    def post_tests(self):
        if self.habituation_idx is None:
            print('no habituation {}'.format(self.mouse_id))
            return self.sessions
        elif self.habituation_type == 'habituation_and_test':
            return self.sessions[self.habituation_idx:]
        else:
            return self.sessions[self.habituation_idx + 1:]

    def nth_pre_test(self, n):
        return self.pre_tests[n]

    def nth_post_test(self, n):
        return self.post_tests[n]

    def _underwent_habituation(self):
        if any(s.contains_habituation for s in self.sessions):
            return True

    def get_pre_and_post_test_trials(self):
        habituation = False
        pre_trials = []
        post_trials = []

        for s in self.sessions:
            if s.contains_habituation:

                for t in s.trials:
                    if t.sample_number < s.habituation_protocol_start:
                        pre_trials.append(t)
                    elif t.sample_number > s.habituation_loom_idx[-1]:
                        post_trials.append(t)
                    if t.sample_number == s.habituation_loom_idx[-1]:
                        habituation = True
                continue

            for t in s.trials:
                if not habituation:
                    pre_trials.append(t)
                else:
                    post_trials.append(t)
        return pre_trials, post_trials


class SessionGroup(object):
    """this class if for experimentally similar groups of sessions. a single session group can be considered
    to be a group of sessions that belong in the same plot together"""

    def __init__(self, sessions=None, group_key=None, fig=None):
        self.sessions = sessions
        self.group_key = group_key
        self.fig = fig
        self.n_trials = sum([s.n_test_trials for s in self.sessions])
        self.n_flees = sum([s.n_flees for s in self.sessions])
        self.title = '{} (n = {} flees / {} trials, {} mice)'.format(self.group_key, self.n_flees,
                                                                     self.n_trials, len(self.sessions))

    @property
    def injection_site_image(self):
        for s in self.sessions:
            if s.histology:
                return s.histology
        return False

    @property
    def heatmap_data(self, type='test'):
        all_speeds = []

        for s in self.sessions:

            for t in s.get_trials(type):
                try:
                    all_speeds.append(t.smoothed_x_speed)
                except session.LoomsNotTrackedError:
                    continue
        return all_speeds

    @property
    def figure(self):
        return plt.figure()

    def plot_all_sessions(self, ax=None):  # TODO: pass functions around instead of copying code
        plt.figure(figsize=(8, 3))
        if ax is None:
            ax = plt.subplot(111)

        for s in self.sessions:
            try:
                s.plot_trials()
            except session.LoomsNotTrackedError as err:
                print(err)
                continue

        plotting.plot_looms_ax(ax)
        plt.title(self.title)
        plt.ylabel('x position in box (cm)')
        plt.xlabel('time (s)')
        track_length = get_x_length(ax)
        self.convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        self.convert_x_axis(track_length, n_steps=11)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

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
                    all_accs.append(t.smoothed_x_acceleration)
                except session.LoomsNotTrackedError:
                    continue
        plt.imshow(all_accs, cmap='Greys', aspect='auto', vmin=-0.0075, vmax=0.0075)
        plt.title(self.title)
        generic_functions.neaten_plots(fig.axes)


def get_x_length(ax):
    line = ax.lines[0]
    xdata = line.get_xdata()
    return len(xdata)

