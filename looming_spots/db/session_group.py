import matplotlib.pyplot as plt
import numpy as np

from looming_spots.analysis import plotting
from looming_spots.db import session

ARENA_SIZE_CM = 50
FRAME_RATE = 30


class SessionGroup(object):
    def __init__(self, sessions=None, group_key=None):
        self.sessions = sessions
        self.group_key = group_key

    def plot_all_sessions(self, fig=None):
        if fig is None:
            fig = plt.figure()

        for s in self.sessions:
            try:
                s.plot_trials()
            except session.LoomsNotTrackedError:
                continue

        plotting.plot_looms(fig)
        plt.title(self.group_key)
        plt.ylabel('x position in box (cm)')  # TODO: plot actual position in cm
        plt.xlabel('frame number')  # TODO: plot time in seconds
        plt.yticks(np.linspace(0, 1, 6), np.linspace(0, 50, 6))
        track_length = get_x_length(fig)
        plt.xticks(np.linspace(0, track_length-1, 11), np.linspace(0, (track_length-1)/FRAME_RATE, 11))


def get_x_length(fig):
    line = fig.axes[0].lines[0]
    xdata = line.get_xdata()
    print(len(xdata))
    return len(xdata)
