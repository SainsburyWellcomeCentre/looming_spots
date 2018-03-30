import os
import numpy as np
import matplotlib.pyplot as plt

from looming_spots.analysis import tracks


class Trial(object):
    def __init__(self, session, path):
        self.path = path
        self.context = session.context
        self.session = session
        self.ref = self.session.reference_frame

    @property
    def loom_idx(self):
        fname = os.path.split(self.path)[-1]
        return int(''.join([char for char in fname if char.isdigit()]))

    @property
    def raw_track(self):
        return tracks.load_raw_track(self.path)

    @property
    def normalised_x_track(self):
        return tracks.load_normalised_track(self.path, self.context)

    def normalised_speed(self):
        return np.diff(self.normalised_x_track)

    def is_flee(self):
        return tracks.classify_flee(self.path, self.context)

    def plot_track_on_image(self):
        x_track, y_track = self.raw_track
        plt.imshow(self.ref)
        plt.plot(x_track, y_track, color='b')

    def __gt__(self, other):
        return self.loom_idx > other.loom_idx
