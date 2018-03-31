import os
import numpy as np
import matplotlib.pyplot as plt

from looming_spots.analysis import tracks

LOOM_ONSET = 200
END_OF_CLASSIFICATION_WINDOW = 400


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

    @property
    def normalised_speed(self):
        return np.diff(self.normalised_x_track)

    def loc_peak_acceleration(self):
        acc = self.normalised_acceleration[LOOM_ONSET:END_OF_CLASSIFICATION_WINDOW]
        return np.where(acc == min(acc))[0] + LOOM_ONSET

    @property
    def normalised_acceleration(self):
        return np.diff(self.normalised_speed)

    def is_flee(self):
        return tracks.classify_flee(self.path, self.context)

    def plot_peak_acceleration(self):
        if self.is_flee():
            plt.plot(self.loc_peak_acceleration(), self.normalised_x_track[self.loc_peak_acceleration()], 'o', color='b')

    def plot_track_on_image(self):
        x_track, y_track = self.raw_track
        plt.imshow(self.ref)
        plt.plot(x_track, y_track, color='b')

    def __gt__(self, other):
        return self.loom_idx > other.loom_idx
