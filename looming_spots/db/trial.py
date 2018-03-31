import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from looming_spots.analysis import tracks

LOOM_ONSET = 200
END_OF_CLASSIFICATION_WINDOW = 400


class Trial(object):
    def __init__(self, session, path):
        self.path = path
        self.context = session.context
        self.session = session
        #self.reference_frame = self.session.reference_frame

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
    def smoothed_x_track(self):
        return gaussian_filter(self.normalised_x_track, 3)

    @property
    def smoothed_speed(self):
        return np.diff(self.smoothed_x_track)

    @property
    def normalised_speed(self):
        return np.diff(self.normalised_x_track)

    def loc_peak_acceleration(self):
        acc_window = self.smoothed_acceleration[LOOM_ONSET:END_OF_CLASSIFICATION_WINDOW]
        return np.where(acc_window == min(acc_window))[0] + LOOM_ONSET

    @property
    def smoothed_acceleration(self):
        return np.diff(self.smoothed_speed)

    def is_flee(self):
        return tracks.classify_flee(self.path, self.context)

    def plot_peak_acceleration(self):
        if self.is_flee():
            plt.plot(self.loc_peak_acceleration(), self.normalised_x_track[self.loc_peak_acceleration()], 'o', color='b')

    def plot_track_on_image(self):
        x_track, y_track = self.raw_track
        plt.imshow(self.reference_frame)
        plt.plot(x_track, y_track, color='b')

    def __gt__(self, other):
        return self.loom_idx > other.loom_idx
