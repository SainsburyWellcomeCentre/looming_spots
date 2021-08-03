
import numpy as np
import seaborn as sns
from cached_property import cached_property
from looming_spots.analyse.escape_classification import classify_escape
from looming_spots.analyse.tracks import projective_transform_tracks, downsample_track, \
    normalised_speed_from_track, smooth_track, smooth_speed_from_track, get_peak_speed, smooth_acceleration_from_track, \
    latency_peak_detect_s, time_in_shelter, time_to_shelter, track_in_standard_space, get_tracking_method, \
    load_box_corner_coordinates
from looming_spots.constants import FRAME_RATE, ARENA_SIZE_CM, LOOMING_STIMULUS_ONSET, END_OF_CLASSIFICATION_WINDOW, \
    N_SAMPLES_TO_SHOW, N_SAMPLES_BEFORE,ARENA_LENGTH_PX, ARENA_WIDTH_PX
from looming_spots.util.plotting import get_x_length, convert_y_axis, convert_x_axis
from matplotlib import pyplot as plt
from scipy import signal


class Track(object):
    def __init__(self, folder, path, start, end, frame_rate):
        self.folder = folder
        self.frame_rate = frame_rate
        self.path = path
        self.start = start
        self.end = end
        self.x, self.y = self.track_in_standard_space

    @property
    def metric_functions(self):
        func_dict = {
            "speed": self.peak_speed,
            "acceleration": self.absolute_acceleration,
            "latency peak detect": self.latency,
            "reaction time": self.reaction_time_s,
            "time in safety zone": self.time_in_safety_zone,
            "classified as flee": self.is_escape,
            "time to reach shelter stimulus onset": self.time_to_shelter,
        }

        return func_dict

    @property
    def tracking_method(self):
        return get_tracking_method(self.path)

    @cached_property
    def track_in_standard_space(self):
        return track_in_standard_space(self.path, self.tracking_method, self.start, self.end, loom_folder=self.folder)

    def load_box_corner_coordinates(self):
        load_box_corner_coordinates(self.path)

    def projective_transform_tracks(self, Xin, Yin):
        new_track_x, new_track_y = projective_transform_tracks(Xin, Yin, self.load_box_corner_coordinates())
        return new_track_x, new_track_y

    @property
    def normalised_x_track(self, target_frame_rate=30):
        normalised_track = 1 - (self.x / ARENA_LENGTH_PX)
        if self.frame_rate != target_frame_rate:
            normalised_track = downsample_track(normalised_track, self.frame_rate)
        return normalised_track

    @property
    def x_track_real_units(self):
        return self.normalised_x_track * ARENA_SIZE_CM

    @property
    def normalised_y_track(self, target_frame_rate=30):
        normalised_track = (self.y / ARENA_WIDTH_PX) * 0.4
        if self.frame_rate != target_frame_rate:
            normalised_track = downsample_track(normalised_track, self.frame_rate)
        return normalised_track

    @property
    def normalised_x_speed(self):
        return normalised_speed_from_track(self.normalised_x_track)

    @property
    def smoothed_x_track(self):
        smoothed_x_track = smooth_track(self.normalised_x_track)
        return smoothed_x_track

    @property
    def smoothed_x_speed(self):
        smoothed_x_speed = smooth_speed_from_track(self.normalised_x_track)
        return smoothed_x_speed

    @property
    def smoothed_y_track(self):
        smoothed_y_track = smooth_track(self.normalised_y_track)
        return smoothed_y_track

    @property
    def smoothed_y_speed(self):
        smoothed_y_speed = smooth_speed_from_track(self.normalised_y_track)
        return smoothed_y_speed

    def absolute_acceleration(self):
        return abs(self.peak_x_acc()) * FRAME_RATE * ARENA_SIZE_CM

    def peak_x_acc(self):
        acc_window = self.get_accelerations_to_shelter()
        return np.nanmin(acc_window)

    def peak_x_acc_idx(self):
        acc_window = self.get_accelerations_to_shelter()
        return int(
            np.where(acc_window == np.nanmin(acc_window))[0]
            + LOOMING_STIMULUS_ONSET
        )

    def peak_speed(self, return_loc=False):
        return get_peak_speed(self.normalised_x_track, return_loc)

    def get_accelerations_to_shelter(self):
        acc_window = self.smoothed_x_acceleration[
            LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW
        ]
        vel_window = self.smoothed_x_speed[
            LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW
        ]
        acc_window[np.where(vel_window[:-1] > 0)] = np.nan  # TEST:
        return acc_window

    def reaction_time(self):
        n_stds = 1.2
        acc = -self.smoothed_x_acceleration[N_SAMPLES_BEFORE:]
        std = np.nanstd(acc)
        start = signal.find_peaks(acc, std * n_stds)[0][0]
        if start > 350:
            start = np.nan
        return start

    def reaction_time_s(self):
        return self.reaction_time() / FRAME_RATE

    @property
    def smoothed_x_acceleration(
        self
    ):
        return smooth_acceleration_from_track(self.normalised_x_track)

    def latency(self):
        return latency_peak_detect_s(self.normalised_x_track)

    def time_in_safety_zone(self):
        return time_in_shelter(self.normalised_x_track)

    def is_escape(self):
        return classify_escape(self.normalised_x_track)

    def time_to_shelter(self):
        return time_to_shelter(self.normalised_x_track)

    def plot(self, ax=None, color=None, n_samples_to_show=N_SAMPLES_TO_SHOW):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        if color is None:
            color = "r" if self.is_escape() else "k"
        plt.plot(self.normalised_x_track, color=color)
        plt.ylabel("x position in box (cm)")
        plt.xlabel("time (s)")
        plt.ylim([-0.1, 1])

        track_length = get_x_length(ax)

        convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        convert_x_axis(track_length, n_steps=11, frame_rate=FRAME_RATE)
        plt.xlim([0, n_samples_to_show])
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
