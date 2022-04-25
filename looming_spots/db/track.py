import numpy as np
import seaborn as sns
from cached_property import cached_property
from looming_spots.analyse.escape_classification import classify_escape
from looming_spots.analyse.track_functions import (
    normalise_x_track,
    normalise_y_track,
    projective_transform_tracks,
    downsample_track,
    normalised_speed_from_track,
    smooth_track,
    smooth_speed_from_track,
    get_peak_speed,
    smooth_acceleration_from_track,
    latency_peak_detect_s,
    estimate_reaction_time,
    time_in_shelter,
    time_to_shelter,
    track_in_standard_space,
    get_tracking_method,
    load_box_corner_coordinates,
)
from looming_spots.constants import (
    FRAME_RATE,
    ARENA_SIZE_CM,
    LOOMING_STIMULUS_ONSET,
    END_OF_CLASSIFICATION_WINDOW,
    N_SAMPLES_TO_SHOW,
    N_SAMPLES_BEFORE,
    ARENA_LENGTH_PX,
    ARENA_WIDTH_PX,
)
from looming_spots.util.plotting import (
    get_x_length,
    convert_y_axis,
    convert_x_axis,
)
from matplotlib import pyplot as plt
from scipy import signal


class Track(object):
    """
    The Track class handles everything to do with positional x-y tracks. Tracks have been designed to belong to
    instances of Trial. All track loading is done here, and tracks are limited based on the start and end of the trial
    the Trial frame rate and paths to the processed data.

    """

    def __init__(
        self, folder, path, start, end, frame_rate, transformed=None, padding=None
    ):
        self.folder = folder
        self.frame_rate = frame_rate
        self.path = path
        self.start = start
        self.end = end
        self.padding = padding
        self.transformed = transformed
        self.x, self.y = self.track_in_standard_space

    @property
    def metric_functions(self):
        func_dict = {
            "speed": self.peak_speed,
            "latency peak detect": self.latency,
            "reaction time": self.reaction_time_s,
            "time in safety zone": self.time_in_safety_zone,
            "classified as flee": self.is_escape,
            "time to reach shelter stimulus onset": self.time_to_shelter,
        }

        return func_dict

    @property
    def tracking_method(self):
        """
        At different stages of the project, tracking was performed differently owing to either logistical constraints
        or due to different experimental needs (e.g. more tracking labels). This tracking method simply returns which
        tracking method was used to generate the tracks for a given trial.
        :return: tracking_method key
        """

        return get_tracking_method(self.path)

    @cached_property
    def track_in_standard_space(self):
        return track_in_standard_space(
            self.path,
            self.tracking_method,
            self.start,
            self.end,
            loom_folder=self.folder,
            transformed=self.transformed,
            padding=self.padding,
        )

    def load_box_corner_coordinates(self):
        """
        Gets the coordinates of the arena box for this trial/track.
        :return:
        """
        return load_box_corner_coordinates(self.path)

    def projective_transform_tracks(self, x_in, y_in):
        """
        To correct for camera angle artifacts, coordinates of the arena and its known real geometry are used to
        get a projective transform that can be applied to positional tracks or raw videos.
        :param x_in:
        :param y_in:
        :return:
        """

        new_track_x, new_track_y = projective_transform_tracks(
            x_in, y_in, self.load_box_corner_coordinates()
        )
        return new_track_x, new_track_y

    @property
    def normalised_x_track(self, display_frame_rate=30):
        return normalise_x_track(self.x, self.frame_rate, display_frame_rate)

    @property
    def x_track_real_units(self):
        return self.normalised_x_track * ARENA_SIZE_CM

    @property
    def normalised_y_track(self, display_frame_rate=30):
        return normalise_y_track(self.y, display_frame_rate)

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

    def peak_speed(self, return_loc=False):
        return get_peak_speed(self.normalised_x_track, return_loc)

    def reaction_time(self):
        return self.estimate_reaction_time(self.smoothed_x_acceleration())

    def reaction_time_s(self):
        return self.reaction_time() / FRAME_RATE

    @property
    def smoothed_x_acceleration(self):
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
