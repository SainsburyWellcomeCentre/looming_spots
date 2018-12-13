import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import datetime
import seaborn as sns

from looming_spots.db.constants import LOOMING_STIMULUS_ONSET, END_OF_CLASSIFICATION_WINDOW, ARENA_SIZE_CM, \
    N_LOOMS_PER_STIMULUS, FRAME_RATE, TRACK_LENGTH

from looming_spots.analysis import tracks
from looming_spots.preprocess import extract_looms
from looming_spots.tracking.pyper_backend.auto_track import pyper_cli_track_trial
from looming_spots.reference_frames.viewer import Viewer
from looming_spots.util import video_processing


class LoomTrial(object):
    def __init__(self, session, directory, sample_number, trial_video_fname='loom{}.h264'):
        self.session = session
        self.sample_number = int(sample_number)
        self.directory = directory
        self.video_name = trial_video_fname.format(self.loom_number)
        self.video_path = os.path.join(self.directory, self.video_name)
        self.folder = os.path.join(self.directory, 'loom{}'.format(self.loom_number))
        self.time_to_first_loom = None  # TODO: this is quite specific, consider better implementation
        self.name = '{}_{}'.format(self.session.mouse_id, self.loom_number)

    def __gt__(self, other):
        return self.time > other.time

    @property
    def context(self):

        if 'A' in self.session.context and self.session.get_reference_frame(self.trial_type) is not None:
            if self.is_a9():
                return 'A9'
            return self.session.context
        else:
            return self.session.context

    @property
    def loom_number(self):
        return int(np.where(self.session.loom_idx == self.sample_number)[0][0] / N_LOOMS_PER_STIMULUS)

    def extract_video(self, overwrite=False):
        if not overwrite:
            if os.path.isfile(self.video_path):
                return 'video already exists... skipping'
        extract_looms.extract_loom_video_trial(self.session.video_path, self.video_path,
                                               self.sample_number, overwrite=overwrite)

    def is_a9(self):
        if np.mean(self.get_reference_frame()[340:390, 200:250]) < 50:  # FIXME: hard code
            return True

    @property
    def time(self):
        return self.session.dt + datetime.timedelta(0, int(self.sample_number/FRAME_RATE))

    @property
    def raw_track(self):
        return tracks.load_raw_track(self.folder)

    @property
    def x_track(self):
        return self.raw_track[0][:TRACK_LENGTH]

    @property
    def y_track(self):
        return self.raw_track[1][:TRACK_LENGTH]

    @property
    def normalised_x_track(self):
        return tracks.normalise_track(self.x_track, self.context)

    @property
    def smoothed_x_track(self):  # TODO: extract implementation to tracks
        return gaussian_filter(self.normalised_x_track, 3)

    @property
    def smoothed_y_track(self):  # TODO: extract implementation to tracks
        return gaussian_filter(self.y_track, 3)

    @property
    def smoothed_x_speed(self):  # TODO: extract implementation to tracks
        return np.diff(self.smoothed_x_track)

    @property
    def smoothed_y_speed(self):
        return np.diff(self.smoothed_y_track)

    @property
    def normalised_x_speed(self):  # TODO: extract implementation to tracks
        return np.diff(self.normalised_x_track)

    def peak_x_acc(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return min(acc_window)

    def peak_x_acc_idx(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return np.where(acc_window == min(acc_window))[0] + LOOMING_STIMULUS_ONSET

    def get_accelerations_to_shelter(self):
        acc_window = self.smoothed_x_acceleration[LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW]
        vel_window = self.smoothed_x_speed[LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW]
        acc_window[np.where(vel_window > 0)] = 10
        return acc_window

    @property
    def smoothed_x_acceleration(self):  # TODO: extract implementation to tracks
        return np.diff(self.smoothed_x_speed)

    def is_flee(self):
        return tracks.classify_flee(self.folder, self.context)

    def plot_peak_x_acceleration(self):
        if self.peak_x_acc() < -0.001:  # FIXME: hard code
            plt.plot(self.peak_x_acc_idx(), self.normalised_x_track[self.peak_x_acc_idx()], 'o', color='b')

    def plot_track_on_image(self, start=0, end=-1):
        x_track, y_track = self.raw_track
        plt.imshow(self.get_reference_frame())
        track_color = 'r' if self.is_flee() else 'k'
        if self.trial_type == 'habituation':
            track_color = 'b'
        plt.plot(x_track[start:end], y_track[start:end], color=track_color)

    @property
    def loom_location(self):
        x_track, y_track = self.raw_track
        return x_track[LOOMING_STIMULUS_ONSET], y_track[LOOMING_STIMULUS_ONSET]

    def loom_start_end_pos(self):
        start = self.get_start_pos()
        end = self.get_end_pos()
        return start, end

    def get_start_pos(self):
        x_track, y_track = self.raw_track
        for coord in zip(x_track, y_track):
            if not np.isnan(coord).any():
                return coord

    def get_end_pos(self):
        x_track, y_track = self.raw_track
        for coord in zip(x_track[::-1], y_track[::-1]):
            if not np.isnan(coord).any():
                return coord

    def max_speed(self):
        return np.nanmax(np.abs(self.normalised_x_speed))

    def plot_loom_location(self):
        #plt.imshow(self.session.reference_frame)
        x, y = self.loom_location
        start, end = self.loom_start_end_pos()
        plt.plot(x, y, 'o', color='k', markersize=20)
        #plt.plot(start[0], start[1], 'o', color='g')
        #plt.plot(end[0], end[1], 'o', color='r')

    @property
    def trial_type(self):
        if not self.session.contains_habituation:
            return 'test'  # FIXME: should probably be pre test here
        elif self.sample_number < self.session.habituation_loom_idx[0]:
            return 'pre_test'
        elif self.sample_number in self.session.habituation_loom_idx:
            return 'habituation'
        elif self.sample_number > self.session.habituation_loom_idx[-1]:
            return 'post_test'

    def plot_track(self, ax=None):
        if ax is None:
            ax = plt.gca()

        color = 'r' if self.is_flee() else 'k'
        plt.plot(self.normalised_x_track, color=color)
        plt.ylabel('x position in box (cm)')
        plt.xlabel('time (s)')

        track_length = self.get_x_length(ax)

        self.convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        self.convert_x_axis(track_length, n_steps=11)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    @staticmethod
    def get_x_length(ax=None):
        if ax is None:
            ax = plt.gca()
        line = ax.lines[0]
        xdata = line.get_xdata()
        return len(xdata)

    @staticmethod
    def convert_x_axis(track_length, n_steps):
        plt.xticks(np.linspace(0, track_length - 1, n_steps), np.linspace(0, track_length / FRAME_RATE, n_steps))

    @staticmethod
    def convert_y_axis(old_min, old_max, new_min, new_max, n_steps):
        plt.yticks(np.linspace(old_min, old_max, n_steps), np.linspace(new_min, new_max, n_steps))

    def get_reference_frame(self):
        self.make_reference_frames()
        return self.session.get_reference_frame(self.trial_type)

    def extract_track(self, overwrite=True):
        if not overwrite:
            if os.path.isdir(self.folder):
                return 'skipping... already tracked'

        if self.get_reference_frame() is None:
            print(self.directory)
            Viewer(self.directory, video_fname='loom{}.h264'.format(self.loom_number), trial_type=self.trial_type)
            #raise NoReferenceFrameError
        pyper_cli_track_trial(self.video_path, '{}_ref.npy'.format(self.trial_type))

    def make_reference_frames(self):
        if self.session.get_reference_frame(self.trial_type) is None:
            Viewer(self.directory, video_fname='loom{}.h264'.format(self.loom_number), trial_type=self.trial_type)

    def viewer(self):
        Viewer(self.directory, trial_type=self.trial_type, video_fname=self.video_name)

    def make_loom_superimposed_video(self, width=640, height=250, origin=(0, 40)):

        path_in = self.video_path
        path_out = '_overlay.'.join(path_in.split('.'))

        video_processing.loom_superimposed_video(path_in, path_out, width=width, height=height, origin=origin)


class NoReferenceFrameError(Exception):
    pass
