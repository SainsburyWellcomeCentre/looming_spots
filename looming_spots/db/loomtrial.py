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
    def __init__(self, session, directory, sample_number, trial_type, trial_video_fname='loom{}.h264'):
        self.session = session
        self.sample_number = int(sample_number)
        self.mouse_id = self.session.mouse_id

        self.directory = directory
        self.video_name = trial_video_fname.format(self.loom_number)
        self.video_path = os.path.join(self.directory, self.video_name)
        self.folder = os.path.join(self.directory, 'loom{}'.format(self.loom_number))

        self.time_to_first_loom = None

        self.name = '{}_{}'.format(self.mouse_id, self.loom_number)

        self.trial_type = trial_type
        self.next_trial = None
        self.previous_trial = None

    def __gt__(self, other):
        return self.time > other.time

    @classmethod
    def set_next_trial(cls, self, other):
        setattr(self, 'next_trial', other)

    @classmethod
    def set_previous_trial(cls, self, other):
        setattr(self, 'previous_trial', other)

    def habituation_loom_after(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == 'habituation':
                return True
            current_trial = current_trial.next_trial

    def habituation_loom_before(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == 'habituation':
                return True
            current_trial = current_trial.previous_trial

    def get_trial_type(self):
        if self.habituation_loom_before() and self.habituation_loom_after():
            return 'habituation'
        elif self.habituation_loom_after():
            return 'pre_test'
        elif self.habituation_loom_before():
            return 'post_test'
        else:
            return 'pre_test'

    def absolute_acceleration(self):
        return abs(self.peak_x_acc())

    @property
    def metric_functions(self):
        func_dict = {'speed': self.peak_speed,
                     'acceleration': self.absolute_acceleration,
                     'latency to escape': self.latency,
                     'time in safety zone': self.time_in_safety_zone,
                     'classified as flee': self.is_flee
                     }

        return func_dict

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

    def time_in_safety_zone(self):
        return tracks.time_spent_hiding(self.folder, self.context)

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
        return tracks.normalise_x_track(self.x_track, self.context)

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
        return np.nanmin(acc_window)

    def peak_x_acc_idx(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return int(np.where(acc_window == np.nanmin(acc_window))[0] + LOOMING_STIMULUS_ONSET)

    def get_accelerations_to_shelter(self):
        acc_window = self.smoothed_x_acceleration[LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW]
        vel_window = self.smoothed_x_speed[LOOMING_STIMULUS_ONSET:END_OF_CLASSIFICATION_WINDOW]
        acc_window[np.where(vel_window > 0)] = np.nan
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

    def peak_speed(self):
        peak_speed, arg_peak_speed = tracks.get_peak_speed_and_latency(self.folder, self.context)
        return peak_speed

    def latency(self):
        return (self.peak_x_acc_idx() - LOOMING_STIMULUS_ONSET)/FRAME_RATE

    @property
    def loom_location(self):
        x_track, y_track = self.raw_track
        return x_track[LOOMING_STIMULUS_ONSET], y_track[LOOMING_STIMULUS_ONSET]

    def max_speed(self):
        return np.nanmax(np.abs(self.normalised_x_speed))

    def plot_loom_location(self):
        plt.imshow(self.session.get_reference_frame)
        x, y = self.loom_location
        plt.plot(x, y, 'o', color='k', markersize=20)

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
