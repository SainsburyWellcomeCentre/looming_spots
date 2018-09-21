import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from looming_spots.analysis import tracks, plotting
import datetime

from looming_spots.preprocess import extract_looms
from looming_spots.tracking.pyper_backend.auto_track import pyper_cli_track_trial
from looming_spots.reference_frames.viewer import Viewer
from looming_spots.util import video_processing
import seaborn as sns

LOOM_ONSET = 200
END_OF_CLASSIFICATION_WINDOW = 550
ARENA_SIZE_CM = 50
FRAME_RATE = 30

# TODO: implement pre/post test attribute


class LoomTrial(object):
    def __init__(self, session, directory, sample_number):
        self.session = session
        self.sample_number = int(sample_number)
        self.directory = directory
        self.video_name = 'loom{}.h264'.format(self.loom_number)
        self.video_path = os.path.join(self.directory, self.video_name)
        self.folder = os.path.join(self.directory, 'loom{}'.format(self.loom_number))
        self.context = session.context
        self.time_to_first_loom = None  # TODO: this is quite specific, consider better implementation
        self.name = '{}_{}'.format(self.session.mouse_id, self.loom_number)

    @property
    def loom_number(self):
        return int(np.where(self.session.loom_idx == self.sample_number)[0][0]/5)

    def extract_video(self, overwrite=False):
        if not overwrite:
            if os.path.isfile(self.video_path):
                return 'video already exists... skipping'
        extract_looms.extract_loom_video_trial(self.session.video_path, self.video_path,
                                               self.sample_number, overwrite=overwrite)

    @property
    def time(self):
        return self.session.dt + datetime.timedelta(0, int(self.sample_number/30))

    @property
    def raw_track(self):
        return tracks.load_raw_track(self.folder)

    @property
    def normalised_x_track(self):
        return tracks.load_normalised_track(self.folder, self.context)

    @property
    def smoothed_x_track(self):  # TODO: extract implementation to tracks
        return gaussian_filter(self.normalised_x_track, 3)

    @property
    def smoothed_y_track(self):  # TODO: extract implementation to tracks
        return gaussian_filter(self.raw_track[1], 3)

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
        return np.where(acc_window == min(acc_window))[0] + LOOM_ONSET

    def get_accelerations_to_shelter(self):
        acc_window = self.smoothed_x_acceleration[LOOM_ONSET:END_OF_CLASSIFICATION_WINDOW]
        vel_window = self.smoothed_x_speed[LOOM_ONSET:END_OF_CLASSIFICATION_WINDOW]
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
        return x_track[LOOM_ONSET], y_track[LOOM_ONSET]

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
            return 'test'
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

    def get_x_length(self, ax=None):
        if ax is None:
            ax = plt.gca()
        line = ax.lines[0]
        xdata = line.get_xdata()
        return len(xdata)

    def convert_x_axis(self, track_length, n_steps):
        plt.xticks(np.linspace(0, track_length - 1, n_steps), np.linspace(0, (track_length - 1) / FRAME_RATE, n_steps))

    @staticmethod
    def convert_y_axis(old_min, old_max, new_min, new_max, n_steps):
        plt.yticks(np.linspace(old_min, old_max, n_steps), np.linspace(new_min, new_max, n_steps))

    def get_reference_frame(self):
        return self.session.get_reference_frame(self.trial_type)

    def track_trial(self, overwrite=True):
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

    def __gt__(self, other):
        return self.time > other.time

    def viewer(self):
        Viewer(self.directory, trial_type=self.trial_type, video_fname=self.video_name)

    def loom_superimposed_video(self, out_folder='/home/slenzi/Desktop/video_analysis/'):
        video_path = os.path.join(self.folder, 'overlay_video.h264')
        if not os.path.isfile(video_path):
            vid = video_processing.load_video_from_path(self.video_path)
            vid = video_processing.crop_video(vid, 640, 250, (0, 40))
            rprof = video_processing.loom_radius_profile(len(vid))
            new_vid = video_processing.plot_loom_on_video(vid, rprof)
            print(self.name)
            out_path = os.path.join(out_folder, self.name) + '.h264'
            print(out_path)
            video_processing.save_video(new_vid, out_path)

        return video_processing.load_video_from_path(out_path)


class LoomTrialGroup(object):
    def __init__(self, trials):
        self.trials = trials

    def add_trials(self, trials):
        for trial in trials:
            self.trials.append(trial)

    def get_trials(self):
        return self.trials

    def plot_all_tracks(self):
        for t in self.get_trials():
            t.plot_track()

    def plot_all_peak_acc(self):
        for t in self.get_trials():
            t.plot_peak_x_acceleration()

    def all_tracks(self):
        return [t.smoothed_x_speed for t in self.trials]

    def sorted_tracks(self, values_to_sort_by=None):
        if values_to_sort_by is None:
            return self.all_tracks()
        else:
            args = np.argsort(values_to_sort_by)
            order = [np.where(args == x)[0][0] for x in range(len(self.all_tracks()))]
            sorted_tracks = []
            for item, arg, sort_var in zip(order, args, values_to_sort_by):
                trial_distances = self.all_tracks()[arg]
                sorted_tracks.append(trial_distances)
            return sorted_tracks

    def plot_hm(self, values_to_sort_by):
        tracks = self.sorted_tracks(values_to_sort_by)
        plt.imshow(tracks, cmap='Greys', aspect='auto', vmin=-0.03, vmax=0.03)

    @property
    def n_flees(self):
        return np.count_nonzero([t.is_flee() for t in self.trials])

    @property
    def n_non_flees(self):
        return len(self.trials) - self.n_flees

    @property
    def flee_rate(self):
        return self.n_flees/len(self.trials)

    def latencies(self):
        latencies = []
        for t in self.trials:
            latency = int(t.peak_x_acc_idx())
            latencies.append(latency)
        return latencies

    def times_to_first_loom(self):
        times_to_first_loom = []
        for t in self.trials:
            times_to_first_loom.append(t.time_to_first_loom)
        return times_to_first_loom

    def plot_probable_jumps(self):
        pass


class NoReferenceFrameError(Exception):
    pass
