import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pims
from looming_spots.analyse.escape_classification import classify_escape

from looming_spots.analyse.tracks import downsample_track, downsample_y_track, smooth_track, smooth_speed, \
    normalise_speed, peak_speed, smooth_acceleration, projective_transform_tracks
from matplotlib import patches
from scipy import signal
from datetime import timedelta
import seaborn as sns
import pandas as pd

import looming_spots.preprocess.normalisation
import looming_spots.util.video_processing

from looming_spots.db.constants import (
    LOOMING_STIMULUS_ONSET,
    END_OF_CLASSIFICATION_WINDOW,
    ARENA_SIZE_CM,
    N_LOOMS_PER_STIMULUS,
    FRAME_RATE,
    N_SAMPLES_BEFORE,
    N_SAMPLES_AFTER,
    N_SAMPLES_TO_SHOW,
    SPEED_THRESHOLD, CLASSIFICATION_WINDOW_END, LOOM_ONSETS, BOX_CORNER_COORDINATES)

from looming_spots.analyse import arena_region_crossings
from looming_spots.preprocess import photodiode

from looming_spots.util import video_processing, plotting

from looming_spots.util.transformations import get_inverse_projective_transform, get_box_coordinates_from_file


class LoomTrial(object):
    def __init__(
        self,
        session,
        directory,
        sample_number,
        trial_type,
        stimulus_type,
        trial_video_fname="{}{}.mp4",
        contrast=None,
        loom_trial_idx=None,
        auditory_trial_idx=None,

    ):
        self.session = session
        self.n_samples_before = int(N_SAMPLES_BEFORE/30*self.frame_rate)
        self.sample_number = int(sample_number)
        self.mouse_id = self.session.mouse_id
        self.stimulus_type = stimulus_type
        self.contrast = contrast
        self.loom_trial_idx = loom_trial_idx
        self.auditory_trial_idx = auditory_trial_idx
        self.directory = directory
        self.video_name = trial_video_fname.format(
            self.stimulus_type, self.stimulus_number()
        )
        self.video_path = os.path.join(self.directory, self.video_name)
        self.folder = os.path.join(
            self.directory, f"{self.stimulus_type}{self.stimulus_number()}"
        )
        self.frame_rate = self.session.frame_rate

        self.time_to_first_loom = None

        self.name = f"{self.mouse_id}_{self.stimulus_number()}"

        self.trial_type = trial_type
        self.next_trial = None
        self.previous_trial = None

        self.start = max(self.sample_number - self.n_samples_before, 0)
        self.end = self.get_end()

    def get_end(self):
        if self.next_trial is not None:
            if self.next_trial.session != self.session:
                end = len(self.session)
            else:
                end = self.next_trial.sample_number
        else:
            end = len(self.session)
        return end

    def __gt__(self, other):
        return self.time > other.time

    def __eq__(self, other):
        return self.time == other.time

    def __add__(self, a):
        if not isinstance(a, int):
            raise TypeError
        self.sample_number += a
        self.start += a
        self.end += a

    @classmethod
    def set_next_trial(cls, self, other):
        setattr(self, "next_trial", other)

    @classmethod
    def set_previous_trial(cls, self, other):
        setattr(self, "previous_trial", other)

    def get_loom_trial_idx(self):
        return self.loom_trial_idx

    def get_auditory_trial_idx(self):
        return self.auditory_trial_idx

    def lsie_loom_after(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "habituation":
                return True
            current_trial = current_trial.next_trial

    def lsie_loom_before(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "habituation":
                return True
            current_trial = current_trial.previous_trial

    def get_last_lsie_trial(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "habituation":
                return current_trial
            current_trial = current_trial.previous_trial

    def set_contrast(self, contrast):
        self.contrast = contrast
        return self.contrast

    def set_loom_trial_idx(self, idx):
        self.loom_trial_idx = idx
        return self.loom_trial_idx

    def set_auditory_trial_idx(self, idx):
        self.auditory_trial_idx = idx
        return self.auditory_trial_idx

    def n_lsies(self):
        current_trial = self.first_trial()
        current_trial_type = current_trial.trial_type
        n_habituations = 1 if current_trial_type == "habituation" else 0

        while current_trial is not None:
            if current_trial.trial_type != current_trial_type:
                n_habituations += 1
                current_trial_type = current_trial.trial_type
            current_trial = current_trial.next_trial
        return n_habituations

    def first_trial(self):
        current_trial = self
        while current_trial.previous_trial is not None:
            current_trial = current_trial.previous_trial
        return current_trial

    def get_trial_type(self):
        if self.trial_type == "habituation":
            return "habituation"
        elif self.lsie_loom_before() and self.lsie_loom_after():
            return "post_test"  # TODO:'pre_and_post_test'
        elif self.lsie_loom_after():
            return "pre_test"
        elif self.lsie_loom_before():
            return "post_test"
        else:
            return "pre_test"

    def absolute_acceleration(self):
        return abs(self.peak_x_acc()) * self.frame_rate * 50

    @property
    def metric_functions(self):
        func_dict = {
            "speed": self.peak_speed,
            "acceleration": self.absolute_acceleration,
            "latency to escape": self.latency,
            "latency peak detect samples": self.latency_peak_detect,
            "latency peak detect": self.latency_peak_detect_s,
            "reaction time": self.reaction_time_s,
            "time in safety zone": self.time_in_safety_zone,
            "classified as flee": self.classify_escape,
            "time of loom": self.get_time,
            "loom number": self.get_loom_trial_idx,#self.get_stimulus_number,
            "time to reach shelter stimulus onset": self.time_to_reach_shelter_stim_onset,
            "time to reach shelter detection": self.time_to_reach_shelter_from_detection,
            "loom_evoked_speed_before_loom": self.loom_evoked_speed_before_loom,
            "loom_evoked_speed_after_loom": self.loom_evoked_speed_after_loom,
            "loom_evoked_speed_change": self.loom_evoked_speed_change,
            "movement_loom_on_vs_loom_off": self.movement_loom_on_vs_loom_off,
        }

        return func_dict

    @property
    def context(self):

        if (
            "A" in self.session.context
            and self.session.get_reference_frame(self.trial_type) is not None
        ):
            if self.session.contains_auditory():
                return "A9_auditory"
            if self.is_a9():
                return "A9"
            return self.session.context
        else:
            return self.session.context

    def stimulus_number(self):
        if self.stimulus_type == "loom":
            return self.loom_number
        elif self.stimulus_type == "auditory":
            return self.auditory_number

    @property
    def loom_number(self):
        if self.sample_number in self.session.loom_idx:
            return int(
                np.where(self.session.loom_idx == self.sample_number)[0][0]
                / N_LOOMS_PER_STIMULUS
            )

    @property
    def auditory_number(self):
        if self.sample_number in self.session.auditory_idx:
            return int(
                np.where(self.session.auditory_idx == self.sample_number)[0][0]
            )

    def get_stimulus_number(self):
        return self.stimulus_number()

    def processed_video_path(self):
        return list(pathlib.Path(self.directory).glob('cam_transform*.mp4'))[0]

    def extract_video(self, overwrite=False):
        print('extracting')
        if not overwrite:
            if os.path.isfile(self.video_path):
                return "video already exists... skipping"
        looming_spots.util.video_processing.extract_loom_video_trial(
            self.processed_video_path(),
            str(pathlib.Path(self.directory) / self.video_name),
            self.sample_number,
            overwrite=overwrite,
        )

    def is_a9(self):
        return True

    def photodiode(self):
        auditory_signal = self.session.data["photodiode"]
        return auditory_signal[self.start : self.end]

    def auditory_data(self):
        auditory_signal = self.session.data["auditory_stimulus"]
        return auditory_signal[self.start : self.end]

    @property
    def time(self):
        return self.session.dt + timedelta(
            0, int(self.sample_number / self.frame_rate)
        )

    def get_time(self):
        return self.time

    def time_in_safety_zone(self):
        samples_to_leave_shelter = self.samples_to_leave_shelter()
        if samples_to_leave_shelter is not None:
            return samples_to_leave_shelter / self.frame_rate
        else:
            return np.nan

    @property
    def tracking_method(self):
        p = pathlib.Path(self.session.path)
        lab5 = p / '5_label'

        if 'x_manual.npy' in os.listdir(str(p)):
            method = 'manual'

        elif "dlc_x_tracks.npy" in os.listdir(str(p)):
            method = 'dlc_1_label'

        elif len(list(lab5.glob("dlc_x_tracks.npy"))) > 0:
            method = 'dlc_5_label'

        else:
            method = 'old_school'

        return method

    @property
    def track_in_standard_space(self):
        p = pathlib.Path(self.session.path)
        lab5 = p / '5_label'

        if 'x_manual.npy' in os.listdir(str(p)):
            print("loading manually tracked")
            x, y = self.load_tracks(p, '{}_manual.npy')

        elif "dlc_x_tracks.npy" in os.listdir(str(p)):
            print("loading tracking results")
            x, y = self.load_tracks(p, 'dlc_{}_tracks.npy')

        elif len(list(lab5.glob("dlc_x_tracks.npy")))>0:
            print("loading 5 label tracking results")
            x, y = self.load_tracks(lab5, 'dlc_{}_tracks.npy')

        else:
            print(f'loading from folders {self.mouse_id}')
            x, y = looming_spots.preprocess.normalisation.load_raw_track(self.folder)
            x, y = self.projective_transform_tracks(x, y)

        return np.array(x), np.array(y)

    def load_tracks(self, p, name):
        x_path = p / name.format('x')
        y_path = p / name.format('y')
        x = np.load(str(x_path))[self.start:self.end]
        y = np.load(str(y_path))[self.start:self.end]
        return x, y

    def get_box_corner_coordinates(self):
        box_path = pathlib.Path(self.folder).parent.glob('box_corner_coordinates.npy')
        if len(list(box_path)) == 0:
            print('no box coordinates found...')

        return get_box_coordinates_from_file(
            str(list(pathlib.Path(self.folder).parent.glob('box_corner_coordinates.npy'))[0]))

    def projective_transform_tracks(self, Xin, Yin):

        new_track_x, new_track_y = projective_transform_tracks(Xin, Yin, self.get_box_corner_coordinates())
        return new_track_x, new_track_y

    @property
    def x_track(self):
        return self.track_in_standard_space[0]

    @property
    def y_track(self):
        return self.track_in_standard_space[1]

    @property
    def normalised_x_track(self, target_frame_rate=30):
        normalised_track = 1 - (self.x_track / 600)
        if self.frame_rate != target_frame_rate:
            normalised_track = downsample_track(normalised_track, self.frame_rate, self.n_samples_before)
        return normalised_track

    @property
    def normalised_y_track(self):
        normalised_track = (self.y_track / 240) * 0.4
        if self.frame_rate != 30:
            normalised_track = downsample_track(normalised_track, self.frame_rate, self.n_samples_before)
        return normalised_track

    @property
    def smoothed_x_track(self):
        smoothed_x_track = smooth_track("x")
        return smoothed_x_track

    @property
    def smoothed_x_speed(self):
        smoothed_x_speed = smooth_speed("x")
        return smoothed_x_speed

    @property
    def smoothed_y_track(self):
        smoothed_y_track = smooth_track("y")
        return smoothed_y_track

    @property
    def smoothed_y_speed(self):
        smoothed_y_speed = smooth_speed("y")
        return smoothed_y_speed

    @property
    def normalised_x_speed(self):
        return normalise_speed(self.normalised_x_track)

    def peak_x_acc(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return np.nanmin(acc_window)

    def peak_x_acc_idx(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return int(
            np.where(acc_window == np.nanmin(acc_window))[0]
            + LOOMING_STIMULUS_ONSET
        )

    def peak_speed(self, return_loc=False):
        return peak_speed(self.normalised_x_track, return_loc)

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
        acc = -self.smoothed_x_acceleration[self.n_samples_before:]
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
        return smooth_acceleration(self.normalised_x_track)

    def classify_escape(self):
        return classify_escape()

    def latency(self):
        return (
            self.estimate_latency(False) - LOOMING_STIMULUS_ONSET
        ) / self.frame_rate

    def latency_peak_detect(self):
        n_stds = 2.5
        speed = -self.smoothed_x_speed[N_SAMPLES_BEFORE:]  # self.n_samples_before
        std = np.nanstd(speed[:600])
        all_peak_starts = signal.find_peaks(speed, std * n_stds, width=1)[1]['left_ips']
        if len(all_peak_starts) > 0:
            if self.mouse_id == '1029105' and self.loom_trial_idx == 0:
                return all_peak_starts[1] + 200
            elif self.mouse_id == 'CA105_2' and self.loom_trial_idx == 2:
                return 260
            elif self.mouse_id == 'CA105_4' and self.loom_trial_idx == 2:
                return 244
            elif self.mouse_id == 'CA106_5' and self.loom_trial_idx == 0:
                return 239
            elif self.mouse_id == 'CA109_2' and self.loom_trial_idx == 2:
                return 252
            elif self.mouse_id == 'CA109_3' and self.loom_trial_idx == 0:
                return 239
            elif self.mouse_id == 'CA109_2' and self.loom_trial_idx == 2:
                return 252
            elif self.mouse_id == '074743' and self.loom_trial_idx == 1:
                return 235
            elif self.mouse_id == '898989' and self.loom_trial_idx == 0:
                return 240
            elif self.mouse_id == '977659' and self.loom_trial_idx == 28:
                return 242

            elif self.mouse_id == '898990' and self.loom_trial_idx == 27:
                return 273
            elif self.mouse_id == '898990' and self.loom_trial_idx == 28:
                return 212
            elif self.mouse_id == '898990' and self.loom_trial_idx == 29:
                return 238

            return all_peak_starts[0] + 200  # signal.find_peaks(acc, height=0.0002)[0][0]

    def latency_peak_detect_s(self):

        latency_pd = self.latency_peak_detect()
        if latency_pd is not None:
            latency_pd -= N_SAMPLES_BEFORE  # self.n_samples_before
            return latency_pd / FRAME_RATE  # self.frame_rate

    def latency_p(self):
        return self.peak_x_acc_idx() - LOOMING_STIMULUS_ONSET

    def has_escaped_by(self, sample_n):
        return self.estimate_latency(False) < sample_n

    def time_to_reach_shelter_stim_onset(self):
        n_samples_to_reach_shelter = self.n_samples_to_reach_shelter()
        if n_samples_to_reach_shelter is None:
            return n_samples_to_reach_shelter

        return (n_samples_to_reach_shelter-N_SAMPLES_BEFORE) / FRAME_RATE  # self.n_samples_before) / self.frame_rate

    def time_to_reach_shelter_from_detection(self):
        n_samples_to_reach_shelter = self.n_samples_to_reach_shelter()
        if n_samples_to_reach_shelter is None:
            return n_samples_to_reach_shelter
        return (n_samples_to_reach_shelter-self.n_samples_before-self.reaction_time()) / self.frame_rate

    @property
    def mouse_location_at_stimulus_onset(self):
        x_track, y_track = self.track_in_standard_space
        return x_track[LOOMING_STIMULUS_ONSET], y_track[LOOMING_STIMULUS_ONSET]

    def max_speed(self):
        return np.nanmax(np.abs(self.normalised_x_speed))

    def plot_mouse_location_at_stimulus_onset(self):
        ref_frame = self.session.get_reference_frame(idx=self.sample_number)
        plt.imshow(ref_frame)
        x, y = self.mouse_location_at_stimulus_onset
        plt.plot(x, y, "o", color="k", markersize=20)

    def plot_track(
        self, ax=None, color=None, n_samples_to_show=N_SAMPLES_TO_SHOW, frame_rate=30
    ):
        n_samples_to_show = int(n_samples_to_show/30*frame_rate)
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        if color is None:
            color = "r" if self.classify_escape() else "k"
        plt.plot(self.normalised_x_track, color=color)
        plt.ylabel("x position in box (cm)")
        plt.xlabel("time (s)")
        plt.ylim([-0.1, 1])

        track_length = self.get_x_length(ax)

        self.convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        self.convert_x_axis(track_length, n_steps=11)
        plt.xlim([0, n_samples_to_show])
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    def get_tracks_path(self, directory):
        fname = '*.h5'
        h5_files = list(directory.glob(f'{fname}'))

        if h5_files:
            for f in h5_files:
                if 'filtered' in str(f):
                    return f
            return h5_files[0]

        return None

    @staticmethod
    def get_x_length(ax=None):
        if ax is None:
            ax = plt.gca()
        line = ax.lines[0]
        xdata = line.get_xdata()
        return len(xdata)

    @staticmethod
    def convert_x_axis(track_length, n_steps):
        plt.xticks(
            np.linspace(0, track_length - 1, n_steps),
            np.linspace(0, track_length / FRAME_RATE, n_steps),
        )

    @staticmethod
    def convert_y_axis(old_min, old_max, new_min, new_max, n_steps):
        plt.yticks(
            np.linspace(old_min, old_max, n_steps),
            np.linspace(new_min, new_max, n_steps),
        )

    def days_since_last_session_type(self):
        if self.lsie_loom_before():
            last_habituation_trial = self.get_last_lsie_trial()
            time_since_last_session_type = (
                self.time - last_habituation_trial.time
            )
            return self.round_timedelta(time_since_last_session_type)

    @staticmethod
    def round_timedelta(td):
        if td.seconds > 60 * 60 * 12:
            return td + timedelta(1)
        else:
            return td

    def actual_loom_onsets(self):
        pd = self.photodiode()
        locs = np.where(np.diff(pd > 0.4))[0]
        starts = locs[::2]
        ends = locs[1::2]
        return starts, ends

    def events_df(self):
        metric_dict = {}

        idx = np.arange(len(self.actual_loom_onsets()[0]))
        trial_idx = [self.loom_number] * len(idx)
        mouse_ids = [self.session.mouse_id] * len(idx)

        metric_dict.setdefault("within_stimulus_loom_id", idx)
        metric_dict.setdefault("trial id", trial_idx)
        metric_dict.setdefault("mouse_id", mouse_ids)

        return pd.DataFrame.from_dict(metric_dict)

    def plot_stimulus(self):  # FIXME: duplicated elsewhere
        ax = plt.gca()
        if self.stimulus_type == "auditory":
            patch = patches.Rectangle(
                (self.n_samples_before, -0.2), 190, 1.3, alpha=0.1, color="r"
            )
            ax.add_patch(patch)
        else:
            plotting.plot_looms_ax(ax)

    def raw_start(self):
        return photodiode.find_nearest_pd_up_from_frame_number(
            self.session.path, self.start
        )

    def n_samples_to_reach_shelter(self):
        n_samples = arena_region_crossings.get_next_entry_from_track(
            self.smoothed_x_track,
            "shelter",
            "middle",
            LOOMING_STIMULUS_ONSET,
        )
        return n_samples

    def samples_to_leave_shelter(self):
        start = self.n_samples_to_reach_shelter()
        if start is None:
            print(
                "mouse never returns to shelter, not computing time to leave shelter"
            )
            return None
        return arena_region_crossings.get_next_entry_from_track(
            self.smoothed_x_track, "middle", "shelter", start
        )

    def n_samples_to_tz_reentry(self):
        return arena_region_crossings.get_next_entry_from_track(
            self.smoothed_x_track,
            "tz",
            "middle",
            LOOMING_STIMULUS_ONSET
        )

    def track_overlay(self, duration_in_samples=200, track_heatmap=None):
        if track_heatmap is None:
            track_heatmap = np.zeros((240, 600))  # TODO: get shape from raw

        x, y = (
            np.array(
                self.track_in_standard_space[0][
                    LOOMING_STIMULUS_ONSET : LOOMING_STIMULUS_ONSET
                    + duration_in_samples
                ]
            ),
            np.array(
                self.track_in_standard_space[1][
                    LOOMING_STIMULUS_ONSET : LOOMING_STIMULUS_ONSET
                    + duration_in_samples
                ]
            ),
        )
        for coordinate in zip(x, y):
            if not np.isnan(coordinate).any():
                track_heatmap[int(coordinate[1]), int(coordinate[0])] += 1
        return track_heatmap

    def loom_evoked_speed_before_loom(self):
        return looming_spots.analyse.escape_classification.loom_evoked_speed_change(
            self.smoothed_x_speed,
            LOOMING_STIMULUS_ONSET,
            window_before=14,
            window_after=14,
        )[
            0
        ]

    def loom_evoked_speed_after_loom(self):
        return looming_spots.analyse.escape_classification.loom_evoked_speed_change(
            self.smoothed_x_speed,
            LOOMING_STIMULUS_ONSET,
            window_before=20,
            window_after=20,
        )[
            1
        ]

    def loom_evoked_speed_change(self):
        if not self.classify_escape():
            return (
                self.loom_evoked_speed_before_loom()
                - self.loom_evoked_speed_after_loom()
            )

    def movement_loom_on_vs_loom_off(self):
        if not self.classify_escape():
            return looming_spots.analyse.escape_classification.movement_loom_on_vs_loom_off(
                self.smoothed_x_speed
            )




    def plot_peak_x_acceleration(self):
        if self.peak_x_acc() < -0.001:  # FIXME: hard code
            plt.plot(
                self.peak_x_acc_idx(),
                self.normalised_x_track[self.peak_x_acc_idx()],
                "o",
                color="b",
            )

    def plot_track_on_image(self, start=0, end=-1):
        x_track, y_track = self.track_in_standard_space
        #plt.imshow(self.get_reference_frame())
        track_color = "r" if self.classify_escape() else "k"
        if self.trial_type == "habituation":
            track_color = "b"
        plt.plot(x_track[start:end], y_track[start:end], color=track_color)
        plt.plot(x_track[self.n_samples_before], y_track[self.n_samples_before], 'o')

class VisualStimulusTrial(LoomTrial):
    def __init__(
        self,
        session,
        directory,
        sample_number,
        trial_type,
        stimulus_type="loom",
    ):
        super().__init__(
            session,
            directory,
            sample_number,
            trial_type,
            stimulus_type,
            trial_video_fname="{}{}.mp4",

        )

    def make_loom_superimposed_video(
        self, width=600, height=240, origin=(0, 40)
    ):

        path_in = str(pathlib.Path(self.directory) / self.video_name)
        path_out = "_overlay.".join(path_in.split("."))

        video_processing.loom_superimposed_video(
            path_in, path_out, width=width, height=height, origin=origin, track=self.track_in_standard_space
        )

    def get_video(self):
        vid_path = pathlib.Path(
            self.session.path.replace("processed", "raw")
        ).joinpath("camera.avi")
        vid = pims.Video(vid_path)
        return vid[self.start:self.end]


class AuditoryStimulusTrial(LoomTrial):
    def __init__(
        self,
        session,
        directory,
        sample_number,
        trial_type,
        stimulus_type="auditory",
    ):
        super().__init__(
            session, directory, sample_number, trial_type, stimulus_type
        )

    def make_stimulus_superimposed_video(
        self, width=640, height=250, origin=(0, 40)
    ):
        raise NotImplementedError
