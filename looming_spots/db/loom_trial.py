import os
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pims
from cached_property import cached_property
from matplotlib import patches
from scipy.ndimage import gaussian_filter
from scipy import signal
from datetime import timedelta
import seaborn as sns
import pandas as pd
import photometry

import looming_spots.track_analysis.escape_classification
import looming_spots.preprocess.normalisation
import looming_spots.util.video_processing
from looming_spots.util.event_detection.events_collection import (
    EventsCollection,
)
from looming_spots.db.constants import (
    LOOMING_STIMULUS_ONSET,
    END_OF_CLASSIFICATION_WINDOW,
    ARENA_SIZE_CM,
    N_LOOMS_PER_STIMULUS,
    FRAME_RATE,
    N_SAMPLES_BEFORE,
    N_SAMPLES_AFTER,
    N_SAMPLES_TO_SHOW,
    SPEED_THRESHOLD, CLASSIFICATION_WINDOW_END)

from looming_spots.track_analysis import arena_region_crossings
from looming_spots.preprocess import photodiode

from looming_spots.util import video_processing, plotting
from photometry.event_detect import (
    get_starts_and_ends,
    subtract_event_from_trace,
)
from photometry import events
from photometry.demodulation import apply_butterworth_lowpass_filter


def remove_pre_stimulus_events(df, event_times):
    for e in event_times():
        pass


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

        #self.end = self.sample_number + N_SAMPLES_AFTER

        self.time_to_first_loom = None

        self.name = f"{self.mouse_id}_{self.stimulus_number()}"

        self.trial_type = trial_type
        self.next_trial = None
        self.previous_trial = None

        self.start = max(self.sample_number - N_SAMPLES_BEFORE, 0)
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

    def habituation_loom_after(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "habituation":
                return True
            current_trial = current_trial.next_trial

    def habituation_loom_before(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "habituation":
                return True
            current_trial = current_trial.previous_trial

    def get_last_habituation_trial(self):
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

    def n_habituations(self):
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
        elif self.habituation_loom_before() and self.habituation_loom_after():
            return "post_test"  # TODO:'pre_and_post_test'
        elif self.habituation_loom_after():
            return "pre_test"
        elif self.habituation_loom_before():
            return "post_test"
        else:
            return "pre_test"

    def absolute_acceleration(self):
        return abs(self.peak_x_acc()) * FRAME_RATE * 50

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
            "classified as flee": self.is_flee,
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
    def normalisation_dict(self):
        normalisation_dict = {
            "speed": 120,
            "acceleration": 6,
            "latency to escape": 12,
            "time in safety zone": 12,
            "classified as flee": 1,
            "time to reach safety": 12,
        }
        return normalisation_dict

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
        #if (
        #    np.mean(self.get_reference_frame()[340:390, 200:250]) < 50
        #):  # FIXME: hard code
        #    return True

    def laser_on(self):
        sig, bg = self.signal_and_background()
        bg_zero = bg - np.median(bg[:100])
        if any(bg_zero[:600] > 0.000001):
            return True

    def delta_f(self):
        df = self.session.data["delta_f"][self.start : self.end][:600]
        #df = remove_pre_stimulus_events(df)
        #df -= np.median(df[185:200])  # 120 # 195
        df -= np.median(df[176:200])  # 120 # 195
        return df

    def signal_and_background(self):
        sig = self.session.data["signal"][self.start : self.end]
        bg = self.session.data["bg_fit"][self.start : self.end]
        return sig, bg
    #
    # def raw_data(self):
    #     return self.session.data['photodetector'][self.start:self.end]
    #
    # def ref211(self):
    #     return self.session.data['led211'][self.start:self.end]
    #
    # def ref531(self):
    #     return self.session.data['led531'][self.start:self.end]
    #
    # def demodulated_trial_delta_f(self):
    #     signal, background, bg_fit, delta_f=photometry.demodulation.lerner_deisseroth_preprocess(self.raw_data(), self.ref211(), self.ref531())
    #     return signal, background, bg_fit, delta_f

    @cached_property
    def fully_sampled_delta_f(self):
        start = self.raw_start() - int(
            200 * 10000 / FRAME_RATE
        )  # TODO: derive from raw data instead for precision
        end = start + int(400 * 10000 / FRAME_RATE)
        df = self.session.fully_sampled_delta_f[0][start:end]

        med_range = [
            int(100 * 10000 / FRAME_RATE),
            int(190 * 10000 / FRAME_RATE),
        ]  # 150, 200
        df -= np.median(df[med_range[0] : med_range[1]])
        return df

    def delta_f_post_events_removed(self, df, template_width=30):
        for e in self.events():
            if e.start_p > LOOMING_STIMULUS_ONSET + 10:
                df = subtract_event_from_trace(df, e.peak_p)
        return df

    def delta_f_with_pre_stimulus_events_removed(self, template_width=30):

        df = self.delta_f().copy()
        for e in self.events():
            if template_width < e.start_p < LOOMING_STIMULUS_ONSET:
                df = subtract_event_from_trace(df, e.peak_p)
        return df

    def photodiode(self):
        auditory_signal = self.session.data["photodiode"]
        return auditory_signal[self.start : self.end]

    def auditory_data(self):
        auditory_signal = self.session.data["auditory_stimulus"]
        return auditory_signal[self.start : self.end]

    @property
    def time(self):
        return self.session.dt + timedelta(
            0, int(self.sample_number / FRAME_RATE)
        )

    def get_time(self):
        return self.time

    def time_in_safety_zone(self):
        samples_to_leave_shelter = self.samples_to_leave_shelter()
        if samples_to_leave_shelter is not None:
            return samples_to_leave_shelter / FRAME_RATE
        else:
            return np.nan

    @property
    def raw_track(self):
        if 'x_manual.npy' in os.listdir(self.session.path):
            print('WARNING this was manually tracked')
            x_path = os.path.join(
                self.session.path, "x_manual.npy"
            )  # TODO: extract to sessionio
            y_path = os.path.join(self.session.path, "y_manual.npy")

            x = np.load(x_path)[self.start:self.end]
            y = np.load(y_path)[self.start:self.end]
            return x, y

        elif "dlc_x_tracks.npy" in os.listdir(self.session.path):
            x_path = os.path.join(
                self.session.path, "dlc_x_tracks.npy"
            )  # TODO: extract to sessionio
            y_path = os.path.join(self.session.path, "dlc_y_tracks.npy")

            x = np.load(x_path)[self.start:self.end]
            y = np.load(y_path)[self.start:self.end]
            return x, y
        else:
            print('loading from folders')
            return looming_spots.preprocess.normalisation.load_raw_track(
                self.folder
            )

    @property
    def x_track(self):
        return self.raw_track[0]

    @property
    def y_track(self):
        return self.raw_track[1]

    @property
    def normalised_x_track(self):
        return 1 - (self.x_track / 600)
        #return looming_spots.preprocess.normalisation.normalise_x_track(
        #    self.x_track, self.context
        #)

    @property
    def normalised_y_track(self):
        return (self.y_track / 240) * 0.4

    @property
    def smoothed_x_track(self):  # TODO: extract implementation to tracks
        #b, a = signal.butter(2, 0.125)

        #y = signal.filtfilt(b, a, self.normalised_x_track, padlen=150) returns nan for some reason
        y = gaussian_filter(self.normalised_x_track, 2)
        return y

    @property
    def smoothed_y_track(self):  # TODO: extract implementation to tracks
        return gaussian_filter(self.normalised_y_track, 2)

    @property
    def smoothed_x_speed(self):  # TODO: extract implementation to tracks
        return np.diff(self.smoothed_x_track)

    @property
    def smoothed_y_speed(self):
        return np.diff(self.smoothed_y_track)

    @property
    def normalised_x_speed(self):  # TODO: extract implementation to tracks
        normalised_x_speed = np.concatenate(
            [[np.nan], np.diff(self.normalised_x_track)]
        )
        return normalised_x_speed

    def peak_x_acc(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return np.nanmin(acc_window)

    def peak_x_acc_idx(self):  # TODO: extract implementation to tracks
        acc_window = self.get_accelerations_to_shelter()
        return int(
            np.where(acc_window == np.nanmin(acc_window))[0]
            + LOOMING_STIMULUS_ONSET
        )

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
        acc = -self.smoothed_x_acceleration[200:]
        std = np.nanstd(acc)
        start = signal.find_peaks(acc, std * n_stds)[0][0]
        if start > 350:
            start = np.nan
        return start #signal.find_peaks(acc, height=0.0002)[0][0]

    def reaction_time_s(self):
        return self.reaction_time() / FRAME_RATE

    @property
    def smoothed_x_acceleration(
        self
    ):  # TODO: extract implementation to tracks
        return np.roll(np.diff(self.smoothed_x_speed), 2)

    def is_flee(self):
        return self.classify_flee()

    def classify_flee(self):
        peak_speed, arg_peak_speed = self.peak_speed(True)
        latency = self.latency_peak_detect()
        time_to_shelter = self.n_samples_to_reach_shelter()
        print(f'speed {peak_speed}, threshold {-SPEED_THRESHOLD}, latency {latency} limit: {CLASSIFICATION_WINDOW_END-20}, time to shelter {time_to_shelter}, limit: {CLASSIFICATION_WINDOW_END}')

        if time_to_shelter is None or latency is None:
            return False

        return (peak_speed > -SPEED_THRESHOLD) and (time_to_shelter < CLASSIFICATION_WINDOW_END) #and (latency < (CLASSIFICATION_WINDOW_END-20))

    def classify_flee_old(self):
        track = gaussian_filter(self.normalised_x_track, 3)
        speed = np.diff(track)

        if looming_spots.track_analysis.escape_classification.fast_enough(
            speed
        ) and looming_spots.track_analysis.escape_classification.reaches_home(
            track, self.context
        ):  # and not leaves_house(track, self.context)
            leaves_house_within = looming_spots.track_analysis.escape_classification.leaves_house(
                track, self.context
            )
            print(f"leaves: {leaves_house_within}")
            return True
        fast_enough = looming_spots.track_analysis.escape_classification.fast_enough(
            speed
        )
        reaches_shelter = looming_spots.track_analysis.escape_classification.reaches_home(
            track, self.context
        )
        print(f"fast enough: {fast_enough}, reaches home: {reaches_shelter}")
        return False

    def plot_peak_x_acceleration(self):
        if self.peak_x_acc() < -0.001:  # FIXME: hard code
            plt.plot(
                self.peak_x_acc_idx(),
                self.normalised_x_track[self.peak_x_acc_idx()],
                "o",
                color="b",
            )

    def plot_track_on_image(self, start=0, end=-1):
        x_track, y_track = self.raw_track
        plt.imshow(self.get_reference_frame())
        track_color = "r" if self.is_flee() else "k"
        if self.trial_type == "habituation":
            track_color = "b"
        plt.plot(x_track[start:end], y_track[start:end], color=track_color)

    def peak_speed(self, return_loc=False):
        peak_speed, arg_peak_speed = looming_spots.track_analysis.escape_classification.get_peak_speed_and_latency(
            self.normalised_x_track
        )
        peak_speed = peak_speed * FRAME_RATE * ARENA_SIZE_CM
        if return_loc:
            return peak_speed, arg_peak_speed
        return peak_speed

    def latency(self):
        return (
            self.estimate_latency(False) - LOOMING_STIMULUS_ONSET
        ) / FRAME_RATE

    def latency_peak_detect(self):
        n_stds = 2.5
        speed = -self.smoothed_x_speed[200:]
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
            return all_peak_starts[0] + 200  # signal.find_peaks(acc, height=0.0002)[0][0]

    def latency_peak_detect_s(self):

        latency_pd = self.latency_peak_detect()
        if latency_pd is not None:
            latency_pd -= 200
            return latency_pd / FRAME_RATE

    def latency_p(self):
        return self.peak_x_acc_idx() - LOOMING_STIMULUS_ONSET

    def estimate_latency(self, smooth=False, limit=600):
        home_front = 0.2 #looming_spots.preprocess.normalisation.normalised_shelter_front(self.context)

        inside_house = self.normalised_x_track[:limit] < home_front

        if smooth:
            speed = self.smoothed_x_speed[:limit]
        else:
            speed = self.normalised_x_speed[:limit]

        towards_house = speed < -0.0001

        starts, ends = get_starts_and_ends(towards_house, 7)

        for s, e in zip(starts, ends):
            if s > LOOMING_STIMULUS_ONSET:
                if s < LOOMING_STIMULUS_ONSET:
                    continue
                elif any(inside_house[s:e]):
                    return s
        print("did not find any starts... attempting with smoothed track")

        if not smooth:
            try:
                return self.estimate_latency(smooth=True) + 5
            except Exception as e:
                print(e)
                return np.nan

    def has_escaped_by(self, sample_n):
        return self.estimate_latency(False) < sample_n

    def time_to_reach_shelter_stim_onset(self):
        n_samples_to_reach_shelter = self.n_samples_to_reach_shelter()
        if n_samples_to_reach_shelter is None:
            return n_samples_to_reach_shelter
        return (n_samples_to_reach_shelter-200) / FRAME_RATE

    def time_to_reach_shelter_from_detection(self):
        n_samples_to_reach_shelter = self.n_samples_to_reach_shelter()
        if n_samples_to_reach_shelter is None:
            return n_samples_to_reach_shelter
        return (n_samples_to_reach_shelter-200-self.reaction_time()) / FRAME_RATE

    @property
    def mouse_location_at_stimulus_onset(self):
        x_track, y_track = self.raw_track
        return x_track[LOOMING_STIMULUS_ONSET], y_track[LOOMING_STIMULUS_ONSET]

    def max_speed(self):
        return np.nanmax(np.abs(self.normalised_x_speed))

    def plot_mouse_location_at_stimulus_onset(self):
        plt.imshow(self.session.get_reference_frame)
        x, y = self.mouse_location_at_stimulus_onset
        plt.plot(x, y, "o", color="k", markersize=20)

    def plot_track(
        self, ax=None, color=None, n_samples_to_show=N_SAMPLES_TO_SHOW
    ):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        if color is None:
            color = "r" if self.is_flee() else "k"
        plt.plot(self.normalised_x_track, color=color)
        plt.ylabel("x position in box (cm)")
        plt.xlabel("time (s)")
        plt.ylim([-0.1, 1])

        track_length = self.get_x_length(ax)

        self.convert_y_axis(0, 1, 0, ARENA_SIZE_CM, n_steps=6)
        self.convert_x_axis(track_length, n_steps=11)
        plt.xlim([0, n_samples_to_show])
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
        if self.habituation_loom_before():
            last_habituation_trial = self.get_last_habituation_trial()
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

    def photometry_peaks(self):
        peaks = []
        starts, ends = self.actual_loom_onsets()
        for start, end in zip(starts, ends):
            peaks.append(max(max(self.delta_f()[start:end]), 0))
        return peaks

    def photometry_sums(self):
        sums = []
        starts, ends = self.actual_loom_onsets()
        for start, end in zip(starts, ends):
            sums.append(max(sum(self.delta_f()[start:end]), 0))
        return sums

    @property
    def event_metric_functions(self):
        func_dict = {  #'peaks': self.photometry_peaks,
            #'sum': self.photometry_sums,
            "integral at latency": self.integral_at_latency,
            "integral at end": self.integral_at_end
            # 'latency': self.photometry_latency,
            # 'duration': self.photometry_duration,
        }

        return func_dict

    def events_df(self):
        metric_dict = {}

        idx = np.arange(len(self.actual_loom_onsets()[0]))
        trial_idx = [self.loom_number] * len(idx)
        peaks = self.photometry_peaks()
        mouse_ids = [self.session.mouse_id] * len(idx)
        sums = self.photometry_sums()

        metric_dict.setdefault("within_stimulus_loom_id", idx)
        metric_dict.setdefault("trial id", trial_idx)
        metric_dict.setdefault("peak", peaks)
        metric_dict.setdefault("sum", sums)
        metric_dict.setdefault("mouse_id", mouse_ids)

        return pd.DataFrame.from_dict(metric_dict)

    def detect_events(self):
        return self.detect_events_scipy(self.delta_f())

    @staticmethod
    def detect_events_scipy(delta_f):
        df = delta_f - apply_butterworth_lowpass_filter(
            delta_f, 0.1, 30, order=8
        )

        pks = signal.find_peaks(df, np.std(df), width=2)  # 1.5 * np.std(df)
        starts = pks[1]["left_ips"].astype(int)
        ends = pks[1]["right_ips"].astype(int)
        peak_locs = pks[0]
        half_rise = np.mean([pks[0], starts], axis=0)
        amplitudes = pks[1]["peak_heights"]

        return starts, ends, peak_locs, half_rise, amplitudes

    def events(self):
        starts, _, peak_locs, half_rise, amplitudes = self.detect_events()
        all_events = []
        for (s, peak_loc, half_rise, amplitude) in zip(
            starts, peak_locs, half_rise, amplitudes
        ):
            e = events.Event(
                s,
                peak_loc,
                half_rise,
                amplitude,
                self.delta_f(),
                sampling_interval=1,
            )
            all_events.append(e)
        return EventsCollection(all_events)

    @cached_property
    def events_trace(self):
        if self.stimulus_type == "auditory":
            mask = np.zeros_like(self.delta_f())
            mask[200:290] = 1
            return mask * self.delta_f()
        return self.events().events_trace(200, 350)  # fixme

    @cached_property
    def cumulative_sum_trace(self):
        return np.cumsum(self.events_trace)

    @cached_property
    def cumulative_sum_raw(self):
        df = self.delta_f_with_pre_stimulus_events_removed()
        df[df < 0] = 0  # remove negative cumsum
        mask = np.zeros_like(df)
        # plt.plot(df)
        mask[200:350] = 1
        cumsum = np.cumsum(df * mask)
        cumsum[350:] = np.nan
        return cumsum

    @cached_property
    def integral(self, end_sample=150):

        end_sample = int(end_sample * 10000 / FRAME_RATE)
        SAMPLING = 10000
        df = self.fully_sampled_delta_f

        mask = np.full(len(df), np.nan)
        start = int(LOOMING_STIMULUS_ONSET * 10000 / FRAME_RATE)
        integral = [
            np.trapz(df[start : start + x], dx=1 / SAMPLING)
            for x in range(end_sample)
        ]

        mask[start : start + end_sample] = integral
        return mask

    def integral_downsampled(self):

        SAMPLING = 30
        end_sample = 150
        df = self.delta_f()

        mask = np.full(len(df), np.nan)
        integral = [
            np.trapz(
                df[LOOMING_STIMULUS_ONSET: LOOMING_STIMULUS_ONSET + x],
                dx=1 / SAMPLING,
            )
            for x in range(end_sample)
        ]

        mask[
            LOOMING_STIMULUS_ONSET : (LOOMING_STIMULUS_ONSET + end_sample)
        ] = integral
        return mask

    def get_delta_f_integral(self, s, e):
        SAMPLING = 30
        end_sample = 150
        df = self.delta_f()

        integral = [
            np.trapz(
                df[s: e + x],
                dx=1 / SAMPLING,
            )
            for x in range(end_sample)
        ]
        return integral

    def get_cumsum(self, scale_factor):
        return self.cumulative_sum_raw / scale_factor

    def plot_delta_f_with_track(self, color, scale_factor=10):
        scale_factor = max(self.delta_f()[:600])
        plt.plot(
            self.delta_f() / scale_factor, #delta_f_with_pre_stimulus_events_removed
            color=color,
            linestyle="--",
        )
        plt.plot(self.normalised_x_track, color=color)
        self.plot_stimulus()
        plt.xlim([0, 600])
        plt.ylim([0, 1])
        plt.hlines(0.5, 500, 530)
        plt.vlines(500, 0.5, 0.5+(0.01/scale_factor))

    def plot_stimulus(self):  # FIXME: duplicated elsewhere
        ax = plt.gca()
        if self.stimulus_type == "auditory":
            patch = patches.Rectangle(
                (200, -0.2), 190, 1.3, alpha=0.1, color="r"
            )
            ax.add_patch(patch)
        else:
            plotting.plot_looms_ax(ax)

    def raw_start(self):
        return photodiode.find_nearest_pd_up_from_frame_number(
            self.session.path, self.start
        )

    def plot_track_and_delta_f(self, axes=None):
        if axes is None:
            fig1 = plt.figure()
            plt.subplot(211)
        else:
            plt.sca(axes[0])

        self.plot_track()
        self.plot_stimulus()
        if axes is None:
            plt.subplot(212)
        else:
            plt.sca(axes[1])
        plt.plot(self.delta_f()[:N_SAMPLES_TO_SHOW])  # TODO: extract nsamplestoshow
        plt.ylim([-0.15, 0.4])
        self.plot_stimulus()
        if axes is None:
            return fig1

    def integral_escape_metric(self, latency=None):
        if latency is not None:
            return self.integral_downsampled()[int(latency)]
        else:
            try:
                return self.integral_downsampled()[
                    self.estimate_latency(False)
                ]
            except Exception as e:
                warnings.warn("returning NaN for escape metric")
                return np.nan

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
            self.context,
            "tz",
            "middle",
            LOOMING_STIMULUS_ONSET,
        )

    def track_overlay(self, duration_in_samples=200, track_heatmap=None):
        if track_heatmap is None:
            track_heatmap = np.zeros((480, 640))  # TODO: get shape from raw

        x, y = (
            np.array(
                self.raw_track[0][
                    LOOMING_STIMULUS_ONSET : LOOMING_STIMULUS_ONSET
                    + duration_in_samples
                ]
            ),
            np.array(
                self.raw_track[1][
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
        return looming_spots.track_analysis.escape_classification.loom_evoked_speed_change(
            self.smoothed_x_speed,
            LOOMING_STIMULUS_ONSET,
            window_before=14,
            window_after=14,
        )[
            0
        ]

    def loom_evoked_speed_after_loom(self):
        return looming_spots.track_analysis.escape_classification.loom_evoked_speed_change(
            self.smoothed_x_speed,
            LOOMING_STIMULUS_ONSET,
            window_before=20,
            window_after=20,
        )[
            1
        ]

    def loom_evoked_speed_change(self):
        if not self.is_flee():
            return (
                self.loom_evoked_speed_before_loom()
                - self.loom_evoked_speed_after_loom()
            )

    def movement_loom_on_vs_loom_off(self):
        if not self.is_flee():
            return looming_spots.track_analysis.escape_classification.movement_loom_on_vs_loom_off(
                self.smoothed_x_speed
            )


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
            path_in, path_out, width=width, height=height, origin=origin, track=self.raw_track
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
