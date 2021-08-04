import os
import pathlib

import looming_spots.util.generate_example_videos
import numpy as np
import matplotlib.pyplot as plt
import pims
from looming_spots.analyse.escape_classification import is_track_a_freeze
from looming_spots.analyse.tracks import get_loom_number_from_latency
from looming_spots.db.track import Track
from looming_spots.util.generic_functions import pad_track

from matplotlib import patches
from datetime import timedelta

import looming_spots.util.video_processing

from looming_spots.constants import (
    LOOMING_STIMULUS_ONSET,
    N_LOOMS_PER_STIMULUS,
    N_SAMPLES_BEFORE,
    ARENA_LENGTH_PX,
    ARENA_WIDTH_PX,
    ARENA_SIZE_CM,
    FRAME_RATE,
    TRACK_LENGTH,
)

from looming_spots.util import plotting
import pandas as pd


class LoomTrial(object):

    """
    The aim of the LoomTrial class is to associate all trial-level metadata and data into one place.
    A trial is defined as a data window surrounding the presentation of a stimulus, as detected using a photodiode
    pulse presented in conjunction with a stimulus.


    This includes:

    - where the data comes from (i.e. the session, see session_io)
    - stimulus metadata (e.g. contrast, time, type)
    - trial data (e.g. the positional xy tracks (see db.track), photodiode traces and other coaquired data)

    """

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
        self.frame_rate = self.session.frame_rate
        self.n_samples_before = int(N_SAMPLES_BEFORE / 30 * self.frame_rate)
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
            if current_trial.trial_type == "lsie":
                return True
            current_trial = current_trial.next_trial

    def lsie_loom_before(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "lsie":
                return True
            current_trial = current_trial.previous_trial

    def get_last_lsie_trial(self):
        current_trial = self
        while current_trial is not None:
            if current_trial.trial_type == "lsie":
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
        n_lsie_protocols = 1 if current_trial_type == "lsie" else 0

        while current_trial is not None:
            if current_trial.trial_type != current_trial_type:
                n_lsie_protocols += 1
                current_trial_type = current_trial.trial_type
            current_trial = current_trial.next_trial
        return n_lsie_protocols

    def first_trial(self):
        current_trial = self
        while current_trial.previous_trial is not None:
            current_trial = current_trial.previous_trial
        return current_trial

    def get_trial_type(self):
        if self.trial_type == "lsie":
            return "lsie"
        elif self.lsie_loom_before() and self.lsie_loom_after():
            return "post_test"  # TODO:'pre_and_post_test'
        elif self.lsie_loom_after():
            return "pre_test"
        elif self.lsie_loom_before():
            return "post_test"
        else:
            return "pre_test"

    @property
    def track(self):
        return Track(
            self.folder,
            self.session.path,
            self.start,
            self.end,
            self.frame_rate,
        )

    def stimulus_number(self):
        if self.stimulus_type == "loom":
            return self.loom_number
        elif self.stimulus_type == "auditory":
            return self.auditory_number

    @property
    def loom_number(self):
        if self.sample_number in self.session.looming_stimuli_idx:
            return int(
                np.where(self.session.looming_stimuli_idx == self.sample_number)[0][0]
                / N_LOOMS_PER_STIMULUS
            )

    @property
    def auditory_number(self):
        if self.sample_number in self.session.auditory_stimuli_idx:
            return int(
                np.where(self.session.auditory_stimuli_idx == self.sample_number)[0][0]
            )

    def get_stimulus_number(self):
        return self.stimulus_number()

    def processed_video_path(self):
        return list(pathlib.Path(self.directory).glob("cam_transform*.mp4"))[0]

    def extract_video(self, overwrite=False):
        print("extracting")
        if not overwrite:
            if os.path.isfile(self.video_path):
                return "video already exists... skipping"
        looming_spots.util.generate_example_videos.extract_loom_video_trial(
            self.processed_video_path(),
            str(pathlib.Path(self.directory) / self.video_name),
            self.sample_number,
            overwrite=overwrite,
        )

    def photodiode(self):
        auditory_signal = self.session.data["photodiode"]
        return auditory_signal[self.start: self.end]

    def auditory_data(self):
        auditory_signal = self.session.data["auditory_stimulus"]
        return auditory_signal[self.start: self.end]

    @property
    def time(self):
        return self.session.dt + timedelta(
            0, int(self.sample_number / self.frame_rate)
        )

    def get_time(self):
        return self.time

    def time_in_safety_zone(self):
        return self.track.time_in_safety_zone()

    @property
    def mouse_location_at_stimulus_onset(self):
        x_track, y_track = self.track.track_in_standard_space
        return x_track[LOOMING_STIMULUS_ONSET], y_track[LOOMING_STIMULUS_ONSET]

    @staticmethod
    def round_timedelta(td):
        if td.seconds > 60 * 60 * 12:
            return td + timedelta(1)
        else:
            return td

    def plot_stimulus(self):
        ax = plt.gca()
        if self.stimulus_type == "auditory":
            patch = patches.Rectangle(
                (self.n_samples_before, -0.2), 190, 1.3, alpha=0.1, color="r"
            )
            ax.add_patch(patch)
        else:
            plotting.plot_looms_ax(ax)

    def plot_mouse_location_at_stimulus_onset(self):
        ref_frame = self.session.get_reference_frame(idx=self.sample_number)
        plt.imshow(ref_frame)
        x, y = self.mouse_location_at_stimulus_onset
        plt.plot(x, y, "o", color="k", markersize=20)

    def to_df(self, group_id):
        n_points = TRACK_LENGTH
        track = pad_track(
            ARENA_SIZE_CM * self.track.normalised_x_track[0:n_points], n_points
        )
        unsmoothed_speed = pad_track(
            FRAME_RATE
            * ARENA_SIZE_CM
            * self.track.normalised_x_speed[0:n_points],
            n_points,
        )
        smoothed_speed = pad_track(
            FRAME_RATE
            * ARENA_SIZE_CM
            * self.track.smoothed_x_speed[0:n_points],
            n_points,
        )
        add_dict = {
            "group_id": group_id,
            "mouse_id": self.mouse_id,
            "track": [track],
            "speed": [smoothed_speed],
            "peak_speed": self.track.peak_speed(),
            "is_flee": self.track.is_escape(),
            "latency": self.track.latency(),
            "last_loom": get_loom_number_from_latency(self.track.latency()),
            "is_freeze": is_track_a_freeze(unsmoothed_speed),
            "time_to_shelter": self.track.time_to_shelter(),
        }
        this_trial_df = pd.DataFrame.from_dict(add_dict)
        return this_trial_df


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
        self, width=ARENA_LENGTH_PX, height=ARENA_WIDTH_PX, origin=(0, 40)
    ):

        path_in = str(pathlib.Path(self.directory) / self.video_name)
        path_out = "_overlay.".join(path_in.split("."))

        looming_spots.util.generate_example_videos.loom_superimposed_video(
            path_in,
            path_out,
            width=width,
            height=height,
            origin=origin,
            track=self.track.track_in_standard_space,
        )

    def get_video(self):
        vid_path = pathlib.Path(
            self.session.path.replace("processed", "raw")
        ).joinpath("camera.avi")
        vid = pims.Video(vid_path)
        return vid[self.start : self.end]


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
