import os
import pathlib
import warnings
from pathlib import Path
from shutil import copyfile

import looming_spots.loom_io.load
import numpy as np
from datetime import datetime

from cached_property import cached_property

import looming_spots.util
from looming_spots.analyse import track_functions

from looming_spots.analyse.track_functions import get_tracking_method
from looming_spots.constants import (
    AUDITORY_STIMULUS_CHANNEL_ADDED_DATE,
    PROCESSED_DATA_DIRECTORY,
)

from looming_spots.db import trial
from looming_spots.loom_io.load import (
    load_all_channels_raw,
)
from looming_spots.exceptions import MouseNotFoundError
from looming_spots.util import generic_functions
from looming_spots.loom_io import load, photodiode


class Session(object):

    """
    The Session class aims to load data that has been acquired in a series of recording sessions and provide
    a simple means to read and access this data, trials are generated from this data and have access to the data
    provided.

    """

    def __init__(
        self,
        dt,
        mouse_id=None,
        n_looms_to_view=0,
        n_lse_looms=120,
        n_trials_to_include=3,
    ):
        self.dt = dt
        self.mouse_id = mouse_id
        self.n_looms_to_view = n_looms_to_view
        self.n_lse_looms = n_lse_looms
        self.n_trials_to_include = n_trials_to_include
        self.next_session = None
        self.previous_session = None

    @property
    def frame_rate(self):
        """
        The camera frame rate is controlled by a clock TTL pulse that is recorded along with all the other data, and this
        clock can be used to infer the frame rate used.

        :return:
        """
        p = pathlib.Path(self.path)
        frame_rate_file_exists = len(list(p.glob("frame_rate.npy"))) == 1

        if frame_rate_file_exists:
            frame_rate = np.load(str(p / "frame_rate.npy"))
        else:
            if "AI.tdms" in os.listdir(self.path):
                clock = self.get_clock_raw()
                clock_ups = looming_spots.loom_io.load.get_clock_ups(clock)
                if np.nanmedian(np.diff(clock_ups)) == 333:
                    frame_rate = 30
                elif np.nanmedian(np.diff(clock_ups)) == 200:
                    frame_rate = 50
                elif np.nanmedian(np.diff(clock_ups)) == 100:
                    frame_rate = 100
            else:
                frame_rate = 30
            np.save(str(p / "frame_rate.npy"), frame_rate)

        return frame_rate

    def __len__(self):
        return len(self.data["photodiode"])

    def __lt__(self, other):
        """
        Allows sorting trials chronologically

        :param other:
        :return:
        """
        return self.dt < other.dt

    def __gt__(self, other):
        """
        Allows sorting trials chronologically

        :param other:
        :return:
        """
        return self.dt > other.dt

    def __add__(self, a):
        for t in self.trials:
            t + a

    # def directory(self):
    #     return os.path.join(PROCESSED_DATA_DIRECTORY, self.mouse_id)

    @property
    def path(self):
        parent_dir = self.mouse_id
        session_dir = datetime.strftime(self.dt, "%Y%m%d_%H_%M_%S")
        return os.path.join(PROCESSED_DATA_DIRECTORY, parent_dir, session_dir)

    def contains_auditory(self):
        aud_idx_path = os.path.join(self.path, "auditory_starts.npy")
        if os.path.isfile(aud_idx_path):
            if len(np.load(aud_idx_path)) > 0:
                return True

        recording_date = datetime.strptime(
            os.path.split(self.path)[-1], "%Y%m%d_%H_%M_%S"
        )

        if "AI.tdms" in os.listdir(self.path):
            return True

        if recording_date > AUDITORY_STIMULUS_CHANNEL_ADDED_DATE:
            ad = looming_spots.loom_io.load.load_all_channels_on_clock_ups(self.path)[
                "auditory_stimulus"
            ]
            if (ad > 0.7).any():
                return True

    @cached_property
    def trials(self):
        visual_trials_idx = self.get_looming_stimuli_test_trials_idx()
        auditory_trials_idx = self.get_auditory_trials_idx()
        visual_trials = self.initialise_trials(
            visual_trials_idx,
            "visual",
        )
        auditory_trials = self.initialise_trials(
            auditory_trials_idx,
            "auditory",
        )

        return sorted(visual_trials + auditory_trials)

    def initialise_trials(self, idx, stimulus_type):
        if idx is not None:
            if len(idx) > 0:
                trials = []
                for i, onset_in_samples in enumerate(idx):
                    if stimulus_type == "visual":
                        t = trial.VisualStimulusTrial(
                            self,
                            directory=self.path,
                            sample_number=onset_in_samples,
                            trial_type=self.get_trial_type(onset_in_samples),
                            stimulus_type="loom",
                        )
                    elif stimulus_type == "auditory":
                        t = trial.AuditoryStimulusTrial(
                            self,
                            directory=self.path,
                            sample_number=onset_in_samples,
                            trial_type=self.get_trial_type(onset_in_samples),
                            stimulus_type="auditory",
                        )

                    else:
                        raise NotImplementedError

                    trials.append(t)
                return trials
            else:
                return []
        return []

    def get_trials_of_protocol_type(self, key):
        """
        Returns trials grouped as either belonging to an LSE protocol, or being a testing trial of some sort
        to later be more specifically classified as a pre- or post- lse testing trial.

        :param key:
        :return:
        """
        if key == "test":
            return [t for t in self.trials if "test" in t.trial_type]

        return [t for t in self.trials if t.trial_type == key]

    @property
    def photodiode_trace(self, raw=False):
        if raw:
            pd = load_all_channels_raw(self.path)["photodiode"]
        else:
            pd = self.data["photodiode"]
        return pd

    def get_clock_raw(self):
        return load_all_channels_raw(self.path)["clock"]

    @cached_property
    def data(self):
        return load.load_all_channels_on_clock_ups(self.path)

    @cached_property
    def looming_stimuli_idx(self):
        loom_idx_path = os.path.join(self.path, "loom_starts.npy")
        if not os.path.isfile(loom_idx_path):
            _ = photodiode.get_loom_idx_from_photodiode_trace(self.path, save=True)[0]
        return np.load(loom_idx_path)

    @cached_property
    def auditory_stimuli_idx(self):
        if self.contains_auditory():
            aud_idx_path = os.path.join(self.path, "auditory_starts.npy")
            if not os.path.isfile(aud_idx_path):
                _ = photodiode.get_auditory_onsets_from_auditory_trace(
                    self.path, save=True
                )
            return np.load(aud_idx_path)

    def get_trial_type(self, onset_in_samples):
        if self.lse_idx is None:
            return "test"
        elif onset_in_samples in self.lse_idx:
            return "lse"
        else:
            return "test"

    @property
    def lse_idx(self):
        if self.contains_auditory():
            if photodiode.contains_lse(self.auditory_stimuli_idx, 1):
                return photodiode.get_lse_loom_idx(self.auditory_stimuli_idx, 1)

        if photodiode.contains_lse(self.looming_stimuli_idx, 5):
            return photodiode.get_lse_loom_idx(self.looming_stimuli_idx, 5)

    @property
    def lse_loom_idx(self):
        return photodiode.get_lse_loom_idx(self.looming_stimuli_idx)

    @property
    def test_loom_idx(self):
        return photodiode.get_test_looms_from_loom_idx(self.looming_stimuli_idx)

    def get_looming_stimuli_test_trials_idx(self):
        return self.looming_stimuli_idx[::5]

    def get_auditory_trials_idx(self):
        return self.auditory_stimuli_idx

    def get_data(self, key):
        data_func_dict = {
            "photodiode": self.get_photodiode_trace,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
            "loom_onsets": self.get_loom_idx,
            "auditory_onsets": self.get_auditory_idx,
            "trials": self.get_session_trials,
        }

        return data_func_dict[key]

    def get_photodiode_trace(self):
        return self.photodiode_trace - np.median(self.photodiode_trace)

    def track(self):
        return track_functions.track_in_standard_space(
            self.path, get_tracking_method(self.path), 0, len(self), None
        )

    def x_pos(self):
        return self.track()[0]

    def y_pos(self):
        return self.track()[1]

    def get(self, key):
        current_session = self
        prev_samples = 0

        selection_dict = {
            "loom_idx": self.looming_stimuli_idx,
            "auditory_idx": self.auditory_stimuli_idx,
            "trials": self.trials,
        }

        while current_session is not None:
            print(f"prev_samples: {prev_samples}")
            current_session = current_session.previous_session
            if current_session is not None:
                prev_samples += len(current_session)

        if key == "trials":
            self + prev_samples
            return self.trials

        return selection_dict[key] + prev_samples

    def get_loom_idx(self):
        return self.get("loom_idx")

    def get_auditory_idx(self):
        return self.get("auditory_idx")

    def get_session_trials(self):
        return self.get("trials")

    @classmethod
    def set_next_session(cls, self, other):
        setattr(self, "next_session", other)

    @classmethod
    def set_previous_session(cls, self, other):
        setattr(self, "previous_session", other)


def load_sessions(mouse_id):
    mouse_directory = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    print(f"loading.... {mouse_directory}")
    session_list = []
    if os.path.isdir(mouse_directory):

        for s in os.listdir(mouse_directory):

            session_directory = os.path.join(mouse_directory, s)
            if not os.path.isdir(session_directory):
                continue

            file_names = os.listdir(session_directory)

            if not contains_analog_input(file_names):
                continue

            if "contrasts.mat" not in s:
                print("no contrasts mat")

                if not os.path.isdir(session_directory):
                    continue

                if not looming_spots.util.generic_functions.is_datetime(s):
                    print("not datetime, skipping")
                    continue

                if not contains_video(file_names) and not contains_tracks(
                    session_directory
                ):
                    print("no video or tracks")
                    if not get_tracks_from_raw(
                        mouse_directory.replace("processed_data", "raw_data")
                    ):
                        continue

            date = datetime.strptime(s, "%Y%m%d_%H_%M_%S")
            s = Session(dt=date, mouse_id=mouse_id)
            session_list.append(s)

        if len(session_list) == 0:
            msg = f"the mouse: {mouse_id} has not been processed"
            raise MouseNotFoundError(msg)

        return sorted(session_list)
    msg = f"the mouse: {mouse_id} has not been copied to the processed data directory"
    warnings.warn(msg)

    raise MouseNotFoundError()


def contains_analog_input(file_names):
    if "AI.bin" in file_names or "AI.tdms" in file_names:
        return True
    return False


def contains_video(file_names):
    return any(".avi" in fname for fname in file_names) or any(
        ".mp4" in fname for fname in file_names
    )


def contains_tracks(session_directory):
    p = pathlib.Path(session_directory)
    if len(list(p.rglob("dlc_x_tracks.npy"))) == 0:
        return False
    else:
        return True


def get_tracks_from_raw(directory):
    print(f"getting tracks from {directory}")
    p = Path(directory)
    track_paths = p.rglob("*tracks.npy")
    if len(list(p.rglob("*tracks.npy"))) == 0:
        print("no track paths found...")
        return False

    for tp in track_paths:
        raw_path = str(tp)
        processed_path = raw_path.replace("raw_data", "processed_data")
        print(f"copying {raw_path} to {processed_path}")
        copyfile(raw_path, processed_path)
    return True
