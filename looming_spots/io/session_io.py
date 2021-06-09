import os
import pathlib
import warnings
from pathlib import Path
from shutil import copyfile

import numpy as np
from datetime import datetime

import scipy
from cached_property import cached_property
from nptdms import TdmsFile

import looming_spots.io.io
import looming_spots.util
from looming_spots.db.constants import (
    AUDITORY_STIMULUS_CHANNEL_ADDED_DATE,
    PROCESSED_DATA_DIRECTORY,
    CONTEXT_B_SPOT_POSITION,
)

from looming_spots.db import loom_trial
from looming_spots.exceptions import LoomsNotTrackedError, MouseNotFoundError

from looming_spots.preprocess import photodiode, normalisation
from looming_spots.track_analysis import arena_region_crossings
from looming_spots.tracking_dlc import process_DLC_output
from looming_spots.util import generic_functions
from photometry import demodulation, load


class Session(object):
    def __init__(
        self,
        dt,
        mouse_id=None,
        n_looms_to_view=0,
        n_habituation_looms=120,
        n_trials_to_consider=3,

    ):
        self.dt = dt
        self.mouse_id = mouse_id
        self.n_looms_to_view = n_looms_to_view
        self.n_habituation_looms = n_habituation_looms
        self.n_trials_to_include = n_trials_to_consider
        self.next_session = None
        self.previous_session = None

    @property
    def frame_rate(self):
        clock = self.get_clock_raw()
        clock_ups = looming_spots.io.io.get_clock_ups(clock)
        if np.nanmedian(np.diff(clock_ups)) == 333:
            return 30
        elif np.nanmedian(np.diff(clock_ups)) == 200:
            return 50
        elif np.nanmedian(np.diff(clock_ups)) == 100:
            return 100

    def __len__(self):
        return len(self.data["photodiode"])

    def __lt__(self, other):
        return self.dt < other.dt

    def __gt__(self, other):
        return self.dt > other.dt

    def __add__(self, a):
        for t in self.trials:
            t + a

    def directory(self):
        return os.path.join(PROCESSED_DATA_DIRECTORY, self.mouse_id)

    @property
    def path(self):
        parent_dir = self.mouse_id
        session_dir = datetime.strftime(self.dt, "%Y%m%d_%H_%M_%S")
        return os.path.join(PROCESSED_DATA_DIRECTORY, parent_dir, session_dir)

    @property
    def video_path(self):
        return os.path.join(self.path.replace('processed_data', 'raw_data'), "camera.avi")

    @property
    def parent_path(self):
        return os.path.split(self.path)[0]

    @property
    def loom_paths(self):
        paths = []

        if self.n_looms == 0:
            print(f"no looms in {self.path}")
            return []

        for name in os.listdir(self.path):
            loom_folder = os.path.join(self.path, name)
            if os.path.isdir(loom_folder) and "loom" in name:
                paths.append(loom_folder)

        if len(paths) == 0:
            raise LoomsNotTrackedError(self.path)

        loom_indices = [int(path[-1]) for path in paths]
        sorted_paths, idx = generic_functions.sort_by(
            paths, loom_indices, descend=False
        )

        return sorted_paths

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
            ad = looming_spots.io.io.load_auditory_on_clock_ups(self.path)
            if (ad > 0.7).any():
                return True

    @cached_property
    def trials(self):
        visual_trials_idx = self.get_visual_trials_idx()
        auditory_trials_idx = self.get_auditory_trials_idx()
        cricket_trials_idx = self.get_cricket_trials_idx()
        visual_trials = self.initialise_trials(visual_trials_idx, "visual",)
        auditory_trials = self.initialise_trials(
            auditory_trials_idx, "auditory",
        )
        cricket_trials = self.initialise_trials(cricket_trials_idx, 'cricket')

        return sorted(visual_trials + auditory_trials + cricket_trials)

    @cached_property
    def frame_rate(self):
        return get_frame_rate(self.directory())

    def initialise_trials(self, idx, stimulus_type):
        if idx is not None:
            if len(idx) > 0:
                trials = []
                for i, onset_in_samples in enumerate(idx):
                    if stimulus_type == "visual":
                        t = loom_trial.VisualStimulusTrial(
                            self,
                            directory=self.path,
                            sample_number=onset_in_samples,
                            trial_type=self.get_trial_type(onset_in_samples),
                            stimulus_type="loom", frame_rate=self.frame_rate
                        )
                    elif stimulus_type == "auditory":
                        t = loom_trial.AuditoryStimulusTrial(
                            self,
                            directory=self.path,
                            sample_number=onset_in_samples,
                            trial_type=self.get_trial_type(onset_in_samples),
                            stimulus_type="auditory",
                            frame_rate=self.frame_rate,
                        )

                    elif stimulus_type == "cricket":
                        t = loom_trial.CricketStimulusTrial(
                            self,
                            directory=self.path,
                            sample_number=onset_in_samples,
                            trial_type=self.get_trial_type(onset_in_samples),
                            stimulus_type="cricket",
                            frame_rate=self.frame_rate,
                        )

                    else:
                        raise NotImplementedError

                    # t.time_to_first_loom = self.time_to_first_loom()
                    trials.append(t)
                return trials
            else:
                return []
        return []

    def get_trials(self, key):

        if key == "test":
            return [t for t in self.trials if "test" in t.trial_type]

        return [t for t in self.trials if t.trial_type == key]

    @property
    def context(self):  # FIXME: this is BS
        try:
            return get_context_from_stimulus_mat(self.path)
        except FileNotFoundError as e:
            print(e)
            print("guessing A10")
            return "A"

    @property
    def contrast_protocol(self):  # TEST: FIXME:
        pd = self.photodiode_trace
        n_samples = 500
        pre_start = self.trials[0].sample_number - 600
        pre_end = pre_start + n_samples
        post_start = self.trials[0].sample_number + 500
        post_end = post_start + n_samples

        baseline_before_protocol = np.median(pd[pre_start:pre_end])
        baseline_during_protocol = np.median(pd[post_start:post_end])

        if (
            baseline_before_protocol - baseline_during_protocol > 0.01
        ):  # FIXME: hard code
            return "gradient"
        else:
            return "constant"

    @property
    def trials_results(self):
        test_trials = [t for t in self.trials if "test" in t.trial_type]
        return np.array(
            [t.is_flee() for t in test_trials[: self.n_trials_to_include]]
        )

    @property
    def n_trials_total(self):
        return len(self.trials)

    @property
    def n_test_trials(self):
        return len(self.get_trials("test"))

    @property
    def n_habituation_trials(self):
        return len(self.get_trials("habituation"))

    def hours(self):
        return self.dt.hour + self.dt.minute / 60

    @property
    def n_flees(self):
        return np.count_nonzero(self.trials_results)

    @property
    def n_non_flees(self):
        n_trials = len(self.trials_results)
        return n_trials - np.count_nonzero(self.trials_results)

    @property
    def n_looms(self):
        return len(self.loom_idx)

    def get_reference_frame(self, idx=0):
        import skvideo.io
        vid = skvideo.io.vreader(self.video_path)
        for i, frame in enumerate(vid):
            if i == idx:
                return frame


    @property
    def photodiode_trace(self, raw=False):
        if raw:
            pd, clock = looming_spots.io.io.load_pd_and_clock_raw(self.path)
        else:
            pd = self.data["photodiode"]
        return pd

    def get_clock_raw(self):
        _, clock, _ = looming_spots.io.io.load_pd_and_clock_raw(self.path)
        return clock

    @property
    def auditory_trace(self, raw=False):
        if raw:
            ad = looming_spots.io.io.load_auditory_on_clock_ups(self.path)
        else:
            ad = self.data["auditory_stimulus"]
        return ad

    def load_demodulated_photometry(self):
        pd, _, auditory, photometry, led211, led531 = load.load_all_channels_raw(
            self.path
        )
        dm_signal, dm_background = demodulation.demodulate(
            photometry, led211, led531
        )
        if (
            self.path
            == "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/074744/20190404_20_33_34"
        ):
            print("flipping channels")
            dm_signal, dm_background = dm_background, dm_signal
        return dm_signal, dm_background

    @cached_property
    def signal(self):
        return self.data["signal"]

    @cached_property
    def background(self):
        return self.data["background"]

    @cached_property
    def fully_sampled_delta_f(self, filter_artifact_cutoff_samples=40000):
        pd, clock, auditory, pmt, led211, led531 = load.load_all_channels_raw(
            self.path
        )

        if (
            self.path
            == "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/074744/20190404_20_33_34"
        ):
            print("flipping channels")
            led211, led531 = led531, led211

        pmt[:filter_artifact_cutoff_samples] = np.median(pmt)
        delta_f = demodulation.lerner_deisseroth_preprocess(
            pmt, led211, led531
        )
        delta_f[:filter_artifact_cutoff_samples] = np.median(delta_f)
        return delta_f, clock

    @cached_property
    def data(self):
        return load.load_all_channels_on_clock_ups(self.path)

    @property
    def contains_habituation(self):
        return photodiode.contains_habituation(self.loom_idx)

    @cached_property
    def loom_idx(self):
        if self.path == '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/processed_data/1114179/20210415_12_41_00/':
            return []
        loom_idx_path = os.path.join(self.path, "loom_starts.npy")
        if not os.path.isfile(loom_idx_path):
            _ = photodiode.get_loom_idx_from_raw(self.path, save=True)[0]
        return np.load(loom_idx_path)

    @cached_property
    def auditory_idx(self):
        if self.contains_auditory():
            aud_idx_path = os.path.join(self.path, "auditory_starts.npy")
            if not os.path.isfile(aud_idx_path):
                _ = photodiode.get_auditory_onsets_from_analog_input(
                    self.path, save=True
                )
            return np.load(aud_idx_path)

    def get_trial_type(self, onset_in_samples):
        if self.habituation_idx is None:
            return "test"
        elif onset_in_samples in self.habituation_idx:
            return "habituation"
        else:
            return "test"

    @property
    def habituation_idx(self):
        if self.contains_auditory():
            if photodiode.contains_habituation(self.auditory_idx, 1):
                return photodiode.get_habituation_idx(self.auditory_idx, 1)

        if photodiode.contains_habituation(self.loom_idx, 5):
            return photodiode.get_habituation_idx(self.loom_idx, 5)

    @property
    def habituation_loom_idx(self):
        return photodiode.get_habituation_loom_idx(self.loom_idx)

    @property
    def test_loom_idx(self):
        return photodiode.get_manual_looms(self.loom_idx)

    @property
    def habituation_protocol_start(self):
        return photodiode.get_habituation_start(self.loom_idx)

    def test_loom_classification(self):  # TEST
        assert len(self.habituation_loom_idx) + len(self.test_loom_idx) == int(
            len(self.loom_idx) / 5
        )  # FIXME: hard code

    def extract_trials(self):
        for t in self.trials:
            t.extract_video()

    def contains_visual(self):
        pd = looming_spots.io.io.load_pd_on_clock_ups(self.path)
        if (pd > 0.5).any():
            return True

    def get_visual_trials_idx(self):
        return self.loom_idx[::5]

    def get_auditory_trials_idx(self):
        return self.auditory_idx

    def get_data(self, key):
        data_func_dict = {
            "photodiode": self.get_photodiode_trace,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
            "delta_f": self.get_delta_f,
            "loom_onsets": self.get_loom_idx,
            "auditory_onsets": self.get_auditory_idx,
            "x_pos_norm": self.get_normalised_x_pos,
            "trials": self.get_session_trials,
        }

        return data_func_dict[key]

    def get_photodiode_trace(self):
        return self.photodiode_trace - np.median(self.photodiode_trace)

    def track(self):
        p = pathlib.Path(self.path)
        if len(list(p.rglob('dlc_x_tracks.npy'))) > 0:
            x_path = str(list(p.rglob("dlc_x_tracks.npy"))[0])
            y_path = str(list(p.rglob("dlc_y_tracks.npy"))[0])
            x = np.load(x_path)
            y = np.load(y_path)
            if len(x) - len(self) > 5:
                warning_text = f"noticed a mismatch between track length: {len(x)} and session data: {len(self)}"
                warnings.warn(warning_text)

            return x[: len(self)], y[: len(self)]

    def x_pos(self):
        return self.track()[0]

    def y_pos(self):
        return self.track()[1]

    def get_delta_f(self):
        return self.data["delta_f"]

    def get(self, key):
        current_session = self
        prev_samples = 0

        selection_dict = {
            "loom_idx": self.loom_idx,
            "auditory_idx": self.auditory_idx,
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

    def get_normalised_x_pos(self):
        return normalisation.normalise_x_track(self.x_pos(), self.context)

    def get_cricket_trials_idx(self):
        mouse_x, _ = self.get_all_bodyparts()
        if mouse_x is None:
            return []
        entries = arena_region_crossings.get_all_entries(mouse_x)
        return entries

    def get_all_bodyparts(self):
        import pandas as pd
        p = list(pathlib.Path(self.path).rglob('cameraDLC_resnet50_cricketsApr7shuffle1_1030000filtered.h5'))
        if len(p) > 0:
            p=p[0]
        if os.path.isfile((str(p))):
            df = pd.read_hdf(p)

            df = df[df.keys()[0][0]]
            if np.count_nonzero(df['cricket']['likelihood'] == 1) < 2000:
                return None, None
        else:
            return None,None

        start = process_DLC_output.get_first_and_last_likely_frame(df, 'cricket')
        print(f'start: {start}')
        df = process_DLC_output.replace_low_likelihood_as_nan(df)

        body_part_labels = ['body', 'cricket']
        body_parts = {body_part_label: df[body_part_label] for body_part_label in body_part_labels}

        df_y = pd.DataFrame({body_part_label: body_part["y"] for body_part_label, body_part in body_parts.items()})
        df_x = pd.DataFrame({body_part_label: body_part["x"] for body_part_label, body_part in body_parts.items()})
        body_x = 1 - (df_x['body'] / 600)
        body_y = 1 - (df_y['body'] / 200)

        cricket_x = 1 - (df_x['cricket'] / 600)
        cricket_y = 1 - (df_y['cricket'] / 200)

        body_x[:start] = 0
        body_y[:start] = 0

        cricket_x[:start] = 0
        cricket_y[:start] = 0
        return body_x, cricket_x


def get_context_from_stimulus_mat(directory):
    stimulus_path = os.path.join(directory, "stimulus.mat")
    if os.path.isfile(stimulus_path):
        stimulus_params = scipy.io.loadmat(stimulus_path)["params"]
        dot_locations = [
            x[0] for x in stimulus_params[0][0] if len(x[0]) == 2
        ]  # only spot position has length 2

        return (
            "B"
            if any(CONTEXT_B_SPOT_POSITION in x for x in dot_locations)
            else "A"
        )
    else:
        return "A10"


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

                if not contains_video(file_names) and not contains_tracks(session_directory):
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
    p=pathlib.Path(session_directory)
    if len(list(p.rglob("dlc_x_tracks.npy"))) ==0:
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


def get_frame_rate(directory):

    tdms_path = os.path.join(directory, 'AI.tdms')
    if not os.path.exists(tdms_path):
        frame_rate = 30
    else:
        tdms_file = TdmsFile(tdms_path)
        tdms_groups = tdms_file.groups()
        all_channels = tdms_groups[0].channels()
        clock = all_channels[1].data
        clock_on = (clock > 2.5).astype(int)
        ni_sample_rate = 1/all_channels[1].properties['wf_increment']
        clock_ups = np.where(np.diff(clock_on) == 1)[0]
        frame_rate = round(np.mean(ni_sample_rate/np.diff(clock_ups)))

    return frame_rate
