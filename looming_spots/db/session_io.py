import os
import warnings

import numpy as np
from datetime import datetime
import scipy.ndimage
from cached_property import cached_property
from matplotlib import pyplot as plt

from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY

from looming_spots.db import loomtrial
from looming_spots.exceptions import LoomsNotTrackedError

from looming_spots.preprocess import photodiode
from looming_spots.util import generic_functions
from looming_spots.db.metadata import experiment_metadata


class Session(object):

    def __init__(self, dt, mouse_id=None, n_looms_to_view=0, n_habituation_looms=120, n_trials_to_consider=3):
        self.dt = dt
        self.mouse_id = mouse_id
        self.n_looms_to_view = n_looms_to_view
        self.n_habituation_looms = n_habituation_looms
        self.n_trials_to_include = n_trials_to_consider
        self.next_session = None

    def __lt__(self, other):
        return self.dt < other.dt

    def __gt__(self, other):
        return self.dt > other.dt

    def directory(self):
        return os.path.join(PROCESSED_DATA_DIRECTORY, self.mouse_id)

    @property
    def path(self):
        parent_dir = self.mouse_id
        session_dir = datetime.strftime(self.dt, '%Y%m%d_%H_%M_%S')
        return os.path.join(PROCESSED_DATA_DIRECTORY, parent_dir, session_dir)

    @property
    def video_path(self):
        return os.path.join(self.path, 'camera.mp4')

    @property
    def parent_path(self):
        return os.path.split(self.path)[0]

    @property
    def loom_paths(self):
        paths = []

        if self.n_looms == 0:
            print('no looms in {}'.format(self.path))
            return []

        for name in os.listdir(self.path):
            loom_folder = os.path.join(self.path, name)
            if os.path.isdir(loom_folder) and 'loom' in name:
                paths.append(loom_folder)

        if len(paths) == 0:
            raise LoomsNotTrackedError(self.path)

        loom_indices = [int(path[-1]) for path in paths]
        sorted_paths, idx = generic_functions.sort_by(paths, loom_indices, descend=False)

        return sorted_paths

    @property
    def trials(self):
        trials = []
        for i, loom_onset_in_samples in enumerate(self.loom_idx[::5]):
            t = loomtrial.LoomTrial(self, directory=self.path, sample_number=loom_onset_in_samples,
                                    trial_type=self.get_trial_type(loom_onset_in_samples))
            t.time_to_first_loom = self.time_to_first_loom()
            trials.append(t)
        return trials

    def get_trials(self, key):

        if key == 'test':
            return [t for t in self.trials if 'test' in t.trial_type]

        return [t for t in self.trials if t.trial_type == key]

    @property
    def context(self):
        return experiment_metadata.get_context_from_stimulus_mat(self.path)

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

        if baseline_before_protocol - baseline_during_protocol > 0.01:  # FIXME: hard code
            return 'gradient'
        else:
            return 'constant'

    @property
    def manual_loom_idx(self):
        return self.metadata['manual_loom_idx']

    @property
    def trials_results(self):
        test_trials = [t for t in self.trials if 'test' in t.trial_type]
        return np.array([t.is_flee() for t in test_trials[:self.n_trials_to_include]])

    @property
    def n_trials_total(self):
        return len(self.trials)

    @property
    def n_test_trials(self):
        return len(self.get_trials('test'))

    @property
    def n_habituation_trials(self):
        return len(self.get_trials('habituation'))

    def hours(self):
        return self.dt.hour + self.dt.minute/60

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

    def get_reference_frame(self, trial_type):
        fpath = os.path.join(self.path, '{}_ref.npy'.format(trial_type))
        if os.path.isfile(fpath):
            return np.load(fpath)

    @property
    def photodiode_trace(self, raw=False):
        if raw:
            pd, clock = photodiode.load_pd_and_clock_raw(self.path)
        else:
            pd = photodiode.load_pd_on_clock_ups(self.path)
        return pd

    @property
    def contains_habituation(self):
        return photodiode.contains_habituation(self.loom_idx)

    @cached_property
    def loom_idx(self):
        loom_idx_path = os.path.join(self.path, 'loom_starts.npy')
        if not os.path.isfile(loom_idx_path):
            _ = photodiode.get_loom_idx_from_raw(self.path, save=True)[0]
        return np.load(loom_idx_path)

    def get_trial_type(self, loom_onset_in_samples):
        if self.habituation_loom_idx is None:
            return 'test'
        elif loom_onset_in_samples in self.habituation_loom_idx:
            return 'habituation'
        else:
            return 'test'

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
        assert len(self.habituation_loom_idx) + len(self.test_loom_idx) == int(len(self.loom_idx)/5)  # FIXME: hard code

    def time_to_first_loom(self):
        if len(self.loom_idx) > 0:
            if 'time_of_mouse_entry' in self.metadata:
                return (int(self.loom_idx[0]) - int(self.metadata['time_of_mouse_entry']))/30/60
            return int(self.loom_idx[0])/30/60

    def extract_trials(self):
        for t in self.trials:
            t.extract_video()

    def track_trials(self):
        for t in self.trials:
            t.extract_track()

    @property
    def metadata(self):
        return experiment_metadata.load_metadata(self.path)

    def histology(self, histology_name='injection_site.png'):
        histology_path = os.path.join(self.parent_path, histology_name)
        if os.path.isfile(histology_path):
            return scipy.ndimage.imread(histology_path)
        return False

    @property
    def grid_location(self):
        mtd = experiment_metadata.load_metadata(self.path)
        grid_location = mtd['grid_location']
        return grid_location

