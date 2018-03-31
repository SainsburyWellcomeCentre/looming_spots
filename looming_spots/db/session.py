import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from looming_spots.analysis import tracks
from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY
from looming_spots.db.metadata import experiment_metadata
from looming_spots.db.trial import Trial
from looming_spots.util import generic_functions


class Session(object):

    def __init__(self, dt=None, protocol=None, stimulus=None, context=None, mouse_id=None):
        self.dt = dt
        self.protocol = protocol
        self.stimulus = stimulus
        self.context = context
        self.mouse_id = mouse_id

    @property
    def data_path(self):
        return os.path.join('./',  self.mouse_id, self.dt.strftime('%Y%m%d_%H_%M_%S'))

    def __lt__(self, other):
        return self.dt < other.dt

    def __gt__(self, other):
        return self.dt > other.dt

    @property
    def path(self):
        parent_dir = self.mouse_id
        session_dir = datetime.strftime(self.dt, '%Y%m%d_%H_%M_%S')
        print(parent_dir, session_dir)
        return os.path.join(PROCESSED_DATA_DIRECTORY, parent_dir, session_dir)

    @property
    def mouse_name(self):
        return self.path.split('/')[-2]

    @property
    def loom_paths(self):
        paths = []
        for name in os.listdir(self.path):
            loom_folder = os.path.join(self.path, name)
            if os.path.isdir(loom_folder) and 'loom' in name:
                paths.append(loom_folder)
        if len(paths) == 0:
            raise LoomsNotTrackedError()
        loom_indices = [int(path[-1]) for path in paths]
        sorted_paths, idx = generic_functions.sort_by(paths, loom_indices,descend=False)
        return sorted_paths

    @property
    def trials_results(self):
        return np.array([tracks.classify_flee(p, self.context) for p in self.loom_paths])

    @property
    def n_trials(self):
        return len(self.trials_results)

    def hours(self):
        return self.dt.hour + self.dt.minute/60

    @property
    def n_flees(self):
        return np.count_nonzero(self.trials_results)

    @property
    def n_non_flees(self):
        return self.n_trials - self.n_flees

    @property
    def n_looms(self):
        mtd = experiment_metadata.load_metadata(self.path)
        loom_idx = mtd['loom_idx']
        return len(loom_idx)

    @property
    def grid_location(self):
        mtd = experiment_metadata.load_metadata(self.path)
        grid_location = mtd['grid_location']
        return grid_location

    @property
    def reference_frame(self):
        fpath = os.path.join(self.path, 'ref.npy')
        return np.load(fpath)

    @property
    def trials(self):
        trials = []
        for loom_path in self.loom_paths:
            t = Trial(self, loom_path)
            trials.append(t)
        return trials

    def plot_trials(self):
        for t in self.trials:
            color = 'r' if t.is_flee() else 'k'
            plt.plot(t.normalised_x_track, color=color)


class LoomsNotTrackedError(Exception):
    def __init__(self):
        print('no loom folder paths, please check you have tracked this session')
