import os
import numpy as np
from datetime import datetime
import scipy.ndimage
from cached_property import cached_property

from looming_spots.db.constants import AUDITORY_STIMULUS_CHANNEL_ADDED_DATE, PROCESSED_DATA_DIRECTORY

from looming_spots.db import loomtrial
from looming_spots.exceptions import LoomsNotTrackedError

from looming_spots.preprocess import photodiode
from looming_spots.util import generic_functions
from looming_spots.db.metadata import experiment_metadata
from looming_spots.analysis.photometry import demodulation


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

    def contains_auditory(self):
        aud_idx_path = os.path.join(self.path, 'auditory_starts.npy')
        if os.path.isfile(aud_idx_path):
            if len(np.load(aud_idx_path)) > 0:
                return True

        recording_date = datetime.strptime(os.path.split(self.path)[-1], '%Y%m%d_%H_%M_%S')

        if 'AI.tdms' in os.listdir(self.path):
            return True

        if recording_date > AUDITORY_STIMULUS_CHANNEL_ADDED_DATE:
            ad = photodiode.load_auditory_on_clock_ups(self.path)
            if (ad > 0.7).any():
                return True

    @cached_property
    def trials(self):
        visual_trials_idx = self.get_visual_trials_idx()
        auditory_trials_idx = self.get_auditory_trials_idx()
        visual_trials = self.initialise_trials(visual_trials_idx, 'visual')
        auditory_trials = self.initialise_trials(auditory_trials_idx, 'auditory')
        auditory_trials = self.initialise_trials(auditory_trials_idx, 'auditory')

        return sorted(visual_trials + auditory_trials)

    def initialise_trials(self, idx, stimulus_type):
        if idx is not None:
            if len(idx) > 0:
                trials = []
                for i, onset_in_samples in enumerate(idx):
                    if stimulus_type == 'visual':
                        t = loomtrial.VisualStimulusTrial(self, directory=self.path, sample_number=onset_in_samples,
                                                          trial_type=self.get_trial_type(onset_in_samples),
                                                          stimulus_type='loom')
                    elif stimulus_type == 'auditory':
                        t = loomtrial.AuditoryStimulusTrial(self, directory=self.path, sample_number=onset_in_samples,
                                                            trial_type=self.get_trial_type(onset_in_samples),
                                                            stimulus_type='auditory')
                    else:
                        raise NotImplementedError

                    t.time_to_first_loom = self.time_to_first_loom()
                    trials.append(t)
                return trials
            else:
                return []
        return []

    def get_trials(self, key):

        if key == 'test':
            return [t for t in self.trials if 'test' in t.trial_type]

        return [t for t in self.trials if t.trial_type == key]

    @property
    def context(self):
        try:
            return experiment_metadata.get_context_from_stimulus_mat(self.path)
        except FileNotFoundError as e:
            print(e)
            print('guessing A10')
            return 'A'


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
    def auditory_trace(self):
        return photodiode.load_auditory_on_clock_ups(self.path)

    def load_demodulated_photometry_on_clock_ups(self):
        pd, clock, auditory = photodiode.load_pd_and_clock_raw(self.path)
        clock_ups = photodiode.get_clock_ups(clock)
        dm_signal, dm_background = self.load_demodulated_photometry()
        return dm_signal[clock_ups], dm_background[clock_ups]

    def load_demodulated_photometry(self):
        pd, _, auditory, photometry, led211, led531 = photodiode.load_all_channels_raw(self.path)
        dm_signal, dm_background = demodulation.demodulate(photometry, led211, led531)
        if self.path == '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/074744/20190404_20_33_34':
            print('flipping channels')
            dm_signal, dm_background = dm_background, dm_signal
        return dm_signal, dm_background

    @cached_property
    def dm_signal(self):
        return self.load_demodulated_photometry_on_clock_ups()[0]

    @cached_property
    def dm_background(self):
        return self.load_demodulated_photometry_on_clock_ups()[1]

    def load_all_channels_on_clock_ups(self):
        all_channels = photodiode.load_all_channels_raw(self.path)
        clock_ups = photodiode.get_clock_ups(all_channels[1])
        return [channel[clock_ups] for channel in all_channels]

    @cached_property
    def fully_sampled_delta_f(self, filter_artifact_cutoff_samples=40000):
        pd, clock, auditory, pmt, led211, led531 = photodiode.load_all_channels_raw(self.path)

        if self.path == '/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/processed_data/074744/20190404_20_33_34':
            print('flipping channels')
            led211, led531 = led531, led211

        pmt[:filter_artifact_cutoff_samples] = np.median(pmt)
        delta_f = demodulation.lerner_deisseroth_preprocess(pmt, led211, led531)
        delta_f[:filter_artifact_cutoff_samples] = np.median(delta_f)  # to remove filter artifact
        return delta_f, clock

    @cached_property
    def delta_f(self):
        delta_f, clock = self.fully_sampled_delta_f
        clock_ups = photodiode.get_clock_ups(clock)
        return delta_f[clock_ups]

        # sig, bg = self.load_demodulated_photometry_on_clock_ups()
        # normalised_sig = sig - gaussian_filter(sig, 30)
        # normalised_bg = bg - gaussian_filter(bg, 30)
        # return normalised_sig-normalised_bg  # gaussian_filter((normalised_sig-normalised_bg), 2)



    @property
    def contains_habituation(self):
        return photodiode.contains_habituation(self.loom_idx)

    @cached_property
    def loom_idx(self):
        loom_idx_path = os.path.join(self.path, 'loom_starts.npy')
        if not os.path.isfile(loom_idx_path):
            _ = photodiode.get_loom_idx_from_raw(self.path, save=True)[0]
        return np.load(loom_idx_path)

    @cached_property
    def auditory_idx(self):
        if self.contains_auditory():
            aud_idx_path = os.path.join(self.path, 'auditory_starts.npy')
            if not os.path.isfile(aud_idx_path):
                _ = photodiode.get_auditory_onsets_from_analog_input(self.path, save=True)
            return np.load(aud_idx_path)

    def get_trial_type(self, onset_in_samples):
        if self.habituation_idx is None:
            return 'test'
        elif onset_in_samples in self.habituation_idx:
            return 'habituation'
        else:
            return 'test'

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

    def contains_visual(self):
        pd = photodiode.load_pd_on_clock_ups(self.path)
        if (pd > 0.5).any():
            return True

    def get_visual_trials_idx(self):
        return self.loom_idx[::5]

    def get_auditory_trials_idx(self):
        return self.auditory_idx



