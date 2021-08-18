import os

import numpy as np
import pandas as pd
import scipy.io
from cached_property import cached_property

import looming_spots.io.session_io
from looming_spots.util.generic_functions import flatten_list


class MouseLoomTrialGroup(object):

    """
    The aim of the MouseLoomTrialGroup class is to provide easy trial navigation for a given mouse and track
    these over different recording sessions etc.

    This includes the following functionality:

    - grouping trials temporally (i.e. before, during or after LSIE)
    - grouping trials by trial type (e.g. visual or auditory)
    - associating mouse-level metadata
    - generating mouse-level dataframes for further analysis


    """

    def __init__(self, mouse_id, exp_key=None, photometry=False):

        self.mouse_id = mouse_id
        if exp_key is not None:
            self.exp_key = exp_key
        else:
            self.exp_key = "no exp key given"
        self.trial_type_to_analyse = None
        self.kept_trials = None

        if len(self.contrasts()) > 0:
            self.set_contrasts()

        self.set_loom_trial_idx()
        self.set_auditory_trial_idx()
        self.photometry = photometry

        if self.photometry:
            self.set_normalisation_factors()
            self.set_avg_pretest_latencies()
            self.set_integral_normalisation_factors()

    def set_loom_trial_idx(self):
        for i, t in enumerate(self.loom_trials()):
            t.set_loom_trial_idx(i)

    def set_auditory_trial_idx(self):
        for i, t in enumerate(self.auditory_trials()):
            t.set_auditory_trial_idx(i)

    def mixed_post_test(self):
        all_escape = all(
            t.classify_escape() for t in self.post_test_trials()[:3]
        )
        none_escape = all(
            not t.classify_escape() for t in self.post_test_trials()[:3]
        )
        return not (all_escape or none_escape)

    def contrasts(self):
        for t in self.all_trials:
            for fname in os.listdir(t.directory):
                fpath = os.path.join(t.directory, fname)
                if "contrasts.mat" in fpath:
                    return scipy.io.loadmat(fpath)["contrasts"][0]
                elif "contrasts.npy" in fpath:
                    return np.load(fpath)
        return []

    def set_contrasts(self):
        for t, c in zip(self.loom_trials(), self.contrasts()):
            t.set_contrast(c)

    @cached_property
    def all_trials(
        self,
    ):  # TODO: reimplement without linked list e.g. weakref
        print(self.mouse_id)

        unlinked_trials = sorted(
            flatten_list(
                [
                    s.trials
                    for s in looming_spots.io.session_io.load_sessions(
                        self.mouse_id
                    )
                ]
            )
        )
        singly_linked_trials = []
        doubly_linked_trials = []

        for i, (t_current, t_next) in enumerate(
            zip(unlinked_trials[0:-1], unlinked_trials[1:])
        ):
            t_current.set_next_trial(t_current, t_next)
            singly_linked_trials.append(t_current)
        singly_linked_trials.append(unlinked_trials[-1])

        doubly_linked_trials.append(singly_linked_trials[0])
        for i, (t_current, t_next) in enumerate(
            zip(singly_linked_trials[0:-1], singly_linked_trials[1:])
        ):
            t_next.set_previous_trial(t_next, t_current)
            doubly_linked_trials.append(t_next)

        return doubly_linked_trials

    def data(self, key):
        return np.concatenate([s.get_data(key)() for s in self.sessions])

    @cached_property
    def sessions(self):
        unlinked_sessions = looming_spots.io.session_io.load_sessions(
            self.mouse_id
        )
        singly_linked_trials, doubly_linked_sessions = [], []

        for i, (s_current, s_next) in enumerate(
            zip(unlinked_sessions[0:-1], unlinked_sessions[1:])
        ):
            s_current.set_next_session(s_current, s_next)
            singly_linked_trials.append(s_current)
        singly_linked_trials.append(unlinked_sessions[-1])

        doubly_linked_sessions.append(singly_linked_trials[0])
        for i, (s_current, s_next) in enumerate(
            zip(singly_linked_trials[0:-1], singly_linked_trials[1:])
        ):
            s_next.set_previous_session(s_next, s_current)
            doubly_linked_sessions.append(s_next)
        return doubly_linked_sessions

    def loom_trials(self):
        return [t for t in self.all_trials if t.stimulus_type == "loom"]

    def auditory_trials(self):
        return [t for t in self.all_trials if t.stimulus_type == "auditory"]

    def pre_test_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == "pre_test"]

    def post_test_trials(self):
        return [
            t for t in self.all_trials if t.get_trial_type() == "post_test"
        ]

    def lsie_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == "lsie"]

    def get_trials_of_type(self, key, n_trials=3):
        if key == "pre_test":
            return self.pre_test_trials()[0:n_trials]
        elif key == "post_test":
            return self.post_test_trials()[0:n_trials]
        elif key == "lsie":
            return self.lsie_trials()
        else:
            return self.all_trials

    def get_loom_idx(self, trial):
        for i, t in enumerate(self.all_trials):
            if t == trial:
                return i

    def n_flees(self, trial_type="pre_test"):
        return np.count_nonzero(
            [t.classify_escape() for t in self.get_trials_of_type(trial_type)]
        )

    def n_non_flees(self, trial_type="pre_test"):
        return len(self.get_trials_of_type(trial_type)) - self.n_flees(
            trial_type
        )

    def flee_rate(self, trial_type):
        return self.n_flees(trial_type) / (
            len(self.n_non_flees(trial_type)) + self.n_flees(trial_type)
        )

    def to_df(self, group_id, trial_type="pre_test", extra_data=None, n_trials=3):

        if extra_data is None:
            extra_data = {}

        mouse_df = pd.DataFrame()
        trials = self.get_trials_of_type(trial_type, n_trials=n_trials)
        delta_f_norm_factor = get_normalisation_factor(self)
        delta_f_integral_norm_factor = self.get_metric_normalising_factor()
        for t in trials:
            extra_data.setdefault("loom_idx", self.get_loom_idx(t))
            extra_data.setdefault("delta_f_norm_factor", delta_f_norm_factor)
            extra_data.setdefault("delta_f_integral_norm_factor", delta_f_integral_norm_factor)

            trial_df = t.to_df(
                group_id, extra_data=extra_data, photometry=self.photometry
            )
            mouse_df = mouse_df.append(trial_df)
        mouse_df['normalised_delta_f_0.5s'] = mouse_df['delta_f_0.5s'] / mouse_df['delta_f_norm_factor']

        return mouse_df

    def normalisation_factor_metric(self):
        return get_normalisation_factor(self)

    def normalisation_factor_raw(self):
        return get_raw_normalisation_factor(self)

    def set_normalisation_factors(self):
        normalisation_factor = self.normalisation_factor_raw()
        [t.set_delta_f_norm_factor(normalisation_factor) for t in self.loom_trials()]

    def set_avg_pretest_latencies(self):
        avg_latency = np.mean([t.track.latency_in_samples() for t in self.pre_test_trials()[:3]])
        [t.set_avg_pretest_latency(avg_latency) for t in self.loom_trials()]

    def set_integral_normalisation_factors(self):
        normalisation_factor = self.get_metric_normalising_factor()
        [t.set_integral_norm_factor(normalisation_factor) for t in self.loom_trials()]

    def get_metric_normalising_factor(self):
        return max([t.integral_escape_metric(t.avg_pretest_latency) for t in self.loom_trials()[:30]])


def get_raw_normalisation_factor(mtg):
    all_peak_df_values = []
    for t in mtg.loom_trials():
        if t.delta_f() is not None:
            all_peak_df_values.append(max(t.delta_f()))
    if all_peak_df_values:
        return max(all_peak_df_values)


def get_normalisation_factor(mtg, timepoint=215):

    normalising_factor = max(
        [
            np.nanmax(
                [
                    t.integral_escape_metric(int(timepoint))
                    for t in mtg.loom_trials()[:30]
                ]
            )
        ]
    )
    return normalising_factor
