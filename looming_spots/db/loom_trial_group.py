import numpy as np
import pandas as pd
from cached_property import cached_property

from looming_spots.loom_io import session_io


class MouseLoomTrialGroup(object):

    """
    MouseLoomTrialGroup provides trial navigation for a given mouse
    across all recording sessions.

    This includes the following functionality:

    - grouping trials temporally (i.e. before, during or after LSE)
    - grouping trials by trial type (e.g. visual or auditory)
    - associating mouse-level metadata
    - generating mouse-level dataframes
    """

    def __init__(self, mouse_id, exp_key=None):

        self.mouse_id = mouse_id
        if exp_key is not None:
            self.exp_key = exp_key
        else:
            self.exp_key = "no exp key given"
        self.trial_type_to_analyse = None
        self.kept_trials = None

        self.set_loom_trial_idx()
        self.set_auditory_trial_idx()

    def set_loom_trial_idx(self):
        for i, t in enumerate(self.loom_trials()):
            t.set_loom_trial_idx(i)

    def set_auditory_trial_idx(self):
        for i, t in enumerate(self.auditory_trials()):
            t.set_auditory_trial_idx(i)

    def mixed_post_test(self):
        all_escape = all(t.classify_escape() for t in self.post_test_trials()[:3])
        none_escape = all(not t.classify_escape() for t in self.post_test_trials()[:3])
        return not (all_escape or none_escape)

    @cached_property
    def all_trials(
        self,
    ):
        trials = []
        for i, s in enumerate(session_io.load_sessions(self.mouse_id)):
            print(f"loading {self.mouse_id}, session {i}")
            trials.extend(s.trials)
        unlinked_trials = sorted(trials)

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
        unlinked_sessions = session_io.load_sessions(self.mouse_id)
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
        return [t for t in self.all_trials if t.get_trial_type() == "post_test"]

    def lse_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == "lse"]

    def get_trials_of_type(self, key, limit=3):
        if key == "pre_test":
            return self.pre_test_trials()[0:limit]
        elif key == "post_test":
            return self.post_test_trials()[0:limit]
        elif key == "lse":
            return self.lse_trials()
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
        return len(self.get_trials_of_type(trial_type)) - self.n_flees(trial_type)

    def flee_rate(self, trial_type):
        return self.n_flees(trial_type) / (
            len(self.n_non_flees(trial_type)) + self.n_flees(trial_type)
        )

    def get_metric_data(self, metric, trial_type="pre_test", limit=3):
        metric_values = []
        for i, t in enumerate(self.get_trials_of_type(trial_type)[0:limit]):
            metric_value = t.metric_functions[metric]()
            metric_values.append(metric_value)
        return metric_values

    def to_df(self, group_id, trial_type="pre_test"):
        mouse_df = pd.DataFrame()
        trials = self.get_trials_of_type(trial_type)
        for t in trials:
            trial_df = t.to_df(group_id, extra_data={"loom_idx": self.get_loom_idx(t)})
            mouse_df = mouse_df.append(trial_df)
        return mouse_df
