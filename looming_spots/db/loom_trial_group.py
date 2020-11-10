import os

import numpy as np
import pandas as pd
import scipy.io
from cached_property import cached_property
from tqdm import tqdm

import looming_spots.io.session_io
from looming_spots.util.generic_functions import flatten_list
from looming_spots.db import experimental_log


class MouseLoomTrialGroup(object):
    def __init__(self, mouse_id, exp_key=None):
        self.mouse_id = mouse_id
        if exp_key is not None:
            self.exp_key = exp_key
        else:
            self.exp_key = 'no exp key given'
        self.trial_type_to_analyse = None
        self.kept_trials = None

        if len(self.contrasts()) > 0:
            self.set_contrasts()

        self.set_loom_trial_idx()
        self.set_auditory_trial_idx()

    def set_loom_trial_idx(self):
        for i, t in enumerate(self.loom_trials()):
            t.set_loom_trial_idx(i)

    def set_auditory_trial_idx(self):
        for i, t in enumerate(self.auditory_trials()):
            t.set_auditory_trial_idx(i)

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

    @classmethod
    def analysed_metrics(cls):
        metrics = [
            "speed",
            "acceleration",
            "latency to escape",
            "latency peak detect",
            "time in safety zone",
            "classified as flee",
            "time of loom",
            "loom number",
            "time to reach shelter stimulus onset",

        ]
        return metrics

    @classmethod
    def analysed_event_metrics(cls):
        metrics = ["integral at latency", "integral at end"]
        return metrics

    @cached_property
    def all_trials(
        self
    ):  # TODO: this can probably be achieved more elegantly  #TODO: weakref
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
    def sessions(self):  # TODO: weakref
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

    def habituation_trials(self):
        return [
            t for t in self.all_trials if t.get_trial_type() == "habituation"
        ]

    def get_trials_of_type(self, key, limit=3):
        if key == "pre_test":
            return self.pre_test_trials()[0:limit]
        elif key == "post_test":
            return self.post_test_trials()[0:limit]
        elif key == "habituation":
            return self.habituation_trials()
        else:
            return self.all_trials

    def get_loom_idx(self, trial):
        for i, t in enumerate(self.all_trials):
            if t == trial:
                return i

    def n_flees(self, trial_type="pre_test"):
        return np.count_nonzero(
            [t.is_flee() for t in self.get_trials_of_type(trial_type)]
        )

    def n_non_flees(self, trial_type="pre_test"):
        return len(self.get_trials_of_type(trial_type)) - self.n_flees(
            trial_type
        )

    def flee_rate(self, trial_type):
        return self.n_flees(trial_type) / (
            len(self.n_non_flees(trial_type)) + self.n_flees(trial_type)
        )

    def get_reference_frame(self, key):
        if key == "pre_test":
            return self.pre_test_trials()[0].get_reference_frame()
        elif key == "post_test":
            return self.post_test_trials()[0].get_reference_frame()
        elif key == "habituation":
            return [
                t
                for t in self.all_trials
                if t.get_trial_type() == "habituation"
            ][0].get_reference_frame()

    def habituation_heatmap(self, n_trials_to_show):
        if n_trials_to_show is None:
            n_trials_to_show = -1
        trials = self.get_trials_of_type("habituation")[0:n_trials_to_show]
        return make_trial_heatmap_location_overlay(
            trials, self.get_reference_frame("habituation")
        )

    def get_metric_data(self, metric, trial_type="pre_test", limit=3):
        metric_values = []
        for i, t in enumerate(self.get_trials_of_type(trial_type)[0:limit]):
            metric_value = t.metric_functions[metric]()
            metric_values.append(metric_value)
        return metric_values

    def to_trials_df(self, trial_type="pre_test"):
        metrics_dict = {}
        all_metric_labels = []
        all_metric_values = []

        for metric in MouseLoomTrialGroup.analysed_metrics():
            data = self.get_metric_data(metric, trial_type=trial_type)
            # metrics_dict.setdefault(metric, data)
            metric_labels = [metric] * len(data)
            metric_values = data
            all_metric_labels.extend(metric_labels)
            all_metric_values.extend(metric_values)

        all_loom_idx = []
        trials = self.get_trials_of_type(trial_type)
        for t in trials:
            loom_idx = self.get_loom_idx(t)
            all_loom_idx.append(loom_idx)

        n_trials = len(trials)
        n_metrics = len(MouseLoomTrialGroup.analysed_metrics())
        all_loom_idx = list(all_loom_idx) * n_metrics
        metrics_dict.setdefault("loom_idx", all_loom_idx)
        # n_trials = len(self.get_trials_of_type(trial_type))
        metrics_dict.setdefault(
            "mouse_id", [self.mouse_id] * n_metrics * n_trials
        )
        metrics_dict.setdefault("metric_label", all_metric_labels)
        metrics_dict.setdefault("metric_value", all_metric_values)

        return pd.DataFrame.from_dict(metrics_dict)

    def to_avg_df(self, trial_type="pre_test"):
        mouse_dict = {}
        for metric in MouseLoomTrialGroup.analysed_metrics():

            ignore_metrics = ["time of loom", "loom number"]
            if metric not in ignore_metrics:
                values = [
                    t.metric_functions[metric]()
                    for t in self.get_trials_of_type(trial_type)
                ]
                mouse_dict.setdefault(metric, [np.nanmean(values)])
                mouse_dict.setdefault("mouse_id", [self.mouse_id])

        return pd.DataFrame.from_dict(mouse_dict)

    def events_df(self, trial_type=None):
        df_all_metrics = pd.DataFrame()
        for metric in MouseLoomTrialGroup.analysed_event_metrics():
            event_metric_df = self.get_event_metric_data(
                metric, trial_type=trial_type
            )
            df_all_metrics = df_all_metrics.append(
                event_metric_df, ignore_index=True
            )
        return df_all_metrics

    def get_event_metric_data(
        self, metric, trial_type=None, stimulus_type="loom"
    ):

        all_trials_df = pd.DataFrame()
        for i, t in tqdm(enumerate(self.get_trials_of_type(trial_type))):
            event_metric_dict = {}
            if t.stimulus_type == stimulus_type:

                metric_values = t.event_metric_functions[metric]()
                trial_idx = [self.get_loom_idx(t)]
                mouse_ids = [self.mouse_id]
                metric_labels = [metric]

                event_metric_dict.setdefault("metric", metric_labels)
                event_metric_dict.setdefault("values", metric_values)
                event_metric_dict.setdefault("trial_idx", trial_idx)
                event_metric_dict.setdefault("mouse_id", mouse_ids)

                trial_df = pd.DataFrame.from_dict(event_metric_dict)
                all_trials_df = all_trials_df.append(trial_df)
        return all_trials_df

    def percentage_time_in_tz_middle(self):
        hm = make_trial_heatmap_location_overlay(self.habituation_trials())
        return sum(sum(hm[115:190, 150:245]) / sum(sum(hm)))

    def sort_trials_by_contrast(self):
        test_contrast_trials = [t for t in self.all_trials if t.contrast == 0]
        low_contrast_trials = [t for t in self.all_trials if t.contrast != 0]
        low_contrast_values = [t.contrast for t in low_contrast_trials]

        Z = [x for _, x in sorted(zip(low_contrast_values, low_contrast_trials))]
        return Z


class ExperimentalConditionGroup(object):
    def __init__(self, labels, mouse_ids=None, ignore_ids=None, limit=None):
        if isinstance(labels, str):
            self.labels = [labels]
        else:
            self.labels = labels
        self.mouse_ids = mouse_ids
        self.avg_df = pd.DataFrame()
        self.trials_df = pd.DataFrame()
        self.ignore_ids = ignore_ids
        self.limit = limit
        if mouse_ids is None:
            self.groups = self.get_groups_from_record_sheet()
        else:
            self.groups = {
                label: list(mouse_id_group)
                for label, mouse_id_group in zip(labels, mouse_ids)
            }

    def remove_ignore_mice(self, mouse_ids):
        if self.ignore_ids is None:
            return mouse_ids
        return list(set(mouse_ids).symmetric_difference(set(self.ignore_ids)))

    def get_groups_from_record_sheet(self):
        mouse_group_dictionary = {}
        for label in self.labels:
            mouse_ids_in_group = experimental_log.get_mouse_ids_in_experiment(
                label
            )
            mouse_ids_in_group = self.remove_ignore_mice(mouse_ids_in_group)
            mouse_group_dictionary.setdefault(label, mouse_ids_in_group)

        return mouse_group_dictionary

    def trials(self, trial_type):
        trial_group_dictionary = {}
        for experimental_label, mouse_ids in self.groups.items():
            trials = []
            for mid in mouse_ids:
                mltg = MouseLoomTrialGroup(mid)
                trials.extend(mltg.get_trials_of_type(trial_type, self.limit))
            trial_group_dictionary.setdefault(experimental_label, trials)
        return trial_group_dictionary

    def mouse_trial_groups(self):
        mtg_dict = {}
        for experimental_label, mouse_ids in self.groups.items():
            mtgs = [MouseLoomTrialGroup(mid) for mid in mouse_ids]
            mtg_dict.setdefault(experimental_label, mtgs)
        return mtg_dict

    def to_df(self, trial_type, average=False):
        """

        :param string trial_type:
        :param boolean average: trial wise if False, mouse wise if True
        :return : a pandas dataframe containing all trial metrics
        """

        for experimental_label, mouse_ids in self.groups.items():
            experimental_condition_df = pd.DataFrame()
            n_rows = 0
            for mid in mouse_ids:
                print(mouse_ids, type(mouse_ids))
                mtg = MouseLoomTrialGroup(mid)
                get_df_func = mtg.to_avg_df if average else mtg.to_trials_df

                mouse_trials_df = get_df_func(trial_type)
                experimental_condition_df = experimental_condition_df.append(
                    mouse_trials_df
                )
                n_rows += len(mouse_trials_df)
            experimental_labels = [experimental_label] * n_rows
            experimental_condition_df["experimental condition"] = pd.Series(
                experimental_labels, index=experimental_condition_df.index
            )
            self.trials_df = self.trials_df.append(experimental_condition_df)
        return self.trials_df


def make_trial_heatmap_location_overlay(trials):
    hm = None
    for t in trials:
        hm = t.track_overlay(track_heatmap=hm)
    return hm
