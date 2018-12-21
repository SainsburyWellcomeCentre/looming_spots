import numpy as np
import pandas as pd
from util.generic_functions import flatten_list

from looming_spots.analysis.trial_group_analysis import make_trial_heatmap_location_overlay
from looming_spots.db import load, experimental_log


class ExperimentalConditionGroup(object):
    def __init__(self, labels, mouse_ids=None, ignore_ids=None):
        self.labels = labels
        self.mouse_ids = mouse_ids
        self.avg_df = pd.DataFrame()
        self.trials_df = pd.DataFrame()
        self.ignore_ids = ignore_ids

        if mouse_ids is None:
            self.groups = self.get_groups_from_record_sheet()
        else:
            self.groups = {label: list(mouse_id_group) for label, mouse_id_group in zip(labels, mouse_ids)}

    def remove_ignore_mice(self, mouse_ids):
        return list(set(mouse_ids).symmetric_difference(set(self.ignore_ids)))

    def get_groups_from_record_sheet(self):
        mouse_group_dictionary = {}
        for label in self.labels:
            mouse_ids_in_group = experimental_log.get_mouse_ids_in_experiment(label)
            mouse_ids_in_group = self.remove_ignore_mice(mouse_ids_in_group)
            mouse_group_dictionary.setdefault(label, mouse_ids_in_group)

        return mouse_group_dictionary

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
                experimental_condition_df = experimental_condition_df.append(mouse_trials_df)
                n_rows += len(mouse_trials_df)
            experimental_labels = [experimental_label] * n_rows
            experimental_condition_df['experimental condition'] = pd.Series(experimental_labels,
                                                                            index=experimental_condition_df.index)
            self.trials_df = self.trials_df.append(experimental_condition_df)
        return self.trials_df


class MouseLoomTrialGroup(object):
    def __init__(self, mouse_id):
        self.mouse_id = mouse_id
        self.trial_type_to_analyse = None
        self.kept_trials = None

    @classmethod
    def analysed_metrics(cls):
        metrics = ['speed', 'acceleration', 'latency to escape', 'time in safety zone', 'classified as flee']
        return metrics

    @property
    def all_trials(self):  # TODO: this can probably be achieved more elegantly  #TODO: weakref
        print(self.mouse_id)
        unlinked_trials = sorted(flatten_list([s.trials for s in load.load_sessions(self.mouse_id)]))
        singly_linked_trials = []
        doubly_linked_trials = []

        for i, (t_current, t_next) in enumerate(zip(unlinked_trials[0:-1], unlinked_trials[1:])):
            t_current.set_next_trial(t_current, t_next)
            singly_linked_trials.append(t_current)
        singly_linked_trials.append(unlinked_trials[-1])

        doubly_linked_trials.append(singly_linked_trials[0])
        for i, (t_current, t_next) in enumerate(zip(singly_linked_trials[0:-1], singly_linked_trials[1:])):
            t_next.set_previous_trial(t_next, t_current)
            doubly_linked_trials.append(t_next)

        return doubly_linked_trials

    def pre_test_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == 'pre_test']

    def post_test_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == 'post_test']

    def habituation_trials(self):
        return [t for t in self.all_trials if t.get_trial_type() == 'habituation']

    def get_trials_of_type(self, key):
        if key == 'pre_test':
            return self.pre_test_trials()
        elif key == 'post_test':
            return self.post_test_trials()
        elif key == 'habituation':
            return self.habituation_trials()

    def n_flees(self, trial_type='pre_test'):
        return np.count_nonzero([t.is_flee() for t in self.get_trials_of_type(trial_type)])

    def n_non_flees(self, trial_type='pre_test'):
        return len(self.get_trials_of_type(trial_type)) - self.n_flees(trial_type)

    def flee_rate(self, trial_type):
        return self.n_flees(trial_type) / (len(self.n_non_flees(trial_type)) + self.n_flees(trial_type))

    def get_reference_frame(self, key):
        if key == 'pre_test':
            return self.pre_test_trials()[0].get_reference_frame()
        elif key == 'post_test':
            return self.post_test_trials()[0].get_reference_frame()
        elif key == 'habituation':
            return [t for t in self.all_trials if t.get_trial_type() == 'habituation'][0].get_reference_frame()

    def habituation_heatmap(self, n_trials_to_show):
        if n_trials_to_show is None:
            n_trials_to_show = -1
        trials = self.get_trials_of_type('habituation')[0:n_trials_to_show]
        return make_trial_heatmap_location_overlay(trials, self.get_reference_frame('habituation'))

    def get_metric_data(self, metric, trial_type='pre_test'):
        metric_values = []
        for t in self.get_trials_of_type(trial_type):
            metric_value = t.metric_functions[metric]()
            metric_values.append(metric_value)
        return metric_values

    def to_trials_df(self, trial_type='pre_test'):
        metrics_dict = {}
        for metric in MouseLoomTrialGroup.analysed_metrics():
            data = self.get_metric_data(metric, trial_type=trial_type)
            metrics_dict.setdefault(metric, data)

        n_trials = len(self.get_trials_of_type(trial_type))
        metrics_dict.setdefault('mouse_id', [self.mouse_id]*n_trials)

        return pd.DataFrame.from_dict(metrics_dict)

    def to_avg_df(self, trial_type='pre_test'):
        mouse_dict = {}
        for metric in MouseLoomTrialGroup.analysed_metrics():
            values = [t.metric_functions[metric]() for t in self.get_trials_of_type(trial_type)]
            mouse_dict.setdefault(metric, [np.nanmean(values)])
            mouse_dict.setdefault('mouse_id', [self.mouse_id])

        return pd.DataFrame.from_dict(mouse_dict)
