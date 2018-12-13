import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from util.generic_functions import flatten_list

from looming_spots.analysis import plotting
from looming_spots.db import load, experimental_log


class LoomTrialGroup(object):  # FIXME: deprecated, phase out
    def __init__(self, trials, label):
        self.trials = trials
        self.label = label
        self.n_trials = self.n_non_flees + self.n_flees
        self.n_mice = int(self.n_trials/3)

    @property
    def n_flees(self):
        return np.count_nonzero([t.is_flee() for t in self.trials])

    @property
    def n_non_flees(self):
        return len(self.trials) - self.n_flees

    @property
    def flee_rate(self):
        return self.n_flees/len(self.trials)

    def add_trials(self, trials):
        for trial in trials:
            self.trials.append(trial)

    def get_trials(self):
        return self.trials

    def plot_all_tracks(self):
        fig = plt.gcf()
        for t in self.get_trials():
            t.plot_track()
        plotting.plot_looms(fig)

    def plot_all_peak_acc(self):
        for t in self.get_trials():
            t.plot_peak_x_acceleration()

    def all_tracks(self):
        return [t.smoothed_x_speed for t in self.trials]

    def sorted_tracks(self, values_to_sort_by=None):
        if values_to_sort_by is None:
            return self.all_tracks()
        else:
            args = np.argsort(values_to_sort_by)
            order = [np.where(args == x)[0][0] for x in range(len(self.all_tracks()))]
            sorted_tracks = []
            for item, arg, sort_var in zip(order, args, values_to_sort_by):
                trial_distances = self.all_tracks()[arg]
                sorted_tracks.append(trial_distances[:400])
            return sorted_tracks

    def plot_hm(self, values_to_sort_by):
        fig = plt.figure(figsize=(7, 5))
        tracks = self.sorted_tracks(values_to_sort_by)
        plt.imshow(tracks, cmap='coolwarm_r', aspect='auto', vmin=-0.05, vmax=0.05)
        title = '{}, {} flees out of {} trials, n={} mice'.format(self.label, self.n_flees, self.n_trials, self.n_mice)
        plt.title(title)
        plt.axvline(200, color='k')
        cbar = plt.colorbar()
        cbar.set_label('velocity in x axis a.u.')
        plt.ylabel('trial number')
        plt.xlabel('n frames')
        return fig

    def get_metric_data(self, metric):
        metric_values = []
        for t in self.trials:
            metric_value = t.metric_functions[metric]()
            metric_values.append(metric_value)
        return metric_values

    def times_to_first_loom(self):
        times_to_first_loom = []
        for t in self.trials:
            times_to_first_loom.append(t.time_to_first_loom)
        return times_to_first_loom

    def to_df(self):
        metrics_dict = {}
        for metric in LoomTrialGroup.analysed_metrics():
            labels = [self.label]*self.n_trials
            data = self.get_metric_data(metric)
            metrics_dict.setdefault('condition', labels)
            metrics_dict.setdefault(metric, data)

        return pd.DataFrame.from_dict(metrics_dict)

    def get_mouse_values(self, mouse_id, trials):
        mouse_trials = [t for t in trials if t.mouse_id == mouse_id]
        mouse_dict = {}
        for metric in LoomTrialGroup.analysed_metrics():
            values = [t.metric_functions[metric]() for t in mouse_trials]
            mouse_dict.setdefault(metric, [np.nanmean(values)])
            mouse_dict.setdefault('mouse_id', [mouse_id])

            if 'acceleration' in metric:
                print('mouse_id {} {}: {}'.format(mouse_id, metric, np.nanmean(values)))

        return pd.DataFrame.from_dict(mouse_dict)

    def get_mouse_avg_df(self, trials, experimental_condition=None):

        all_mouse_ids = set([t.mouse_id for t in trials])
        mouse_avg_df = pd.DataFrame()

        for mouse_id in all_mouse_ids:
            mouse_df = self.get_mouse_values(mouse_id, trials)
            mouse_avg_df = mouse_avg_df.append(mouse_df, ignore_index=True)

        if experimental_condition is not None:
            mouse_avg_df['experimental group'] = pd.Series([experimental_condition]*len(all_mouse_ids), index=mouse_avg_df.index)

        return mouse_avg_df

    @classmethod
    def analysed_metrics(cls):
        metrics = ['speed', 'acceleration', 'latency to escape', 'time in safety zone', 'classified as flee']
        return metrics


class ExperimentalConditionGroup(object):
    def __init__(self, labels, mouse_ids=None):
        self.labels = labels
        self.mouse_ids = mouse_ids

    def load_from_mouse_ids(self):
        pass

    def load_from_database(self):
        pass

    @property
    def groups(self):
        mouse_group_dictionary = {}
        for label in self.labels:
            mouse_ids_in_group = experimental_log.get_mouse_ids_in_experiment(label)
            mouse_group_dictionary.setdefault(label, mouse_ids_in_group)
        return mouse_group_dictionary

    def get_all_dfs(self, trial_type):
        df_all = pd.DataFrame()
        for experimental_label, mouse_ids in self.groups.items():
            experimental_condition_df = pd.DataFrame()
            for mid in mouse_ids:
                mtg = MouseLoomTrialGroup(mid)
                mouse_avg_df = mtg.to_avg_df(trial_type)
                experimental_condition_df = experimental_condition_df.append(mouse_avg_df)
            experimental_condition_df['experimental condition'] = pd.Series([experimental_label]*len(mouse_ids),
                                                                            index=experimental_condition_df.index)
            df_all = df_all.append(experimental_condition_df)
        return df_all

    def get_dfs_from_mids(self, trial_type, experimental_label):
        experimental_condition_df = pd.DataFrame()
        for mid in self.mouse_ids:
            mtg = MouseLoomTrialGroup(mid)
            mouse_avg_df = mtg.to_avg_df(trial_type)
            experimental_condition_df = experimental_condition_df.append(mouse_avg_df)
        experimental_condition_df['experimental condition'] = pd.Series([experimental_label] * len(self.mouse_ids),
                                                                        index=experimental_condition_df.index)
        return experimental_condition_df


class MouseLoomTrialGroup(object):
    def __init__(self, mouse_id):
        self.mouse_id = mouse_id
        self.trial_type_to_analyse = None
        self.kept_trials = None

    @classmethod
    def analysed_metrics(cls):
        metrics = ['speed', 'acceleration', 'latency', 'time in safety zone', 'classified as flee']
        return metrics

    @property
    def all_trials(self):  # TODO: this can probably be achieved more elegantly
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

    def habituation_heatmap(self, n_trials_to_show=None):
        track_heatmap = np.zeros_like(self.get_reference_frame('habituation'))

        if n_trials_to_show is None:
            n_trials_to_show = -1
        for t in self.get_trials_of_type('habituation')[:n_trials_to_show]:
            x, y = np.array(t.raw_track[0]), np.array(t.raw_track[1])
            for coordinate in zip(x, y):
                if not np.isnan(coordinate).any():
                    track_heatmap[int(coordinate[1]), int(coordinate[0])] += 1
        return track_heatmap

    def get_metric_data(self, metric, trial_type='pre_test'):
        metric_values = []
        for t in self.get_trials_of_type(trial_type):
            metric_value = t.metric_functions[metric]()
            metric_values.append(metric_value)
        return metric_values

    def to_df(self, trial_type='pre_test'):
        metrics_dict = {}
        for metric in LoomTrialGroup.analysed_metrics():
            data = self.get_metric_data(metric, trial_type=trial_type)
            metrics_dict.setdefault(metric, data)

        return pd.DataFrame.from_dict(metrics_dict)

    def to_avg_df(self, trial_type='pre_test'):
        mouse_dict = {}
        for metric in LoomTrialGroup.analysed_metrics():
            values = [t.metric_functions[metric]() for t in self.get_trials_of_type(trial_type)]
            mouse_dict.setdefault(metric, [np.nanmean(values)])
            mouse_dict.setdefault('mouse_id', [self.mouse_id])

            if 'acceleration' in metric:
                print('mouse_id {} {}: {}'.format(self.mouse_id, metric, np.nanmean(values)))

        return pd.DataFrame.from_dict(mouse_dict)
