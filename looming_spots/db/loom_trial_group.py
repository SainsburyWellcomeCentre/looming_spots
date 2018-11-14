import numpy as np
from matplotlib import pyplot as plt

from looming_spots.analysis import plotting


class LoomTrialGroup(object):
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

    def latencies(self):
        latencies = []
        for t in self.trials:
            latency = int(t.peak_x_acc_idx())
            latencies.append(latency)
        return latencies

    def times_to_first_loom(self):
        times_to_first_loom = []
        for t in self.trials:
            times_to_first_loom.append(t.time_to_first_loom)
        return times_to_first_loom

    def plot_latencies(self, i):
        plt.gca()
        for t in self.trials:
            latency = int(t.peak_x_acc_idx())
            color = 'r' if t.is_flee() else 'k'
            plt.plot(i+np.random.rand(1)[0]/50, (latency-200)/30, 'o', color=color)