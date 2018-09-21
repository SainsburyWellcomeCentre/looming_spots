import matplotlib.pyplot as plt


# TODO: refactor and work out trial vs session groups

class Experiment(object):
    def __init__(self, session_groups=None, name=None, histology=False):
        self.session_groups = session_groups
        self.name = name
        if histology:
            self.n_plots = len(self.session_groups)+1
            self.plotting_idx = 1
        else:
            self.n_plots = len(self.session_groups)+1  # TODO: make this work for single plots and remove +1
            self.plotting_idx = 0
        self.figure, self.axes = plt.subplots(1, self.n_plots)

        self.plotting_axes = self.axes[self.plotting_idx:]
        #self.histology_axis = self.axes[0]

    @property
    def histology(self):
        for sg in self.session_groups:
            return sg.injection_site_image()

    def plot_all_conditions(self):
        for session_group, ax in zip(self.session_groups, self.plotting_axes):
            plt.axes(ax)
            session_group.plot_all_sessions(ax=ax)

    def plot_histology(self):
        plt.axes(self.histology_axis)
        plt.imshow(self.histology)

    def plot_all_heatmaps(self):
        for session_group in self.session_groups:
            session_group.plot_all_sessions_heatmaps()

    def plot_all_acc_heatmaps(self):
        for session_group in self.session_groups:
            session_group.plot_acc_heatmaps()
