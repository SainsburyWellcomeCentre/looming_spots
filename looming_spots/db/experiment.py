

class Experiment(object):
    def __init__(self, session_groups=None, name=None):
            self.session_groups = session_groups
            self.name = name

    def plot_all_conditions(self):
        for session_group in self.session_groups:
            session_group.plot_all_sessions()

    def plot_all_heatmaps(self):
        for session_group in self.session_groups:
            session_group.plot_all_sessions_heatmaps()

    def plot_all_acc_heatmaps(self):
        for session_group in self.session_groups:
            session_group.plot_acc_heatmaps()
