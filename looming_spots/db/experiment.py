

class Experiment(object):
    def __init__(self, session_groups=None, name=None):
            self.session_groups = session_groups
            self.name = name

    def plot_all_conditions(self):
        for session_group in self.session_groups:
            session_group.plot_all_sessions()
