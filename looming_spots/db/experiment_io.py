import numpy as np
from cached_property import cached_property

from looming_spots.db import load


class ExperimentIo(object):  # TODO: merge with ltg
    def __init__(self, mouse_id):
        self.mouse_id = mouse_id

    def data(self, key):
        return np.concatenate([s.get_data(key)() for s in self.sessions])

    @cached_property
    def sessions(self):  # TODO: weakref
        unlinked_sessions = load.load_sessions(self.mouse_id)
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

    @cached_property
    def linked_trials(self):
        unlinked_trials = self.data("trials")
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
