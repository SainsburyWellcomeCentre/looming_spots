import numpy as np
from cached_property import cached_property

from looming_spots.db import load


class ExperimentIo(object):
    def __init__(self, mouse_id):
        self.mouse_id = mouse_id

    def data(self, key):
        return np.concatenate([s.get_data(key)() for s in self.sessions])

    @cached_property
    def sessions(self):  # TODO: weakref
        unlinked_sessions = load.load_sessions(self.mouse_id)
        singly_linked_trials = []
        doubly_linked_sessions = []

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
