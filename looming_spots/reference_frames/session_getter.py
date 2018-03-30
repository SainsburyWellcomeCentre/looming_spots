import os

from looming_spots.reference_frames import viewer
from looming_spots.util import generic_functions
from zarchive import load_sessions


class SessionGetter(object):
    def __init__(self, mouse_id_list=None, session_list=None):
        if mouse_id_list is not None:
            self.mice = [load_sessions(mid) for mid in mouse_id_list]
            self.sessions = generic_functions.flatten_list(self.mice)

        elif session_list is not None:
            self.sessions = session_list

    def get_next_session(self):
        """gets the next session for reference frame building"""
        for s in self.sessions:
            if 'ref.npy' not in os.listdir(s.path):
                if any('loom' in fname for fname in os.listdir(s.path)):
                    return viewer.Viewer(s.path)
