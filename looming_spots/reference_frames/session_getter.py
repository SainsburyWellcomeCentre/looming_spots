import os

from looming_spots.reference_frames import viewer
from looming_spots.util import generic_functions
from looming_spots.db.load import load_sessions
from looming_spots.tracking.pyper_backend import auto_track


class SessionGetter(object):
    def __init__(self, mouse_id_list=None, session_list=None):
        if mouse_id_list is not None:
            self.mice = filter(None, [load_sessions(mid) for mid in mouse_id_list])  #FIXME: hacky
            self.sessions = generic_functions.flatten_list(self.mice)

        elif session_list is not None:
            self.sessions = session_list

    def get_next_ref_building_session(self):
        """gets the next session for reference frame building"""
        for s in self.sessions:
            if any(t.trial_type == 'habituation' for t in s.trials):
                if 'habituation_ref.npy' not in os.listdir(s.path):
                    return viewer.Viewer(s.path)
            if any(t.trial_type == 'test' for t in s.trials):
                if 'test_ref.npy' not in os.listdir(s.path):
                    return viewer.Viewer(s.path)

    def get_next_tracking_session(self):
        """gets the next session ready for tracking"""
        for s in self.sessions:
            if any(t.trial_type == 'habituation' for t in s.trials):
                if 'habituation_ref.npy' not in os.listdir(s.path):
                    return s.path
            if any(t.trial_type == 'test' for t in s.trials):
                if 'test_ref.npy' not in os.listdir(s.path):
                    return s.path
            # if 'ref.npy' in os.listdir(s.path):
            #     loom_paths = [os.path.join(s.path, fname) for fname in os.listdir(s.path) if 'loom' in fname]
            #     if not any(os.path.isdir(path) for path in loom_paths):
            #         return s.path
        return None

    def get_next_session_start(self):
        for s in self.sessions:
            if 'time_of_mouse_entry' not in s.metadata:
                print(s.path)
                return viewer.Viewer(s.path, video_fname='camera.mp4')

    def track_all(self):
        untracked_sessions = True
        while untracked_sessions:
            path = self.get_next_tracking_session()
            print('tracking {}'.format(path))
            auto_track.pyper_cli_track(path)
            if path is None:
                untracked_sessions = False
