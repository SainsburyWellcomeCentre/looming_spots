import os
from datetime import datetime

import looming_spots.db.metadata.experiment_metadata
import looming_spots.preprocess.extract_looms
import looming_spots.preprocess.photodiode
from looming_spots.db import mouse, session
from looming_spots.preprocess import extract_looms

DIRECTORY_DEFAULT = '/home/slenzi/spine_shares/imaging/s/slenzi/data_working_copy/'


def get_experiments_from_source(directory):
    all_mice = []
    for mouse_id in os.listdir(directory):
        m = mouse.Mouse(mouse_id)
        sessions_path = os.path.join(directory, mouse_id)
        for session_folder_name in os.listdir(sessions_path):
            context, date, protocol = get_session_info(session_folder_name, sessions_path)
            s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context)
            m.sessions.append(s)
        all_mice.append(m)
    return all_mice


def get_session_info(session_folder_name, sessions_path):
    date = datetime.strptime(session_folder_name, '%Y%m%d_%H_%M_%S')
    session_directory = os.path.join(sessions_path, session_folder_name)
    context = looming_spots.db.metadata.experiment_metadata.get_context_from_stimulus_mat(session_directory)
    loom_idx = extract_looms.get_loom_idx_from_raw(session_directory)
    protocol = looming_spots.db.metadata.experiment_metadata.get_session_label_from_loom_idx(loom_idx)
    return context, date, protocol



def load_mice_2(directory):
    all_mice = []
    for mouse_id in os.listdir(directory):
        print('mouse_id {}'.format(mouse_id))
        m = mouse.Mouse(mouse_id)
        sessions_path = os.path.join(directory, mouse_id)
        print('sessions_path: {}'.format(sessions_path))
        for s in os.listdir(sessions_path):
            if not looming_spots.preprocess.extract_looms.is_datetime(s):
                continue
            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s_path = os.path.join(sessions_path, s)
            if not any(['loom0' == x for x in os.listdir(s_path)]):
                continue
            context = get_context(s_path)
            print(context)
            if context == 'n/a':
                return all_mice  # FIXME:
            loom_idx = extract_looms.get_loom_idx_from_raw(s_path)
            protocol = get_session_label_from_loom_idx(loom_idx)
            s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context)
            m.sessions.append(s)
        all_mice.append(m)
    return all_mice


def load_mice_from_store():
    import dumb
    root = '/home/slenzi/code/python/build/loom/json_store'
    json = dumb.backends.json_m.JSONBackend(root=root)
    for exp_meta_fname in os.listdir(root):
        m = json.load(exp_meta_fname)
        print(m)
        return m
