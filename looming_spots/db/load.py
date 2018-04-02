import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session
from looming_spots.db.metadata.experiment_metadata import get_context, get_session_label_from_loom_idx, get_loom_idx

from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY
from looming_spots.db import session_group, experiment


def load_sessions(mouse_id):
    sessions_path = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    for s in os.listdir(sessions_path):
        if not looming_spots.util.generic_functions.is_datetime(s):
            continue
        date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
        s_path = os.path.join(sessions_path, s)
        context = get_context(s_path)
        loom_idx = get_loom_idx(s_path)
        protocol = get_session_label_from_loom_idx(loom_idx)
        s = session.Session(dt=date, protocol=protocol, stimulus='looming_spot', context=context, mouse_id=mouse_id)
        session_list.append(s)
        print('session path: {}, context: {}'.format(s.path, s.context))
    return session_list


def load_experiment_from_groups(groups, group_labels):
    sgs = []
    for ids, label in zip(groups, group_labels):
        sessions = []
        for mid in ids:
            mouse_sessions = load_sessions(mid)
            s = max(mouse_sessions)
            sessions.append(s)
        sgs.append(session_group.SessionGroup(sessions, group_key=label))
    exp = experiment.Experiment(sgs)
    return exp