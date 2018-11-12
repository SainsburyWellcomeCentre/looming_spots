import os
from datetime import datetime

import looming_spots.preprocess.extract_looms
import looming_spots.util.generic_functions
from looming_spots.db import session

from looming_spots.db.paths import PROCESSED_DATA_DIRECTORY
from looming_spots.db import session_group, experiment
import warnings

class NotExtractedError(Exception):
    pass


def load_sessions(mouse_id):
    sessions_path = os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)
    session_list = []
    if os.path.isdir(sessions_path):
        for s in os.listdir(sessions_path):
            if not looming_spots.util.generic_functions.is_datetime(s):
                continue
            date = datetime.strptime(s, '%Y%m%d_%H_%M_%S')
            s = session.Session(dt=date, mouse_id=mouse_id)
            session_list.append(s)
        return sorted(session_list)
    warnings.warn('the mouse: {} has not been processed'.format(mouse_id))
    #raise(NotExtractedError('{}'.format(mouse_id)))


def load_experiment_from_groups(groups, group_labels, test_type='post'):
    sgs = []
    for ids, label in zip(groups, group_labels):
        sessions = []
        for mid in ids:
            ms = session_group.MouseSessionGroup(mid)
            print(ms.mouse_id)
            if test_type == 'pre':
                s = ms.nth_pre_test(0)
            elif test_type == 'post':
                s = ms.nth_post_test(0)
            sessions.append(s)
        sgs.append(session_group.SessionGroup(sessions, group_key=label))
    exp = experiment.Experiment(sgs)
    return exp