import os
from ploom import mouse, session
from looming_spots.analysis import extract_looms
from datetime import datetime

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
    context = extract_looms.get_context_from_stimulus_mat(session_directory)
    protocol = extract_looms.get_session_label(session_directory)
    return context, date, protocol
