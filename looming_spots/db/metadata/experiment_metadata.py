import os
import warnings

import scipy
import scipy.io
from configobj import ConfigObj
import numpy as np

from looming_spots.preprocess import photodiode

METADATA_PATH = 'metadata.cfg'
CONTEXT_B_SPOT_POSITION = 1240


def config_exists(directory):
    config_path = os.path.join(directory, METADATA_PATH)
    return True if os.path.isfile(config_path) else False


def load_metadata(directory):
    config_path = os.path.join(directory, METADATA_PATH)
    metadata = ConfigObj(config_path, encoding="UTF8", indent_type='    ', unrepr=False,
                         create_empty=True, write_empty_values=True)
    return metadata


def save_key_to_metadata(mtd, key, item):
    mtd[key] = item
    mtd.write()


def load_from_metadata(metadata, key):
    if key in metadata:
        return metadata[key]


def write_context_to_metadata(directory):  # TODO: deprecated
    mtd = load_metadata(directory)
    context = get_context_from_stimulus_mat(directory)
    mtd['context'] = context
    mtd.write()


def write_session_label_to_metadata(directory, loom_idx):  # TODO: deprecated
    mtd = load_metadata(directory)
    session_label = get_session_label_from_loom_idx(loom_idx)
    mtd['session_label'] = session_label
    mtd.write()


def initialise_metadata(directory):  # TODO: extract configobj boilerplate
    metadata = load_metadata(directory)
    metadata['video_name'] = './camera.mp4'
    left_frame_idx, right_frame_idx = _parse_ref_idx_from_exp_metadata(directory)
    context = get_context_from_stimulus_mat(directory)
    loom_idx, _ = photodiode.get_loom_idx_from_raw(directory)
    session_label = get_session_label_from_loom_idx(loom_idx)

    if left_frame_idx:
        metadata['reference_frame'] = {}
        metadata['reference_frame']['right'] = right_frame_idx
        metadata['reference_frame']['left'] = left_frame_idx

    metadata['context'] = context
    metadata['session_label'] = session_label
    metadata.write()


def get_loom_idx(directory):  # TODO: move to extract looms?
    mtd = load_metadata(directory)
    if 'loom_idx' not in load_metadata(directory):
        print('loom_idx not found in {}'.format(directory))
        loom_idx, _ = photodiode.get_loom_idx_from_raw(directory)
        save_key_to_metadata(mtd, 'loom_idx', list(loom_idx))
    loom_idx = np.array(load_from_metadata(mtd, 'loom_idx')).astype(int)
    return loom_idx


def _parse_ref_idx_from_exp_metadata(directory, fname='metadata.txt'):  # TODO: remove?
    import re
    file_path = os.path.join(directory, fname)
    if not os.path.isfile(file_path):
        return None, None

    with open(file_path) as f:
        data = f.readlines()
        left, right = None, None
        for item in data:
            if 'right' in item:
                right = re.search(r'\d+', item).group()
            elif 'left' in item:
                left = re.search(r'\d+', item).group()
        print('left frame: {}, right_frame:{}'.format(left, right))
    return left, right


def get_context_from_stimulus_mat(directory):
    stimulus_path = os.path.join(directory, 'stimulus.mat')
    if os.path.isfile(stimulus_path):
        stimulus_params = scipy.io.loadmat(stimulus_path)['params']
        dot_locations = [x[0] for x in stimulus_params[0][0] if len(x[0]) == 2]  # only spot position has length 2

        return 'B' if any(CONTEXT_B_SPOT_POSITION in x for x in dot_locations) else 'A'
    else:
        print('no stimulus parameters file')
        return 'n/a'


def get_context(directory):
    mtd = load_metadata(directory)
    if 'context' not in mtd:
        return get_context_from_stimulus_mat(directory)
    return mtd['context']


def get_session_label_from_loom_idx(loom_idx, n_habituation_looms=120):

    warnings.warn('deprecated')

    print("{} looms detected".format(len(loom_idx)))

    if len(loom_idx) == 0:
        return 'no_stimuli'
    elif len(loom_idx) == n_habituation_looms:
        return 'habituation_only'
    elif len(loom_idx) < n_habituation_looms:
        return 'test_only'
    elif len(loom_idx) > n_habituation_looms:
        return 'habituation_and_test'
