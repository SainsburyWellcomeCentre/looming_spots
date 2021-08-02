import os
from collections import namedtuple
from datetime import datetime

from configobj import ConfigObj

FRAME_RATE = 30
VIDEO_SHAPE = (480, 640)
ARENA_SIZE_CM = 50
SHELTER_FRONT = 0.2

CLASSIFICATION_WINDOW_END = 350  # 345
SPEED_THRESHOLD = -0.017 * FRAME_RATE * ARENA_SIZE_CM
CLASSIFICATION_LATENCY = 5
END_OF_CLASSIFICATION_WINDOW = 550
BOX_CORNER_COORDINATES = [[0, 240], [0, 0], [600, 240], [600, 0]]

LOOM_ONSETS = [200, 228, 256, 284, 312]

# TODO: make this generic and automatic not hardcode
LOOMING_STIMULUS_ONSET = LOOM_ONSETS[0]
CLASSIFICATION_WINDOW_START = LOOM_ONSETS[0]
N_LOOMS_PER_STIMULUS = len(LOOM_ONSETS)

N_LSIE_LOOMS = 120

ContextParams = namedtuple(
    "ContextParams", ["left", "right", "house_front", "flip"]
)

A2 = ContextParams(28, 538, 445, True)
A9 = ContextParams(23, 608, 490, True)
C = ContextParams(0, 600, 85, False)
A9_auditory = ContextParams(46, 597, 484, True)

AUDITORY_STIMULUS_CHANNEL_ADDED_DATE = datetime(2019, 1, 25)
HEADBAR_REMOVED_DATE = datetime(2018, 2, 23)

context_params = {
    "A2": A2,
    "A9": A9,
    "C": C,
    "B": C,
    "A": A2,
    "A9_auditory": A9_auditory,
    "A10": A9_auditory,
}

METADATA_PATH = "./metadata.cfg"
TRACK_LENGTH = 600

N_SAMPLES_BEFORE = 200
N_SAMPLES_AFTER = 400
N_SAMPLES_TO_SHOW = N_SAMPLES_BEFORE + N_SAMPLES_AFTER


def get_processed_mouse_directory(mouse_id):
    return os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)


def get_raw_path(mouse_id):
    return os.path.join(RAW_DATA_DIRECTORY, mouse_id)


HEAD_DIRECTORY = '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/'
FIGURE_DIRECTORY = '~/lsie/docs/figures/'
OLD_RAW_DIRECTORY = (
    "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/raw_data/"
)
FILE_PATH = '~/Downloads/updated_loom_sheet_format.csv'
RAW_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "raw_data")
PROCESSED_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "processed_data")
CONTEXT_B_SPOT_POSITION = 1240


def load_metadata(directory):
    config_path = os.path.join(directory, METADATA_PATH)
    metadata = ConfigObj(
        config_path,
        encoding="UTF8",
        indent_type="    ",
        unrepr=False,
        create_empty=True,
        write_empty_values=True,
    )
    return metadata


