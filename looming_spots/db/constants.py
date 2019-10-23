import os
from collections import namedtuple
from datetime import datetime

FRAME_RATE = 30
VIDEO_SHAPE = (480, 640)
ARENA_SIZE_CM = 50

CLASSIFICATION_WINDOW_END = 350  # 345
CLASSIFICATION_SPEED = (
    -0.021
)  # original was-0.027, then -0.024 changed to 0.021 for contrasts
CLASSIFICATION_SPEED_FIBER_ATTACHED = -0.024
SPEED_THRESHOLD = -0.01
CLASSIFICATION_LATENCY = 5
END_OF_CLASSIFICATION_WINDOW = 550

STIMULUS_ONSETS = [
    200,
    228,
    256,
    284,
    312,
]  # TODO: make this generic and automatic not hardcode
LOOMING_STIMULUS_ONSET = STIMULUS_ONSETS[0]
CLASSIFICATION_WINDOW_START = STIMULUS_ONSETS[0]
N_LOOMS_PER_STIMULUS = len(STIMULUS_ONSETS)

N_HABITUATION_LOOMS = 120

ContextParams = namedtuple(
    "ContextParams", ["left", "right", "house_front", "flip"]
)

A2 = ContextParams(28, 538, 445, True)
A9 = ContextParams(23, 608, 490, True)
C = ContextParams(0, 600, 85, False)
A9_auditory = ContextParams(46, 597, 484, True)

AUDITORY_STIMULUS_CHANNEL_ADDED_DATE = datetime(2019, 1, 25)

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


def get_processed_mouse_directory(mouse_id):
    return os.path.join(PROCESSED_DATA_DIRECTORY, mouse_id)


def get_raw_path(mouse_id):
    return os.path.join(RAW_DATA_DIRECTORY, mouse_id)


HEAD_DIRECTORY = "/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/"
RAW_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "raw_data")
PROCESSED_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "processed_data")
FIGURE_DIRECTORY = "/home/slenzi/figures/"
OLD_RAW_DIRECTORY = (
    "/home/slenzi/spine_shares/loomer/srv/glusterfs/imaging/l/loomer/raw_data/"
)
