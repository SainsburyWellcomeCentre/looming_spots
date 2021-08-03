import os
from datetime import datetime


FRAME_RATE = 30
VIDEO_SHAPE = (480, 640)
ARENA_SIZE_CM = 50
SHELTER_FRONT = 0.2
ARENA_WIDTH_PX = 240
ARENA_LENGTH_PX = 600
BOX_CORNER_COORDINATES = [[0, 240], [0, 0], [600, 240], [600, 0]]

CLASSIFICATION_WINDOW_END = 350
SPEED_THRESHOLD = -0.017 * FRAME_RATE * ARENA_SIZE_CM
CLASSIFICATION_LATENCY = 5
END_OF_CLASSIFICATION_WINDOW = 550

LOOM_ONSETS = [200, 228, 256, 284, 312]
LOOMING_STIMULUS_ONSET = LOOM_ONSETS[0]
CLASSIFICATION_WINDOW_START = LOOM_ONSETS[0]
N_LOOMS_PER_STIMULUS = len(LOOM_ONSETS)

N_LSIE_LOOMS = 120

AUDITORY_STIMULUS_CHANNEL_ADDED_DATE = datetime(2019, 1, 25)
HEADBAR_REMOVED_DATE = datetime(2018, 2, 23)

TRACK_LENGTH = 600

N_SAMPLES_BEFORE = 200
N_SAMPLES_AFTER = 400
N_SAMPLES_TO_SHOW = N_SAMPLES_BEFORE + N_SAMPLES_AFTER


HEAD_DIRECTORY = '/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/'
FIGURE_DIRECTORY = '~/lsie/docs/figures/'
EXPERIMENTAL_RECORDS_PATH = '~/Downloads/updated_loom_sheet_format.csv'
RAW_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "raw_data")
PROCESSED_DATA_DIRECTORY = os.path.join(HEAD_DIRECTORY, "processed_data")


ORDERED_ACQUISITION_CHANNEL_LABELS = [
    "photodiode",
    "clock",
    "auditory_stimulus",
]
PROCESSED_OUTPUT_VARIABLE_LABELS = []
