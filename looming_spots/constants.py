import pathlib
from datetime import datetime


FRAME_RATE = 30
VIDEO_SHAPE = (480, 640)

ARENA_SIZE_CM = 50
SHELTER_SIZE_CM = 10
ARENA_WIDTH_PX = 240
ARENA_LENGTH_PX = 600
NORMALISED_SHELTER_FRONT = 0.2

BOX_CORNER_COORDINATES = [[0, 240], [0, 0], [600, 240], [600, 0]]

CLASSIFICATION_WINDOW_END = 350
SPEED_THRESHOLD = -0.017 * FRAME_RATE * ARENA_SIZE_CM
CLASSIFICATION_LATENCY = 5
END_OF_CLASSIFICATION_WINDOW = 550
FREEZE_BUFFER_FRAMES = (
    12  # number of frames after loom onset to ignore in classifying freeze
)


LOOM_ONSETS = [200, 228, 256, 284, 312]
LOOM_ONSETS_S = [(s - 200.0) / 30 for s in LOOM_ONSETS]
LOOMING_STIMULUS_ONSET_SAMPLE = LOOM_ONSETS[0]

CLASSIFICATION_WINDOW_START = LOOM_ONSETS[0]
N_LOOMS_PER_STIMULUS = len(LOOM_ONSETS)

N_LSIE_LOOMS = 120

AUDITORY_STIMULUS_CHANNEL_ADDED_DATE = datetime(2019, 1, 25)
HEADBAR_REMOVED_DATE = datetime(2018, 2, 23)

N_SAMPLES_BEFORE = 200
N_SAMPLES_AFTER = 400
TRACK_LENGTH = N_SAMPLES_BEFORE + N_SAMPLES_AFTER

DLC_DIRECTORY = "/home/slenzi/winstor/margrie/slenzi/dlc/"

HEAD_DIRECTORY = pathlib.Path(
    "/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/"
)
EXPERIMENTAL_RECORDS_PATH = "~/Downloads/updated_loom_sheet_format.csv"
RAW_DATA_DIRECTORY = HEAD_DIRECTORY / "raw_data"
PROCESSED_DATA_DIRECTORY = HEAD_DIRECTORY / "processed_data"

FIGURE_DIRECTORY = HEAD_DIRECTORY / "figures"
DF_PATH = HEAD_DIRECTORY / "cricket_dfs"
TRANSFORMED_DF_PATH = HEAD_DIRECTORY / "transformed_cricket_paths"


ORDERED_ACQUISITION_CHANNEL_LABELS = [
    "photodiode",
    "clock",
    "auditory_stimulus",
    "photodetector",
    "led211",
    "led531",
]

PROCESSED_OUTPUT_VARIABLE_LABELS = [
    "signal",
    "background",
    "bg_fit",
    "delta_f",
]

flatui = ["#9b59b6", "#3498db", "#95a5a6"]