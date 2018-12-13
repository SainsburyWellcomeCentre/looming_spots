from collections import namedtuple
FRAME_RATE = 30
VIDEO_SHAPE = (480, 640)
ARENA_SIZE_CM = 50

CLASSIFICATION_WINDOW_END = 350  # 345
CLASSIFICATION_SPEED = -0.027
SPEED_THRESHOLD = -0.01
CLASSIFICATION_LATENCY = 5
END_OF_CLASSIFICATION_WINDOW = 550

STIMULUS_ONSETS = [200, 228, 256, 284, 312]  # TODO: make this generic and automatic not hardcode
LOOMING_STIMULUS_ONSET = STIMULUS_ONSETS[0]
CLASSIFICATION_WINDOW_START = STIMULUS_ONSETS[0]
N_LOOMS_PER_STIMULUS = len(STIMULUS_ONSETS)

N_HABITUATION_LOOMS = 120

ContextParams = namedtuple('ContextParams', ['left', 'right', 'house_front', 'flip'])

A2 = ContextParams(28, 538, 445, True)
A9 = ContextParams(23, 608, 490, True)
C = ContextParams(0, 600, 85, False)

context_params = {
                  'A2': A2,
                  'A9': A9,
                  'C':  C,
                  'B':  C,
                  'A':  A2,
                  }

METADATA_PATH = './metadata.cfg'
TRACK_LENGTH = 600