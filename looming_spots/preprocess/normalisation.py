import os

import numpy as np
import pandas as pd

from looming_spots.db import constants


def normalise_x_track(x_track, context):

    left_wall_pixel = constants.context_params[context].left
    right_wall_pixel = constants.context_params[context].right

    arena_length = right_wall_pixel - left_wall_pixel
    normalised_track = (x_track - left_wall_pixel) / arena_length

    if constants.context_params[context].flip:
        return 1 - normalised_track

    return normalised_track


def load_normalised_track(loom_folder, context):
    x_track, _ = load_raw_track(loom_folder)
    norm_x = normalise_x_track(x_track, context=context)
    return norm_x


def load_normalised_speeds(loom_folder, context):
    x_track = load_normalised_track(loom_folder, context)
    norm_speeds = np.diff(x_track)
    return norm_speeds


def normalised_shelter_front(context):
    house_front_raw = constants.context_params[context].house_front
    house_front_normalised = normalise_x_track(house_front_raw, context)
    # print(house_front_normalised)
    return house_front_normalised


def load_raw_track(loom_folder, name="tracks.csv"):
    track_path = os.path.join(loom_folder, name)
    df = pd.read_csv(track_path, sep="\t")
    x_pos = np.array(df["x_position"])
    y_pos = np.array(df["y_position"])
    return x_pos, y_pos
