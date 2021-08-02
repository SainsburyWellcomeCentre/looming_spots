import numpy as np

from looming_spots import constants


def normalise_x_track(x_track, context):

    left_wall_pixel = constants.context_params[context].left
    right_wall_pixel = constants.context_params[context].right

    arena_length = right_wall_pixel - left_wall_pixel
    normalised_track = (x_track - left_wall_pixel) / arena_length

    if constants.context_params[context].flip:
        return 1 - normalised_track

    return normalised_track


def normalised_shelter_front(context):
    house_front_raw = constants.context_params[context].house_front
    house_front_normalised = normalise_x_track(house_front_raw, context)
    # print(house_front_normalised)
    return house_front_normalised

