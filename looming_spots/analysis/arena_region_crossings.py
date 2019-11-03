import numpy as np

from looming_spots.preprocess.normalisation import normalised_shelter_front


def is_entry(sample, place_of_entry, place_of_exit):
    if place_of_entry[sample+1]:
        if place_of_exit[sample] == 1:
            return True
    return False


def get_next_entry(start, place_of_entry, place_of_exit):
    tz_transitions = np.where(np.diff(place_of_entry))[0]
    for t in tz_transitions:
        if t < start:
            continue
        if is_entry(t, place_of_entry, place_of_exit):
            return t


def get_next_entry_from_track(normalised_x_track, context, place_of_entry_key, place_of_exit_key, start):
    """

    :param normalised_x_track:
    :param context:
    :param place_of_entry_key:
    :param place_of_exit_key:
    :param start:
    :return:
    """
    shelter_front = normalised_shelter_front(context)
    tz_boundary = 0.6
    positions = {
        'shelter': normalised_x_track < shelter_front,
        'middle': np.logical_and((tz_boundary > normalised_x_track), (normalised_x_track > shelter_front)),
        'tz': normalised_x_track > tz_boundary,
    }
    return get_next_entry(start, positions[place_of_entry_key], positions[place_of_exit_key])


def is_tz_entry(sample, tz, middle):
    """

    :param sample:
    :param tz: boolean array of tz occupance
    :param middle: boolean array of middle region occupance
    :return bool:
    """
    return is_entry(sample, tz, middle)


def is_shelter_entry(sample, shelter, middle):
    """

    :param sample:
    :param tz: boolean array of tz occupance
    :param middle: boolean array of middle region occupance
    :return bool:
    """
    return is_entry(sample, shelter, middle)


def get_next_tz_entry(start, tz, middle):
    get_next_entry(start, tz, middle)

