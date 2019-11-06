import numpy as np

from looming_spots.preprocess.normalisation import normalised_shelter_front

"""
Functions for labelling mouse position and defining crossings from each region 
into the others for analysis of responses to aversive stimuli.

 --------
|        |
|        |
|   TZ   |
|        |
------------ (30cm)
|        |
|        |
| MIDDLE |
|        |
------------ (10cm)
|SHELTER |
|        |
 --------

"""


def is_entry(sample, place_of_entry, place_of_exit):
    """
    checks if the mouse has moved from the place of exit to the place of entry at a specific sample

    :param sample: sample number to check
    :param bool_array place_of_entry: box region that the mouse is entering (1 if in region, 0 if not)
    :param bool_array place_of_exit: box region that the mouse is leaving (1 if in region, 0 if not)
    :return:
    """
    if place_of_entry[sample + 1]:
        if place_of_exit[sample] == 1:
            return True
    return False


def get_next_entry(start, place_of_entry, place_of_exit):
    """
    returns the next timepoint in samples that the mouse has moved from the place of exit to the place of entry

    :param start: starting point in the track in samples
    :param place_of_entry: box region that the mouse is entering
    :param place_of_exit: box region that the mouse is leaving
    :return:
    """

    region_crossings = np.where(np.diff(place_of_entry))[0]
    for t in region_crossings:
        if t < start:
            continue
        if is_entry(t, place_of_entry, place_of_exit):
            return t


def get_next_entry_from_track(
    normalised_x_track, context, place_of_entry_key, place_of_exit_key, start
):
    """

    :param normalised_x_track:
    :param context:
    :param place_of_entry_key:
    :param place_of_exit_key:
    :param start:
    :return:
    """
    shelter_boundary = normalised_shelter_front(context)
    tz_boundary = 0.6
    positions = {
        "shelter": normalised_x_track < shelter_boundary,
        "middle": np.logical_and(
            (tz_boundary > normalised_x_track),
            (normalised_x_track > shelter_boundary),
        ),
        "tz": normalised_x_track > tz_boundary,
    }
    return get_next_entry(
        start, positions[place_of_entry_key], positions[place_of_exit_key]
    )
