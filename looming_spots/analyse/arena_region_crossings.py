import numpy as np
from looming_spots.constants import NORMALISED_SHELTER_FRONT

"""
Functions for labeling mouse position and defining crossings from each region 
into the others for analysis of responses to aversive stimuli.

 ---------
|         |
|         |
|    TZ   |
|         |
------------ (30cm)
|         |
|         |
|  MIDDLE |
|         |
------------ (10cm)
| SHELTER |
|         |
 ---------

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
    print("no entries found")


def get_next_entry_from_track(
    normalised_x_track, place_of_entry_key, place_of_exit_key, start
):
    """

    :param normalised_x_track:
    :param place_of_entry_key:
    :param place_of_exit_key:
    :param start:
    :return:
    """
    shelter_boundary = 0.2
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


def arena_entry_bools_dictionary(normalised_x_track):
    shelter_boundary = NORMALISED_SHELTER_FRONT
    tz_boundary = 0.6
    positions = {
        "shelter": normalised_x_track < shelter_boundary,
        "middle": np.logical_and(
            (tz_boundary > normalised_x_track),
            (normalised_x_track > shelter_boundary),
        ),
        "tz": normalised_x_track > tz_boundary,
    }
    return positions


def get_all_tz_entries(normalised_x_track):
    """
    Finds the sample idx where the threat zone (TZ) is entered by the mouse.
    :param normalised_x_track:
    :return:
    """
    entries = arena_entry_bools_dictionary(normalised_x_track)
    shelter_entries = np.where(np.diff(entries["shelter"].astype(int)) == 1)[0]
    tz_entries = np.where(np.diff(entries["tz"].astype(int)) == 1)[0]
    middle_entries = np.where(np.diff(entries["middle"].astype(int)) == 1)[0]
    track_starts = []

    for i, tz_entry in enumerate(tz_entries):
        try:
            first_next_middle = min(
                middle_entries[middle_entries > tz_entry],
                key=lambda x: abs(x - tz_entry),
            )

            first_next_shelter = min(
                shelter_entries[shelter_entries > first_next_middle],
                key=lambda x: abs(x - first_next_middle),
            )
            first_next_tz = min(
                tz_entries[tz_entries > first_next_middle],
                key=lambda x: abs(x - first_next_middle),
            )
            if first_next_tz < first_next_shelter:
                continue
            else:
                track_starts.append(tz_entry)
        except Exception as e:
            print(e)
            return track_starts

    return track_starts

