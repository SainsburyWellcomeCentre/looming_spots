import cv2
import numpy as np


def convert_loom_to_triggers():
    pass


def load_frame(rdr, idx):
    for i, frame in enumerate(rdr):
        if i == idx:
            return frame


def get_mouse_centre(img_uint8):
    contours = extract_contours(img_uint8)
    mouse_cnt = get_largest(contours)
    return np.mean(mouse_cnt, axis=0)[0]


def extract_contours(silhouette):
    contours_data = cv2.findContours(np.copy(silhouette), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    img, contours, hierarchy = contours_data  # FIXME: normally should not return image
    return contours


def get_largest(contour_list):
    contour_areas = [cv2.contourArea(contour) for contour in contour_list]
    largest_idx = np.argmax(np.array(contour_areas))
    return contour_list[largest_idx]


def get_track(reader, crop_y=[95, 325], crop_x=[100, 900], start_frame=3000):
    """

    computes track from reader object, note: this may require re-instantiation of the reader object every time.

    :param reader:
    :param crop_y:
    :param crop_x:
    :param start_frame:
    :return:
    """
    track = []

    for j, frame in enumerate(reader):
        if j > start_frame:
            cropped_frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
            grayscale = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)<1
            silhouette = grayscale.astype(np.uint8)
            location = get_mouse_centre(silhouette)
            track.append(location)
    return track


def get_loom_artefact(reader, start_frame_idx, loom_roi_coordinates=[(350, 400), (600, 650)]):
    """

    extract the loom from video for identifying stimulus starts

    :param reader:
    :param start_frame_idx:
    :param loom_roi_coordinates:
    :return:
    """
    loom_artefact = []
    xs, xe = loom_roi_coordinates[0]
    ys, ye = loom_roi_coordinates[1]
    for j, frame in enumerate(reader):
        if j > start_frame_idx:
            cropped_frame = frame[xs:xe, ys:ye]
            grayscale = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            value = np.mean(grayscale)
            loom_artefact.append(value)
    return np.array(loom_artefact)


def get_loom_train_starts(reader, start_frame_idx=3000, threshold=15, off_samples=20):
    """

    example usage:
    >>> path = './looming_adaptation/CA_30_2/170726/20170726_CA30_2_posttest_newLoomStadium-Camera.h264'
    >>> reader = skvideo.io.vreader(path)
    >>> loom_locs = get_loom_train_starts(reader)


    :param reader:
    :param start_frame_idx:
    :param threshold:
    :param off_samples:
    :return:
    """

    loom_artefact = get_loom_artefact(reader, start_frame_idx)
    baseline = np.median(loom_artefact)
    loom_locs = loom_artefact<(baseline-threshold)
    loom_transitions = np.where(np.diff(loom_locs))[0]
    loom_train_starts = []
    for i, (this_transition, next_transition) in enumerate(zip(loom_transitions, loom_transitions[1:])):
        if i == 0 or (next_transition - this_transition > off_samples):
            loom_train_starts.append(next_transition)
    return loom_train_starts


def extract_responses(loom_starts, track, n_samples_pre=250, n_samples_post=500):
    """

    for each trigger provided this cuts out the track before and after for extracting trials
    :param loom_starts:
    :param track:
    :param n_samples_pre:
    :param n_samples_post:
    :return:
    """
    n_samples_total = n_samples_pre + n_samples_post
    responses = np.zeros((n_samples_total, len(loom_starts)))
    for i, loc in enumerate(loom_starts):
        s = loc - n_samples_pre
        e = min(loc + n_samples_post, len(track))
        responses[0:(e-s), i] = np.array(track)[s:e, 0]
    return responses


def extract_stimuli_post_hoc(loom_starts, track, n_samples_pre=250, n_samples_post=500):
    """
    for each trigger provided this cuts out the stimuli (for visualising the looms afterwards, not for finding the
    looms

    :param loom_starts:
    :param track:
    :param n_samples_pre:
    :param n_samples_post:
    :return:
    """
    n_samples_total = n_samples_pre + n_samples_post
    stimuli = np.zeros((n_samples_total, len(loom_starts)))
    for i, loc in enumerate(loom_starts):
        s = loc - n_samples_pre
        e = min(loc + n_samples_post, len(track))
        stimuli[0:(e-s), i] = np.array(track)[s:e, 0]
    return stimuli
