import os

import cv2
import numpy as np
from looming_spots.constants import LOOM_ONSETS
from looming_spots.util.video_processing import (
    load_video_from_path,
    crop_video,
    save_video,
    extract_video,
)


def add_track_overlay_to_video(video, track):
    new_video = np.empty_like(video)
    for i, (frame, radius) in enumerate(zip(video, track)):
        cv2.polylines(frame, track[:600], False)
        new_video[i, :, :] = frame

    return new_video


def loom_radius_profile(n_frames):  # TODO: make with proper equations
    radius_profile = np.zeros(n_frames)
    for onset in LOOM_ONSETS:
        onset += 1
        radius_profile[onset : onset + 7] = np.linspace(5, 140, 7)
        radius_profile[onset + 7 : onset + 14] = np.ones(7) * 140
    return radius_profile


def loom_superimposed_video(path_in, path_out, width, height, origin, track):
    """
    function for overlaying illustrative looming stimulus on video and cropping

    :param path_in:
    :param path_out:
    :param width:
    :param height:
    :param origin:
    :return:
    """

    if not os.path.isfile(path_out):
        vid = load_video_from_path(path_in)
        vid = crop_video(vid, width, height, origin)

        looming_stimulus_radius_profile = loom_radius_profile(len(vid))
        new_vid = plot_loom_on_video(
            vid, looming_stimulus_radius_profile, track
        )

        save_video(new_vid, path_out)
    return load_video_from_path(path_out)


def plot_loom_on_video(video, radius_profile, track):

    pts_list = []
    for pos in zip(track[0].astype(int), track[1].astype(int)):
        pts_list.append(pos)

    new_video = np.empty_like(video)
    for i, (frame, radius) in enumerate(zip(video, radius_profile)):
        overlay = frame.copy()
        cv2.circle(frame, (150, 120), int(radius), (0, 0, 0), -1)
        for x in range(i):
            cv2.line(frame, pts_list[:-1][x], pts_list[1:][x], (0, 0, 0), 5)
        alpha = 0.4
        new_image = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_video[i, :, :] = new_image

    return new_video


def extract_loom_video_trial(
    path_in,
    path_out,
    loom_start,
    n_samples_before=200,
    n_samples_after=400,
    overwrite=False,
):
    if not os.path.isfile(path_in):
        print(path_in)
        path_in = path_in.replace(".mp4", ".avi")
        path_in = path_in.replace("processed", "raw")
    loom_start = int(loom_start)
    if not overwrite:
        if os.path.isfile(path_out):
            print("aleady file")
            return
    print("extracting......")
    print(path_out)
    extract_video(
        path_in,
        path_out,
        loom_start - n_samples_before,
        loom_start + n_samples_after,
    )
