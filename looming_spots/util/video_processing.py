import os
import subprocess
import sys

import cv2
import numpy as np
import scipy.misc
import skvideo
import skvideo.io
import pims

import looming_spots.preprocess
from looming_spots.constants import LOOM_ONSETS


def load_video_from_path(vid_path):
    rdr = skvideo.io.vreader(vid_path)
    vid = load_video_from_rdr(rdr, ref=None)
    return vid


def load_video_from_rdr(rdr, ref=None):
    video = []
    if ref is not None:
        video.append(ref)
    for frame in rdr:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video.append(gray_frame)
    return np.array(video)


def save_frame_as_array(frame, directory):
    ref_array = np.mean(frame, axis=2)
    ref_path = os.path.join(directory, "ref.npy")
    print("refarray:{}".format(ref_array))
    print("saving reference frame to: {}".format(ref_path))
    np.save(ref_path, ref_array)


def save_frame_as_png(frame, directory):
    ref_array = np.mean(frame, axis=2)
    save_fpath = os.path.join(directory, "ref.png")
    print("saving reference frame to: {}".format(save_fpath))
    scipy.misc.imsave(save_fpath, ref_array, format="png")


def get_frame(rdr_path, idx):
    idx = int(idx)
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i == idx:
            return frame


def get_frames(rdr_path, idx):
    frames = []
    rdr = skvideo.io.vreader(rdr_path)
    for i, frame in enumerate(rdr):
        if i in idx:
            frames.append(frame)
    return frames


def save_video(video, path):
    skvideo.io.vwrite(path, video)


def crop_video(video, width, height, origin=(0, 0)):
    n_frames = video.shape[0]
    new_video = np.full(
        (n_frames, height - origin[1], width - origin[0]), np.nan
    )

    for i, frame in enumerate(video):
        new_video[i, :, :] = frame[origin[1] : height, origin[0] : width]
    return new_video


def plot_loom_on_video(video, radius_profile,track):

    pts_list = []
    for pos in zip(track[0].astype(int), track[1].astype(int)):
        pts_list.append(pos)

    new_video = np.empty_like(video)
    for i, (frame, radius) in enumerate(zip(video, radius_profile)):
        overlay = frame.copy()
        cv2.circle(frame, (150, 120), int(radius), (0, 0, 0), -1)
        for x in range(i):
            cv2.line(frame, pts_list[:-1][x], pts_list[1:][x], (0, 0, 0), 5)
        alpha=0.4
        new_image=cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        new_video[i, :, :] = new_image

    return new_video


def plot_track_on_video(video, track):
    new_video = np.empty_like(video)
    for i, (frame, radius) in enumerate(zip(video, track)):
        cv2.polylines(frame, track[:600], False)
        new_video[i, :, :] = frame

    return new_video


def loom_radius_profile(n_frames):  # TODO: make this using the maths
    radius_profile = np.zeros(n_frames)
    for onset in LOOM_ONSETS:
        onset+=1
        radius_profile[onset : onset + 7] = np.linspace(5, 140, 7)
        radius_profile[onset + 7 : onset + 14] = np.ones(7)*140
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
        new_vid = plot_loom_on_video(vid, looming_stimulus_radius_profile, track)

        save_video(new_vid, path_out)
    return load_video_from_path(path_out)


def compare_pd_and_video(directory):
    pd_trace = looming_spots.preprocess.io.load_pd_on_clock_ups(directory)
    n_samples_pd = len(pd_trace)
    video_path = os.path.join(directory, "camera.mp4")
    n_samples_video = len(pims.Video(video_path))

    print(
        "pd found {} samples, there are {} frames in the video".format(
            n_samples_pd, n_samples_video
        )
    )

    if n_samples_pd != n_samples_video:
        if n_samples_pd == 0:
            raise NoPdError
        n_samples_ratio = round(n_samples_pd / n_samples_video, 2)
        if n_samples_ratio.is_integer():
            print("downsampling by factor {}".format(n_samples_ratio))
            print(n_samples_pd, n_samples_video, n_samples_ratio)
            downsampled_ai = pd_trace[:: int(n_samples_ratio)]
            save_path = os.path.join(directory, "AI_corrected")
            np.save(save_path, downsampled_ai)


def convert_to_mp4(
    name, directory, remove_avi=False
):  # TODO: remove duplication
    mp4_path = os.path.join(directory, name[:-4] + ".mp4")
    avi_path = os.path.join(directory, name)
    if os.path.isfile(mp4_path):
        print("{} already exists".format(mp4_path))
        if remove_avi:  # TEST this
            if os.path.isfile(avi_path):
                print(
                    "{} present in processed data, deleting...".format(
                        avi_path
                    )
                )
                os.remove(avi_path)
    else:
        print("Creating: " + mp4_path)
        convert_avi_to_mp4(avi_path)


def convert_avi_to_mp4(avi_path):
    mp4_path = avi_path[:-4] + ".mp4"
    print("avi: {} mp4: {}".format(avi_path, mp4_path))

    supported_platforms = ["linux", "windows"]

    if sys.platform == "linux":
        cmd = "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
            avi_path, mp4_path
        )

    elif sys.platform == "windows":  # TEST: on windows
        cmd = "ffmpeg -i {} -c:v mpeg4 -preset fast -crf 18 -b 5000k {}".format(
            avi_path, mp4_path
        ).split(
            " "
        )

    else:
        raise (
            OSError(
                "platform {} not recognised, expected one of {}".format(
                    sys.platform, supported_platforms
                )
            )
        )

    subprocess.check_call([cmd], shell=True)


def upsample_video(path):
    vid = skvideo.io.vread(path)
    new_video = []
    for i, frame in enumerate(vid):
        new_video.append(frame)
        new_video.append(frame)
    return np.array(new_video)


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
            print('aleady file')
            return
    print('extracting......')
    print(path_out)
    extract_video(
        path_in,
        path_out,
        loom_start - n_samples_before,
        loom_start + n_samples_after,
    )


def extract_video(fin_path, fout_path, start, end):
    v = pims.Video(fin_path)
    out_video = v[start:end]
    print(f'writign tp {fout_path}')
    skvideo.io.vwrite(fout_path, out_video)


class NoPdError(Exception):
    pass


class NoProcessedVideoError(Exception):
    def __str__(self):
        print("there is no mp4 video")
