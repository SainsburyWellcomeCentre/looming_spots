import os
import cv2
import numpy as np
import scipy.misc
import skvideo
import skvideo.io
import pims


STIMULUS_ONSETS = [200, 228, 256, 284, 312]


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


def plot_loom_on_video(video, radius_profile):
    new_video = np.empty_like(video)

    for i, (frame, radius) in enumerate(zip(video, radius_profile)):
        cv2.circle(frame, (150, 120), int(radius), -1)
        new_video[i, :, :] = frame

    return new_video


def loom_radius_profile(n_frames):  # TODO: make this using the maths
    radius_profile = np.zeros(n_frames)
    for onset in STIMULUS_ONSETS:
        radius_profile[onset : onset + 14] = np.linspace(5, 140, 14)
    return radius_profile


def loom_superimposed_video(path_in, path_out, width, height, origin):
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
        new_vid = plot_loom_on_video(vid, looming_stimulus_radius_profile)

        save_video(new_vid, path_out)
    return load_video_from_path(path_out)
