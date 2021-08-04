import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io
from cached_property import cached_property

from looming_spots.preprocess.normalisation import load_raw_track
from looming_spots.util import video_processing

N_FRAMES = 600


def get_digit_from_string(string):
    return int(re.search(r"\d+", string).group())


def digit_present(string):
    return any([x.isdigit() for x in string])


class Viewer(object):
    """
    This class allows browsing short videos to build reference frames manually. It allows the manual_tracking selection of left
    and right frames. 'left key': sets the index for the frame containing an empty left hand side of the arena, whereas
    'right key': sets the index of the right. 'Enter': triggers the writing of these indices and video paths to a
    text file called Metadata.txt, and also saves the composite frame. If there are a series of videos that change
    by increment only then 'w' and 'q' enable toggling between videos.
    """

    def __init__(self, directory, video=None, video_fname="loom0.h264"):
        self.frame_idx = 0
        self.video_fname = video_fname
        self.directory = directory
        self.video_idx = None
        self.scroll_step_size = 20
        self.click_step_size = 50
        if video is not None:  # FIXME:
            self.video = video
        elif video_fname:
            self.video_ext = "." + video_fname.split(".")[-1]
            self.video_name = video_fname.split(".")[0]
            if digit_present(self.video_name):
                self.video_idx = get_digit_from_string(self.video_name)
            self.video_fname_fmt = re.sub(r"\d+", "{}", self.video_name)
            self.video = self.load_video()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.image = self.ax.imshow(self.video[self.frame_idx, :, :, :])

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        plt.show()

    @property
    def current_video_name(self):
        return self.video_fname_fmt.format(self.video_idx) + self.video_ext

    @property
    def video_path(self):
        video_path = (
            os.path.join(
                self.directory, self.video_fname_fmt.format(self.video_idx)
            )
            + self.video_ext
        )
        return video_path

    def on_click(self, event):
        if event.button == 1:
            self.frame_idx = min(
                self.frame_idx + self.click_step_size, self.video.shape[0] - 1
            )
            self.update()
        if event.button == 2:
            pass  # TODO: get x and set as mirror point

    def on_key_press(self, event):
        if event.key == "q":
            self.video_idx -= 1
            self.video = self.load_video()
            self.update()

        elif event.key == "w":
            self.video_idx += 1
            self.video = self.load_video()
            self.update()

    def on_scroll(self, event):
        if event.button == "up":
            self.frame_idx = min(
                self.frame_idx + self.scroll_step_size, self.video.shape[0] - 1
            )
        elif event.button == "down":
            self.frame_idx = max(self.frame_idx - self.scroll_step_size, 0)
        self.update()

    def update(self):
        print("Figure updating with frame %s" % self.frame_idx)
        self.ax.imshow(self.video[self.frame_idx, :, :, :])
        self.fig.canvas.draw()

    def load_video(self):
        self.frame_idx = 0
        return skvideo.io.vread(self.video_path)  # , num_frames=400)


class TrackViewer(object):
    def __init__(self, directory, video_fname):
        self.frame_idx = 0
        self.video_fname = video_fname
        self.directory = directory
        self.video_idx = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.image = self.ax.imshow(self.video[self.frame_idx, :, :, :])

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.line_obj = None
        self.tracks = ()
        self.initialise_track_folders()
        self.load_tracks()
        self.add_track()
        self.scroll_step_size = 1
        self.x_positions = np.zeros(len(self.video))
        self.y_positions = np.zeros(len(self.video))

    def initialise_track_folders(self):
        self.tracks = np.arange(0, N_FRAMES, 1), np.arange(0, N_FRAMES, 1)
        if not os.path.isdir(self.directory):
            return "{} is not a valid path".format(self.directory)

        track_folder = os.path.join(self.directory, self.video_fname[:-5])

        if not os.path.isdir(track_folder):
            os.mkdir(track_folder)
            track_path = os.path.join(track_folder, "tracks.csv")
            self.create_tracks_df(track_path)

    @cached_property
    def video(self):
        vid_path = os.path.join(self.track_path + ".h264")
        return np.array(
            video_processing.get_frames(vid_path, self.track_errors)
        )

    @property
    def track_path(self):
        return os.path.join(self.directory, self.video_fname[:-5])

    @property
    def track_errors(self):
        # track_errors = tracks.get_failed_tracking_frames(self.track_path)
        # critical_track_errors = tracks.get_critical_frame_ids(track_errors)
        return np.arange(0, N_FRAMES, 1)

    def load_tracks(self):
        self.tracks = load_raw_track(self.track_path)

    def update_track(self):
        print("Track updating with frame %s" % self.frame_idx)
        self.line_obj.set_xdata(self.tracks[0][: self.frame_idx])
        self.line_obj.set_ydata(self.tracks[1][: self.frame_idx])

    def add_track(self):
        idx = min(self.frame_idx + 1, len(self.tracks) - 1)
        (self.line_obj,) = self.ax.plot(
            self.tracks[0][:idx], self.tracks[1][:idx]
        )

    def on_key_press(self, event):
        if event.key == "t":
            self.line_obj.set_visible(not self.line_obj.get_visible)
            if self.line_obj.get_visible():
                self.update_track()
        elif event.key == "u":
            self.update_tracks()
            self.save()

    def on_click(self, event):
        self.x_positions[self.frame_idx] = event.xdata
        self.y_positions[self.frame_idx] = event.ydata
        if not self.frame_idx == len(self.video) - 1:
            print(event.xdata, event.ydata)
            self.frame_idx += 1
            self.update()

    def on_scroll(self, event):
        self.x_positions[self.frame_idx] = event.xdata
        self.y_positions[self.frame_idx] = event.ydata
        if event.button == "up":
            self.frame_idx = min(
                self.frame_idx + self.scroll_step_size, self.video.shape[0] - 1
            )
        elif event.button == "down":
            self.frame_idx = max(self.frame_idx - self.scroll_step_size, 0)
        self.update()

        if self.line_obj.get_visible():
            self.update_track()

    def update(self):
        print("Figure updating with frame %s" % self.frame_idx)
        self.ax.cla()
        self.ax.imshow(self.video[self.frame_idx, :, :, :])
        self.fig.canvas.draw()

    def update_tracks(self):
        self.tracks[0][self.track_errors] = self.x_positions
        self.tracks[1][self.track_errors] = self.y_positions

    def save(self):
        path = os.path.join(
            self.track_path, "tracks.csv"
        )  # FIXME: when there is no tracks.csv file
        if not os.path.isfile(path):
            self.create_tracks_df(path)
        df = pd.read_csv(path, sep="\t")
        df["x_position"] = self.tracks[0]
        df["y_position"] = self.tracks[1]
        df.to_csv(path, sep="\t", index=False)
        return

    def create_tracks_df(self, path):
        d = {"x_position": self.tracks[0], "y_position": self.tracks[1]}
        df = pd.DataFrame(d, index=np.arange(N_FRAMES))
        df.to_csv(path, sep="\t", index=False)
