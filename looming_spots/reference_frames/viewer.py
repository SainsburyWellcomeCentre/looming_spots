import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import skvideo.io

import looming_spots.create_reference_frame.reference_frames
import looming_spots.metadata.experiment_metadata
import looming_spots.util.video_processing

DEFAULT_VIDEO_PATH = './camera.mp4'
VIEWER_N_FRAMES_LIMIT = 18000


def get_digit_from_string(string):
    return int(re.search(r'\d+', string).group())


def digit_present(string):
    return any([x.isdigit() for x in string])


class Viewer(object):
    """
    This class allows browsing short videos to build reference frames manually. It allows the manual selection of left
    and right frames. 'left key': sets the index for the frame containing an empty left hand side of the arena, whereas
    'right key': sets the index of the right. 'Enter': triggers the writing of these indices and video paths to a
    text file called Metadata.txt, and also saves the composite frame. If there are a series of videos that change
    by increment only then 'w' and 'q' enable toggling between videos.
    """
    def __init__(self, directory, video=None, video_fname='loom0.h264'):
        self.frame_idx = 0
        self.directory = directory
        self.video_idx = None
        self.mirror_idx = 300

        if video is not None:
            self.video = video
        elif video_fname:
            self.video_ext = '.' + video_fname.split('.')[-1]
            self.video_name = video_fname.split('.')[0]

            if digit_present(self.video_name):
                self.video_idx = get_digit_from_string(self.video_name)

            self.video_fname_fmt = re.sub(r'\d+', '{}', self.video_name)
            self.video = self.load_video()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.image = self.ax.imshow(self.video[self.frame_idx, :, :, :])

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.ref = Ref(directory)
        self.left_ref = None
        self.right_ref = None
        plt.show()

    @property
    def current_video_name(self):
        return self.video_fname_fmt.format(self.video_idx) + self.video_ext

    @property
    def video_path(self):
        video_path = os.path.join(self.directory, self.video_fname_fmt.format(self.video_idx)) + self.video_ext
        return video_path

    def on_click(self, event, step_size=4):
        if event.button == 1:
            self.frame_idx = min(self.frame_idx + step_size, self.video.shape[0] - 1)
            self.update()
        if event.button == 3:
            self.mirror_idx = int(event.xdata)
            print(self.mirror_idx)

    def on_key_press(self, event):
        if event.key == 'left':
            self.left_ref = HalfRef(self.ref, 'left', self.current_video_name,
                                    self.frame_idx, self.video[self.frame_idx])
            print('left ref idx: {}'.format(self.frame_idx))

        elif event.key == 'right':
            self.right_ref = HalfRef(self.ref, 'right', self.current_video_name,
                                     self.frame_idx, self.video[self.frame_idx])
            print('right ref idx: {}'.format(self.frame_idx))

        elif event.key == 'enter':
            self.save_reference_frame()

        elif event.key == 'q':
            self.video_idx -= 1
            self.video = self.load_video()
            self.update()

        elif event.key == 'w':
            self.video_idx += 1
            self.video = self.load_video()
            self.update()

        elif event.key == 'k':
            self.ref.metadata['skip'] = True
            self.ref.write_metadata()

        elif event.key == 't':
            self.ref.metadata['track_start'] = self.frame_idx
            self.ref.write_metadata()
        elif event.key == 'z':
            self.ref.metadata['grid_location'] = 'left'
            self.ref.write_metadata()
        elif event.key == 'x':
            self.ref.metadata['grid_location'] = 'right'
            self.ref.write_metadata()

    def on_scroll(self, event, step_size=20):
        if event.button == 'up':
            print(self.frame_idx, self.video.shape[0])
            self.frame_idx = min(self.frame_idx + step_size, self.video.shape[0] - 1)
        elif event.button == 'down':
            self.frame_idx = max(self.frame_idx - step_size, 0)
        self.update()

    def update(self):
        self.ax.imshow(self.video[self.frame_idx, :, :, :])
        self.fig.canvas.draw()

    def load_video(self):
        self.frame_idx = 0
        print(self.video_path)
        if self.video_path.endswith('mp4'):
            return skvideo.io.vread(self.video_path, num_frames=VIEWER_N_FRAMES_LIMIT)
        return skvideo.io.vread(self.video_path)

    def save_reference_frame_indices(self):
        self.save_reference_frame()

    @staticmethod
    def make_reference_frame(left_frame, right_frame, mirror_plane_idx):
        composite_frame = np.zeros_like(left_frame)
        composite_frame[:, :mirror_plane_idx, :] = left_frame[:, :mirror_plane_idx, :]
        composite_frame[:, mirror_plane_idx:, :] = right_frame[:, mirror_plane_idx:, :]
        return composite_frame

    def save_reference_frame(self):
        reference_frame = self.make_reference_frame(self.left_ref.frame, self.right_ref.frame,
                                                    mirror_plane_idx=self.mirror_idx)
        plt.imshow(reference_frame); plt.show()
        ref_array = np.mean(reference_frame, axis=2)
        save_fpath = os.path.join(self.directory, 'ref.png')
        print('saving reference frame to: {}'.format(save_fpath))
        scipy.misc.imsave(save_fpath, ref_array, format='png')
        np.save(save_fpath[:-4] + '.npy', ref_array)


class Ref(object):
    def __init__(self, directory, path=None):
        if not path:
            self.path = DEFAULT_VIDEO_PATH
        self.metadata = looming_spots.metadata.experiment_metadata.load_metadata(directory)
        self.initialise_metadata()
        self.left = None
        self.right = None
        self.load_from_metadata()

    def initialise_metadata(self):
        if 'reference_frame' not in self.metadata:
            self.metadata['reference_frame'] = {}
            self.metadata['video_name'] = self.path

    def load_from_metadata(self):
        if 'reference_frame' not in self.metadata:
            return 'cannot load reference frame, no metadata attributes found'
        print('loading from metadata')
        video_name = self.metadata['video_name']
        for item in self.metadata['reference_frame']:
            if item == 'left':  # TODO: generic concatenation row and column-wise from matrix as list of image pieces
                frame_idx = self.metadata['reference_frame'][item]
                half_ref = HalfRef(self, item, video_name, int(frame_idx))
                self.left = half_ref
            elif item == 'right':
                frame_idx = self.metadata['reference_frame'][item]
                half_ref = HalfRef(self, item, video_name, int(frame_idx))
                self.right = half_ref

    def load_reference_frame(self):
        print('loading reference frame')
        if self.left is None or self.right is None:
            return
        img = looming_spots.create_reference_frame.reference_frames.make_reference_frame(self.left.image,
                                                                                         self.right.image)
        plt.imshow(img); plt.show()

    def write_metadata(self):
        self.metadata.write()


class HalfRef(object):
    def __init__(self, ref, side=None, video_name=None, frame_idx=None, frame=None):
        self.ref = ref
        self.side = side
        self.video_name = video_name
        self.frame_idx = frame_idx
        self.frame = frame
        self.metadata = self.ref.metadata['reference_frame']

    def absolute_frame_idx(self):
        vid_idx = get_digit_from_string(self.video_name)
        loom_frame_idx = self.ref.metadata['manual_loom_idx'][vid_idx]
        abs_idx = int(self.frame_idx) + int(loom_frame_idx) - 200  # TODO: test to ensure the frames are identical
        return abs_idx

    def initialise_metadata(self):
        if self.side not in self.ref.metadata['reference_frame']:  # TODO: remove redundancy
            self.ref.metadata['reference_frame'][self.side] = {}

    def load_from_metadata(self):
        self.video_name = self.metadata['video_name']
        self.frame_idx = self.metadata['frame_idx']

    @property
    def image(self):
        return looming_spots.util.video_processing.get_frame(self.video_name, self.frame_idx)
