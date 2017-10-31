import os
import matplotlib.pyplot as plt
from configobj import ConfigObj
import skvideo.io
import scipy.misc
import numpy as np
import re
from looming_spots.analysis import extract_looms


class Viewer(object):
    """
    This class allows browsing short videos to build reference frames manually. It allows the manual selection of left
    and right frames. 'left key': sets the index for the frame containing an empty left hand side of the arena, whereas
    'right key': sets the index of the right. 'Enter': triggers the writing of these indices and video paths to a
    text file called Metadata.txt, and also saves the composite frame. If there are a series of videos that change
    by increment only then 'w' and 'q' enable toggling between videos.
    """
    def __init__(self, directory, video=None, video_fname=None):
        self.frame_idx = 0
        self.directory = directory

        if video is not None:
            self.video = video
        elif video_fname:
            self.video_ext = '.' + video_fname.split('.')[-1]
            self.video_name = video_fname.split('.')[0]
            self.video_idx = int(re.search(r'\d+', self.video_name).group())
            self.video_fname_fmt = re.sub(r'\d+', '{}', self.video_name)
            self.video = self.load_video()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.image = self.ax.imshow(self.video[self.frame_idx, :, :, :])

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.left_ref = None
        self.right_ref = None

        plt.show()

    @property
    def current_video_name(self):
        return self.video_fname_fmt.format(self.video_idx) + self.video_ext

    @property
    def video_path(self):
        video_path = os.path.join(self.directory, self.video_fname_fmt.format(self.video_idx)) + '.h264'
        return video_path

    def on_click(self, event, step_size=50):
        if event.button == 1:
            self.frame_idx = min(self.frame_idx + step_size, self.video.shape[0] - 1)
            self.update()

    def on_key_press(self, event):
        if event.key == 'left':
            self.left_ref = HalfRef('left', self.current_video_name, self.frame_idx, self.video[self.frame_idx])
            self.left_ref.save_metadata()
            print('left ref idx: {}'.format(self.frame_idx))

        elif event.key == 'right':
            self.right_ref = HalfRef('right', self.current_video_name, self.frame_idx, self.video[self.frame_idx])
            self.right_ref.save_metadata()
            print('left ref idx: {}'.format(self.frame_idx))

        elif event.key == 'enter':
            self.save_reference_frame_indices()

        elif event.key == 'q':
            self.video_idx -= 1
            self.video = self.load_video()
            self.update()

        elif event.key == 'w':
            self.video_idx += 1
            self.video = self.load_video()
            self.update()

    def on_scroll(self, event, step_size=20):
        if event.button == 'up':
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
        return skvideo.io.vread(self.video_path, num_frames=400)

    def save_reference_frame_indices(self):
        self.left_ref.save_metadata()
        self.right_ref.save_metadata()
        self.save_reference_frame()

    @staticmethod
    def make_reference_frame(left_frame, right_frame, x_pos=400):
        composite_frame = np.zeros_like(left_frame)
        composite_frame[:, :x_pos, :] = left_frame[:, :x_pos, :]
        composite_frame[:, x_pos:, :] = right_frame[:, x_pos:, :]
        return composite_frame

    def save_reference_frame(self):
        reference_frame = self.make_reference_frame(self.left_ref.frame, self.right_ref.frame)
        plt.imshow(reference_frame); plt.show()
        ref_array = np.mean(reference_frame, axis=2)
        save_fpath = os.path.join(self.directory, 'ref.png')
        print('saving reference frame to: {}'.format(save_fpath))
        scipy.misc.imsave(save_fpath, ref_array, format='png')


class HalfRef(object):
    def __init__(self, side=None, video_name=None, frame_idx=None, frame=None):
        self.metadata = ConfigObj('./manual_metadata.txt')
        self.side = side
        self.video_name = video_name
        self.frame_idx = frame_idx
        self.frame = frame

    def load_from_metadata(self):
        self.video_name = self.metadata['reference_frame'][self.side]['video_name']
        self.frame_idx = self.metadata['reference_frame'][self.side]['frame_idx']

    def save_metadata(self):
        if 'reference_frame' not in self.metadata:
            self.metadata['reference_frame'] = {}
        if self.side not in self.metadata:
            self.metadata['reference_frame'][self.side] = {}

        self.metadata['reference_frame'][self.side]['video_name'] = self.video_name
        self.metadata['reference_frame'][self.side]['frame_idx'] = self.frame_idx
        self.metadata.write()

    @property
    def image(self):
        return extract_looms.get_frame(self.video_name, self.frame_idx)


class Ref(object):
    def __init__(self):
        self.metadata = ConfigObj('./manual_metadata.txt')
        self.left = None
        self.right = None
        self.load_from_metadata()
        self.load_reference_frame()

    def load_from_metadata(self):
        for item in self.metadata['reference_frame']:
            side = self.metadata['reference_frame'][item]
            video_name = self.metadata['reference_frame'][item]['video_name']
            frame_idx = self.metadata['reference_frame'][item]['frame_idx']
            half_ref = HalfRef(side, video_name, frame_idx)
            if item == 'left':
                self.left = half_ref
            elif item == 'right':
                self.right = half_ref

    def load_reference_frame(self):
        img = extract_looms.make_reference_frame(self.left.image, self.right.image)
        plt.imshow(img); plt.show()
