import skimage
import skimage.transform
import numpy as np


def get_inverse_projective_transform(
    src=np.array([[0, 240], [0, 0], [600, 240], [600, 0]]),
    dest=np.array(
        [
            [27.08333156, 296.33332465],
            [77.49999672, 126.74999637],
            [628.41664697, 308.24999096],
            [607.33331426, 130.41666292],
        ]
    ),
    output_shape=(240, 600),
):
    """
    coordinates np.array([x1, y1], [x2, y2], [x3, y3], [x4, y4])

    x2y2-------------------x4y4
     |                       |
     |                       |
     |                       |
    x1y1-------------------x3y3

    :param output_shape:
    :param img:
    :param src: coordinates of the four corners of the 'source' i.e. the desired standard space
    :param dest: coordinates of the four corners of the actual arena
    :return: transformed image
    """
    p = skimage.transform.ProjectiveTransform()
    p.estimate(src, dest)
    return p


def get_box_coordinates_from_file(box_path):
    napari_fmt_coords = np.load(str(box_path))
    napari_fmt_coords = np.roll(napari_fmt_coords, 1, axis=1)
    new_box_coords = np.empty_like(napari_fmt_coords)
    new_box_coords[0] = napari_fmt_coords[1]
    new_box_coords[1] = napari_fmt_coords[0]
    new_box_coords[2] = napari_fmt_coords[2]
    new_box_coords[3] = napari_fmt_coords[3]
    return new_box_coords
