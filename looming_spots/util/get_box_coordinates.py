
import skvideo.io
import numpy as np
import napari
import pathlib
import os
from looming_spots.io import load
from looming_spots.tracking_dlc.track_mouse import track_mouse


def get_all_video_paths(root_dir='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/'):
    return list(pathlib.Path(root_dir).rglob('*avi'))


def get_all_images(all_video_paths,
                   out_dir='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/all_imgs_dlc_crop_mids_tmp/'):
    """
    saves an image for every video in the path list provided
    :param all_video_paths:
    :param out_dir:
    :return:
    """
    for vpath in list(all_video_paths):
        fpath = f'{out_dir}{vpath.parent.parent.stem}__{vpath.parent.stem}.npy'
        if not os.path.isfile(fpath):
            try:
                img_idx = 500
                videogen = skvideo.io.vreader(str(vpath))
                for frame, idx in zip(videogen, range(img_idx)):
                    f = frame
                print(f'saving to .. {fpath}')
                np.save(fpath, np.mean(f, axis=2))
            except Exception as e:
                print(e)
                continue


def get_all_images_mouse_ids(mouse_ids,root='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/raw_data/',
                   out_dir='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/all_imgs_dlc_crop_mids_tmp/', video_fname='*camera.avi', frame_idx=500):
    """
    saves an image for every video in the path list provided
    :param all_video_paths:
    :param out_dir:
    :return:
    """

    for mid in mouse_ids:
        directory = pathlib.Path(f'{root}{mid}')
        video_paths = directory.rglob(video_fname)

        for vpath in list(video_paths):
            fpath = f'{out_dir}{vpath.parent.parent.stem}__{vpath.parent.stem}.npy'
            if not os.path.isfile(fpath):
                try:
                    img_idx = frame_idx
                    videogen = skvideo.io.vreader(str(vpath))
                    for frame, idx in zip(videogen, range(img_idx)):
                        f = frame
                    print(f'saving to .. {fpath}')
                    np.save(fpath, np.mean(f, axis=2))
                except Exception as e:
                    print(e)
                    print(vpath)
                    continue


def get_save_path(save_path, out_dir):
    """
    gets save path for box coordinates based on file name
    :param save_path:
    :param out_dir:
    :return:
    """
    save_name = save_path.stem
    out_path = out_dir / save_name.split('__')[0] / save_name.split('__')[1].replace('.npy',
                                                                                     '') / 'box_corner_coordinates.npy'
    return out_path


def curate_all(root_dir='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/all_imgs_dlc_crop_mids_tmp/',
               out_dir='/home/slenzi/winstor/margrie/glusterfs/imaging/l/loomer/processed_data/'):

    """
    tool for getting the arena outline

    :param root_dir:
    :param out_dir:
    :return:
    """
    root_dir = pathlib.Path(root_dir)
    out_dir = pathlib.Path(out_dir)
    data_paths = list(root_dir.glob('*.npy'))

    data_paths = [p for p in data_paths if not os.path.isfile(str(get_save_path(p, out_dir)))]
    image = np.array([np.load(p) for p in data_paths])

    with napari.gui_qt():
        v = napari.Viewer(title="curate crop region")
        v.add_image(image)
        shapes_layer = v.add_shapes()


        @v.bind_key("b")
        def conv_to_b(v):
            for data in v.layers[-1].data:
                try:
                    frame_idx = int(data[0, 0])
                    box_vertices = data[:, 1:]
                    p = data_paths[frame_idx]
                    out_path = get_save_path(p, out_dir)
                    print(out_path)
                    np.save(str(out_path), box_vertices)
                except Exception as e:
                    print(e)
                    continue


def main(mouse_ids):
    load.sync_raw_and_processed_data()
    get_all_images_mouse_ids(mouse_ids)
    curate_all()
    for mid in mouse_ids:
        track_mouse(mid)


if __name__ == '__main__':
    main(['1115137',])
