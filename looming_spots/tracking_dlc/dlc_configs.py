CONFIG_PATHS = {
    "bg_5_label": "/home/slenzi/winstor/margrie/slenzi/dlc/config_.yaml",
    "bg_5_label_SL_new": "/home/slenzi/code/python/dlc_hpc_tracking/heading_angle-StephenLenzi-2021-03-23/config.yaml",
    "6_label_cricket": "/home/slenzi/code/python/dlc_hpc_tracking/crickets-StephenLenzi-2021-04-07/config.yaml",
    "one_label": "/home/slenzi/winstor/margrie/slenzi/dlc/one_label-slenzi-2020-04-14/config.yaml",
    "simple_exploration_one_label": "/home/slenzi/winstor/margrie/slenzi/dlc/simple_exploration_one_label-slenzi-2020-04-04/config.yaml",
    "one_label_transform": "/home/slenzi/winstor/margrie/slenzi/dlc/one_label_transform-slenzi-2020-04-21/config_local.yaml",
    "one_label_transform_headbar": "/home/slenzi/winstor/margrie/slenzi/dlc/one_label_transform_headbar-slenzi-2020-05-14/config.yaml",
}

DLC_FNAMES = {
    "bg_5_label": "cam_transform*DLC_resnet50_fibertrialscroppedJun11shuffle1_1030000.h5",
    "bg_5_label_SL_new": "cameraDLC_resnet50_heading_angleMar23shuffle1_1030000*.h5",
    "6_label_cricket": "cameraDLC_resnet50_cricketsApr7shuffle1_1030000*.h5",
    "simple_exploration_one_label": "cameraDLC_resnet50_simple_exploration_one_labelApr4shuffle1_1030000.h5",
    "one_label": "cam_transformDLC_resnet50_one_labelApr14shuffle1_1030000.h5",
    "one_label_transform": "cam_transform*DLC_resnet50_one_label_transformApr21shuffle1_1030000filtered.h5",
    "one_label_transform_headbar": "cam_transformDLC_resnet50_one_label_transform_headbarMay14shuffle1_1030000.h5",
}

BODYPART_LABELS = {
    "one_label_transform": ["body"],
    "one_label_transform_headbar": ["bodypart1"],
    "bg_5_label": [
        "nose",
        "L_ear",
        "R_ear",
        "shoulder_blades",
        "tail_base",
        "tail_tip",
    ],
    "bg_5_label_SL_new": [
        "nose",
        "L_ear",
        "R_ear",
        "body",
        "tail_base",
        "tail_tip",
    ],
}


class MultipleH5FileError(Exception):
    pass


class MouseNotTrackedError(Exception):
    pass
