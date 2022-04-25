from looming_spots.db import loom_trial_group
import pathlib
import pandas as pd


HEAD_DIRECTORY = pathlib.Path("Z:\\margrie\\glusterfs\\imaging\\l\\loomer")
PROCESSED_DATA_DIRECTORY = HEAD_DIRECTORY / "processed_data"

DATA_DIRECTORY = pathlib.Path("D:\\SWC\\lenzietal2022_lse\\test_rerun_raw3\\")
EXAMPLE_TRACKS_DIR = DATA_DIRECTORY / "example_tracks"
FIGURE_DIRECTORY = DATA_DIRECTORY / "figures"
DATAFRAME_PATH = DATA_DIRECTORY / "dataframes"
DLC_PADDING = 40

experimental_groups = [
    "ih_ivc_7day",
    "lse",
    "gh_enriched",
    "ih_enriched",
    "ih_ivc_1mth",
    "auditory_naive",
    "auditory_lse",
    "pre_test_24hr_pre",
    "pre_test_immediate_pre",
    "pre_test_none_post",
    "pre_test_24hr_post",
    "pre_test_immediate_post",
    "lse_with_shelter",
]

mouse_ids = {k: [] for k in experimental_groups}

# MOUSE IDS
mouse_ids["ih_ivc_7day"] = [
    "CA159_1",
    "CA159_4",
    "CA160_1",
    "CA301_1",
    "CA301_2",
    "CA301_3",
    "CA160_2",
    "CA160_5",
    "CA112_5",
    "CA113_1",
    "CA131_1",
    "CA131_2",
    "CA131_4",
    "CA131_5",
    "CA132_1",
    "CA132_2",
    "CA132_3",
    "CA132_4",
    "CA473_1",
    "CA475_2",
    "CA482_1",
    "CA482_2",
    "CA482_3",
    "CA482_4",
    "CA475_4",
    "CA476_1",
    "CA188_4",
    "CA113_2",
    "CA113_3",
    "CA113_4",
    "CA473_2",
    "CA473_3",
    "CA473_5",
    "CA475_5",
    "1114085",
    "1114086",
    "1114087",
    "1114088",
    "1114089",
    "1114290",
    "1114291",
    "1114292",
    "1114293",
    "1114294",
]

mouse_ids["lse"] = [
    "CA73_4",
    "CA124_2",
    "CA127_1",
    "CA127_2",
    "CA127_3",
    "CA129_2",
    "CA130_2",
    "CA136_4",
    "CA88_1",
    "CA88_2",
    "CA95_2",
    "CA95_5",
    "CA135_3",
    "CA84_3",
    "CA85_1",
    "CA86_2",
    "CA86_3",
    "CA137_3",
    "CA83_3",
    "CA83_5",
    "CA90_2",
    "CA90_4",
    "CA81_2",
    "CA81_3",
    "CA83_1",
    "CA284_2",
    "CA284_3",
    "CA284_4",
    "CA284_5",
    "CA285_1",
    "CA285_2",
    "CA325_1",
    "CA319_1",
    "CA319_2",
    "CA319_3",
    "CA319_5",
    "CA320_4",
    "CA320_5",
    "CA54_3",
    "CA54_4",
    "CA54_5",
    "CA55_4",
    "CA72_5",
    "CA74_2",
    "CA71_1",
    "CA72_1",
    "CA69_3",
    "CA70_1",
    "CA53_4",
    "CA53_5",
    "CA55_1",
    "CA55_3",
    "CA50_5",
    "1114186",
    "1114188",
    "1114189",
    "1114307",
    "1114308",
    "CA77_1",
]

# social_housed_in_20
mouse_ids["gh_enriched"] = [
    "CA408_4",
    "CA408_5",
    "CA412_3",
    "CA413_3",
    "CA414_5",
    "social_1",
    "social_11",
    "social_12",
    "social_13",
    "social_15",
    "social_17",
    "social_20",
    "social_4",
    "social_6",
    "social_8",
]

mouse_ids["ih_ivc_1mth"] = [
    "CA409_2",
    "CA412_1",
    "CA413_1",
    "CA415_2",
    "social_10",
    "social_14",
    "social_16",
    "social_18",
    "social_19",
    "social_2",
    "social_3",
    "social_5",
    "social_7",
    "social_9",
    "CA230_2",
    "CA254_1",
    "CA280_1",
    "CA310_1",
    "CA330_2",
    "CA368_1",
]


mouse_ids["ih_enriched"] = [
    "CA230_5",
    "CA254_4",
    "CA280_2",
    "CA310_2",
    "CA330_3",
    "CA368_2",
]


mouse_ids["auditory_lse"] = ["CA429_2", "CA429_3", "CA429_4", "CA429_5"]


mouse_ids["auditory_naive"] = [
    "CA430_3",
    "CA430_4",
    "CA430_5",
    "CA431_1",
    "CA431_2",
]

#
mouse_ids["pre_test_none_post"] = [
    "CA284_5",
    "CA284_4",
    "CA285_1",
    "CA325_1",
    "CA319_3",
    "CA319_1",
    "CA319_5",
    "CA320_5",
    "CA320_4",
    "CA319_2",
]


mouse_ids["pre_test_24hr_pre"] = [
    "CA473_3",
    "CA473_5",
    "CA475_2",
    "CA482_1",
    "CA482_3",
    "CA482_4",
]


mouse_ids["pre_test_immediate_pre"] = [
    "CA473_1",
    "CA473_2",
    "CA475_4",
    "CA475_5",
    "CA476_1",
    "CA482_2",
]

mouse_ids["lse_with_shelter"] = [
    "1116287",
    "1116285",
    "1116289",
    "1116286",
]

mouse_ids["pre_test_24hr_post"] = mouse_ids["pre_test_24hr_pre"]

mouse_ids["pre_test_immediate_post"] = mouse_ids["pre_test_immediate_pre"]


longitudinal_lse_ids = {
    0: [
        "CA73_4",
        "CA124_2",
        "CA127_1",
        "CA127_2",
        "CA127_3",
        "CA129_2",
        "CA130_2",
        "CA284_2",
        "CA284_3",
        "CA284_4",
        "CA284_5",
        "CA285_1",
        "CA285_2",
        "CA325_1",
        "CA53_4",
        "CA53_5",
        "CA55_1",
        "CA55_3",
    ],
    1: [
        "CA136_4",
        "CA88_1",
        "CA88_2",
        "CA95_2",
        "CA95_5",
        "CA54_3",
        "CA54_4",
        "CA54_5",
        "CA55_4",
    ],
    3: [
        "CA135_3",
        "CA84_3",
        "CA85_1",
        "CA86_2",
        "CA86_3",
        "CA319_1",
        "CA319_2",
        "CA319_3",
        "CA319_5",
        "CA320_4",
        "CA320_5",
        "CA72_5",
        "CA74_2",
    ],
    7: ["CA137_3", "CA83_3", "CA83_5", "CA90_2", "CA90_4", "CA71_1", "CA72_1"],
    8: ["CA69_3", "CA70_1"],
    14: ["CA77_1", "CA81_2", "CA81_3", "CA83_1"],
}


test_types = {
    "ih_ivc_7day": "pre_test",
    "lse": "post_test",
    "gh_enriched": "pre_test",
    "ih_enriched": "pre_test",
    "ih_ivc_1mth": "pre_test",
    "auditory_naive": "pre_test",
    "auditory_lse": "pre_test",
    "pre_test_24hr_pre": "pre_test",
    "pre_test_immediate_pre": "pre_test",
    "pre_test_none_post": "post_test",
    "pre_test_24hr_post": "post_test",
    "pre_test_immediate_post": "post_test",
    "lse_with_shelter": "post_test",
}


def load_dataframe(k):
    csv_path = DATAFRAME_PATH / f"{k}.csv"
    df = pd.read_csv(csv_path)
    return df


dataframe_paths = {k: DATAFRAME_PATH / f"{k}.csv" for k in experimental_groups}
dataframe_dict = {k: pd.DataFrame() for k in experimental_groups}


def load_experimental_condition_dataframe_raw(
    group_id, mouse_ids, test_types, dataframe_dict, dataframe_paths
):
    for mouse_id in mouse_ids[group_id]:

        trial_group = loom_trial_group.MouseLoomTrialGroup(mouse_id)

        if test_types[group_id] == "pre_test":
            if group_id == "auditory_naive" or group_id == "auditory_lse":
                these_trials = trial_group.pre_test_trials()[1:4]
            else:
                these_trials = trial_group.pre_test_trials()[0:3]
        else:
            these_trials = trial_group.post_test_trials()[0:3]

        for trial in these_trials:
            dataframe_dict[group_id] = dataframe_dict[group_id].append(
                trial.to_df(group_id), ignore_index=True
            )
    if group_id == "gh_enriched":
        idx = dataframe_dict["gh_enriched"]["mouse_id"] == "social_11"
        dataframe_dict["gh_enriched"]["is_freeze"][idx] = False

    savepath = dataframe_paths[group_id]

    if not savepath.exists():
        dataframe_dict[group_id].to_csv(savepath)

    return dataframe_dict[group_id]


def main():
    for group_id in test_types.keys():
        savepath = dataframe_paths[group_id]

        if not savepath.exists():
            load_experimental_condition_dataframe_raw(
                group_id, mouse_ids, test_types, dataframe_dict, dataframe_paths
            )


if __name__ == "__main__":
    main()
