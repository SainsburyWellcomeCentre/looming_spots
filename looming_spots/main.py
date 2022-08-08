from looming_spots.db import loom_trial_group
import pathlib
import pandas as pd


HEAD_DIRECTORY = pathlib.Path("Z:\\margrie\\glusterfs\\imaging\\l\\loomer")
PROCESSED_DATA_DIRECTORY = HEAD_DIRECTORY / "processed_data"

OUTPUT_DIRECTORY = pathlib.Path("D:\\SWC\\lenzietal2022_lse\\kcl_test\\")
EXAMPLE_TRACKS_DIR = OUTPUT_DIRECTORY / "example_tracks"
FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"
DATAFRAME_PATH = OUTPUT_DIRECTORY / "dataframes"
DLC_PADDING = 40

experimental_groups = [
    "ih_ivc_7day",
    "lse",
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
]

mouse_ids["lse"] = [
    "CA73_4",
    "CA124_2",
    "CA127_1",
    "CA127_2",
    "CA127_3",
    "CA129_2",
]


test_types = {
    "ih_ivc_7day": "pre_test",
    "lse": "post_test",
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
            these_trials = trial_group.pre_test_trials()[0:3]
        else:
            these_trials = trial_group.post_test_trials()[0:3]

        for trial in these_trials:
            dataframe_dict[group_id] = dataframe_dict[group_id].append(
                trial.to_df(group_id), ignore_index=True
            )

    savepath = dataframe_paths[group_id]

    if not savepath.exists():
        if not savepath.parent.exists():
            savepath.parent.mkdir(exist_ok=True)
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
