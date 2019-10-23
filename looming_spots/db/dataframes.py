import matplotlib.pyplot as plt
from looming_spots.db import experimental_log, loom_trial_group
from datetime import timedelta
import pandas as pd


def get_loom_results_from_df(df):
    n_flees = df.groupby(["loom_idx"]).sum()["classified as flee"]
    n_trials = df.groupby(["loom_idx"]).count()["classified as flee"]

    return n_flees, n_trials, n_flees / n_trials


mids_to_process = []
mids_of_flees = []
main_df = pd.DataFrame()
for context in ["A", "A2", "A9"]:
    for mid in experimental_log.get_combination(
        include=["post_test", "habituation"],
        exclude=["pre_test"],
        matching_dict_post_test={
            "test_type": "== post_test",
            "context": f"== {context}",
            "surgery": "== FALSE",
            "drug_given/additional_param": "== FALSE",
            "stimulus": "== looming",
            "line": "== wt",
        },
        matching_dict_habituation={
            "test_type": "== habituation",
            "context": f"== {context}r",
            "surgery": "== FALSE",
            "gradient_protocol": "== 2",
            "drug_given/additional_param": "== FALSE",
            "stimulus": "== looming",
            "line": "== wt",
        },
    ):
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        try:
            mouse_df = mtg.to_trials_df("post_test")
            mouse_df["time_since_habituation"] = mtg.get_trials_of_type(
                "post_test"
            )[0].days_since_last_session_type()
            mouse_df["days_since_habituation"] = [
                mtg.get_trials_of_type("post_test")[0]
                .days_since_last_session_type()
                .days
            ] * 3

            main_df = main_df.append(mouse_df, ignore_index=True)
        except Exception as e:
            mids_to_process.append(mid)
            continue

    for day in [0, 1, 3, 7, 14, 28]:
        df = main_df[main_df["days_since_habituation"] == day]
        subset = df[["loom_idx", "classified as flee"]]
        n_flees, n_total, results = get_loom_results_from_df(subset)
        for i, (nf, nt, result) in enumerate(zip(n_flees, n_total, results)):
            print(
                f"{day} days post habituation, {nf} out of {nt} flees, percentage: {result * 100} %"
            )

            plt.bar(day + 0.33 * i, result, 0.2)

log = experimental_log.load_df()
mouse_ids = experimental_log.get_mouse_ids_with_test_combination(
    log, ["pre_test"], []
)
df = experimental_log.get_subset_df_from_mouse_ids(log, mouse_ids)
for context in ["A", "A2", "A9"]:
    ids = experimental_log.get_mouse_ids_from_query(
        df,
        filter_dict={
            "test_type": "== pre_test",
            "context": "== A9",
            "surgery": "== FALSE",
            "drug_given/additional_param": "== FALSE",
            "stimulus": "== looming",
            "line": "== wt",
        },
    )
    for mid in ids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        try:
            mouse_df = mtg.to_trials_df("pre_test")
            mouse_df["time_since_habituation"] = -1
            mouse_df["days_since_habituation"] = [-1] * 3

            main_df_pre = main_df.append(mouse_df)
        except Exception as e:
            mids_to_process.append(mid)
            continue


for day in [0, 1, 3, 7, 14, 28]:
    df = main_df[main_df["days_since_habituation"] == day]
    subset = df[["loom_idx", "classified as flee"]]
    _, _, results = get_loom_results_from_df(subset)
    for i, result in enumerate(results):
        plt.bar(day + 0.33 * i, result, 0.2)
