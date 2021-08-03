import itertools
import matplotlib.pyplot as plt

import looming_spots.io.session_io


def get_all_context_combos(df):
    results_dict = {}
    all_combos = list(set(df["context"].values))
    all_combos = [context.strip("r") for context in all_combos]
    all_combos = list(itertools.combinations(all_combos, 2))

    for lsie_context, post_test_context in all_combos:

        lsie_context += "r"

        exclude_pre_tests_df = df[df["test_type"] != "pre_test"]
        lsie_df = exclude_pre_tests_df[
            exclude_pre_tests_df["test_type"] == "lsie"
        ]
        post_test_df = exclude_pre_tests_df[
            exclude_pre_tests_df["test_type"] == "post_test"
        ]

        lsie_in_context_df = lsie_df[
            lsie_df.isin([lsie_context])["context"]
        ]
        post_tests_in_context_df = post_test_df[
            post_test_df.isin([post_test_context])["context"]
        ]

        mids_in_group = set(
            lsie_in_context_df["mouse_id"]
        ).intersection(post_tests_in_context_df["mouse_id"])

        results_dict.setdefault(
            "".join([lsie_context, post_test_context]), mids_in_group
        )

    return results_dict


def plot_from_mid_dict(condition_mouse_id_dictionary):
    fig, axes = plt.subplots(len(condition_mouse_id_dictionary.keys()), 2)

    for j, (condition, mids) in enumerate(
        condition_mouse_id_dictionary.items()
    ):
        for mid in mids:
            sessions = looming_spots.io.session_io.load_sessions(mid)
            for i, s in enumerate(sorted(sessions)):
                plt.sca(axes[j][i])
                title = f"{condition}_{s.contains_lsie}"
                axes[j][i].set_title(title)
                s.plot_trials()
