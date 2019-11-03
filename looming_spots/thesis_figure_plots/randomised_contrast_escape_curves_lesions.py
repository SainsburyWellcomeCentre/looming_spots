import pandas as pd
import pingouin as pg

from looming_spots.analysis.escape_metric_dataframes import (
    get_behaviour_metric_dataframe,
)
from looming_spots.db import experimental_log, loom_trial_group

# "CA451A_4",ohda
GROUPS = {
    "OHDA": ["CA451A_1", "CA451A_2", "CA451A_3", "CA478_2", "CA476_4"],
    "NMDA": ["276585A", "276585B", "CA439_1", "CA439_4"],
    "CONTROL": [
        "276585D",
        "276585E",
        "CA452_1",
        "CA439_5",
        "CA451A_5",
        "CA459A_2",
        "CA478_3",
    ],
    "d1_caspase": experimental_log.get_mouse_ids_in_experiment(
        "d1MSN_caspase_lesion_TS"
    ),
    "d2_caspase": experimental_log.get_mouse_ids_in_experiment(
        "d2MSN_caspase_lesion_TS"
    ),
}


def get_df(metric, groups=GROUPS):
    all_df = pd.DataFrame()
    for label, mids in groups.items():
        mtgs = [loom_trial_group.MouseLoomTrialGroup(mid) for mid in mids]
        df = get_behaviour_metric_dataframe(mtgs, metric, "variable_contrast")
        df["experimental group"] = [label] * len(df)
        all_df = all_df.append(df)
    return all_df


def get_escape_curve_df(df):
    escape_curve_df = pd.DataFrame()
    for group in df["experimental group"].unique():
        sub_df = df[df["experimental group"] == group]

        for mid in sub_df["mouse id"].unique():
            escape_curve = (
                df[df["mouse id"] == mid][["contrast", "escape"]]
                .groupby("contrast")
                .mean()
                .reset_index()
            )
            escape_curve["mouse id"] = [mid] * len(escape_curve)
            escape_curve["experimental group"] = [group] * len(escape_curve)
            escape_curve_df = escape_curve_df.append(escape_curve)
    return escape_curve_df


def two_way_mixed_anova(group_label_1, group_label_2, post_hoc=False):
    """
    perform two way mixed ANOVA on two groups of mice with variable contrast curve experiment

    :param str group_label_1:
    :param str group_label_2:
    :return:
    """
    groups = {
        group_label_1: GROUPS[group_label_1],
        group_label_2: GROUPS[group_label_2],
    }
    print(groups)

    df = get_df("classified as flee", groups)
    escape_curve_df = get_escape_curve_df(df)

    aov = pg.mixed_anova(
        dv="escape",
        within="contrast",
        between="experimental group",
        subject="mouse id",
        data=escape_curve_df,
    )
    pg.print_table(aov)

    if post_hoc:
        pg.pairwise_ttests(
            dv="escape",
            within="contrast",
            between="experimental group",
            subject="mouse id",
            data=escape_curve_df,
        )
    return escape_curve_df, aov
