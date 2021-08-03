import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from looming_spots.trial_group_analysis.escape_metric_dataframes import (
    get_behaviour_metric_dataframe,
)
from looming_spots.db import experimental_log, loom_trial_group

sns.set_style("whitegrid")
LINEWIDTH = 3
flatui = ["#9b59b6", "#3498db", "#95a5a6"]

GROUPS = {
    "OHDA": [
        "CA451A_1",
        "CA451A_2",
        "CA451A_3",
        "CA451A_4",
        "CA507_1",
        "CA507_3",
        "CA507_4",
        "CA493_1",
        "CA493_2",
    ],  # "CA478_2", "CA476_4" exclude - bad drug batch
    "NMDA": ["276585A", "276585B", "CA439_1", "CA439_4"],
    "CONTROL": [
        "276585D",
        "276585E",
        "CA452_1",
        "CA439_5",
        "CA451A_5",
        "CA459A_2",
        "CA478_3",
        "CA476_5",
        "CA507_2",
        "CA507_5",
        "CA493_3",
        "CA493_4",
    ],
    "d1_caspase": experimental_log.get_mouse_ids_in_experiment(
        "d1MSN_caspase_lesion_TS"
    ),
    "d2_caspase": experimental_log.get_mouse_ids_in_experiment(
        "d2MSN_caspase_lesion_TS"
    ),
    "d2_caspase_AAV2": ["1032008", "1041545", "1041546", "1032007", "1032010"],
    "d1_caspase_AAV5": ["1068966", "1068091", "1068090", "1068089"],
    "d1_caspase_AAV2": ["FI1_1", "FI1_2", "FI1_3", "FI1_4", "1057614"],
    "caspase": [
        "FI1_1",
        "FI1_2",
        "FI1_3",
        "FI1_4",
        "1057614",
        "1068966",
        "1068091",
        "1068090",
        "1068089",
    ]
    + experimental_log.get_mouse_ids_in_experiment("d1MSN_caspase_lesion_TS"),
    "naive_escape": experimental_log.get_mouse_ids_in_experiment(
        "naive_escape_in_A"
    ),
    "d1flexGCaMP_var_contrast": experimental_log.get_mouse_ids_in_experiment(
        "d1flexGCaMP_var_contrast"
    ),
    "escape_in_a": [
        "CA105_1",
        "CA105_2",
        "CA105_3",
        "CA105_4",
        "CA105_5",
        "CA106_1",
        "CA106_2",
        "CA106_3",
        "CA106_4",
        "CA106_5",
        "CA109_1",
        "CA109_2",
        "CA109_3",
    ],
    "A10_naive_pre_tests": [
        "CA389_5",
        "CA389_4",
        "CA389_2",
        "CA389_3",
    ],
    "to_process": [
        "CA114_2",
        "CA50_3",
        "CA114_3",
        "CA131_4",
        "CA132_1",
        "CA114_5",
        "CA41_4",
        "CA40_1",
        "CA475_1",
        "CA113_4",
        "CA132_3",
        "CA188_4",
        "CA114_1",
        "CA131_3",
        "CA114_4",
        "CA132_4",
    ],
}


def get_sub_dictionary(wanted_keys, bigdict):
    if wanted_keys is None:
        return bigdict
    return dict((k, bigdict[k]) for k in wanted_keys if k in bigdict)


def get_behaviour_metrics_df(metric, group_keys=None):
    group_dict = get_sub_dictionary(group_keys, GROUPS)
    if len(group_dict) == 0:
        group_dict = {
            key: experimental_log.get_mouse_ids_in_experiment(key)
            for key in group_keys
        }
    all_df = pd.DataFrame()
    for label, mids in group_dict.items():
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

    df = get_behaviour_metrics_df("classified as flee", groups)
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


def plot_lesion_experiments(metric="speed", units="cm/s", group_keys=None):
    flatui = ["b", "k", "r", "g", "c"][: len(group_keys)]
    df_all = get_behaviour_metrics_df(metric, group_keys=group_keys)
    ax = plt.subplot(221)
    ax.set_title("escape % vs contrast")
    # sns.lineplot(
    #     data=df_all,
    #     x="contrast",
    #     y="escape",
    #     hue="experimental group",
    #     err_style="bars",
    #     linewidth=LINEWIDTH,
    #     palette=flatui
    # )

    sns.pointplot(
        data=df_all,
        x="contrast",
        y="escape",
        hue="experimental group",
        palette=flatui,
        order=[0.1507, 0.1407, 0.1307, 0.1207, 0.1107, 0.1007, 0],
        facecolor="w",
    )

    ax.set(xlabel="spot luminance", ylabel="escape (%)")
    # ax.invert_xaxis()

    ax2 = plt.subplot(222)
    ax2.set_title("raw (all trials) speed vs trial no.")
    sns.pointplot(
        data=df_all,
        x="loom number",
        y="metric value",
        hue="experimental group",
        legend=False,
        linewidth=LINEWIDTH,
        palette=flatui,
    )
    # sns.scatterplot(
    #     data=df_all,
    #     x="loom number",
    #     y="metric value",
    #     hue="experimental group",
    #     legend=False,
    #     palette=flatui
    # )
    ax2.set(xlabel="trial number", ylabel=f"{metric} ({units})")

    df_test_contrast = df_all[df_all["contrast"] == 0]
    ax3 = plt.subplot(223)
    ax3.set_title("test contrast trials speed vs trial no.")
    sns.lineplot(
        data=df_test_contrast,
        x="loom number",
        y="metric value",
        hue="experimental group",
        err_style="bars",
        legend=False,
        linewidth=LINEWIDTH,
        palette=flatui,
    )
    sns.scatterplot(
        data=df_test_contrast,
        x="loom number",
        y="metric value",
        hue="experimental group",
        legend=False,
        edgecolor="w",
        alpha=0.0,
        palette=flatui,
    )

    ax3.set(xlabel="trial number", ylabel=f"{metric} ({units})")

    ax4 = plt.subplot(224)
    ax4.set_title("test contrast trials escape % vs trial no.")
    sns.lineplot(
        data=df_test_contrast,
        x="loom number",
        y="escape",
        hue="experimental group",
        err_style="bars",
        legend=False,
        linewidth=LINEWIDTH,
        palette=flatui,
    )
    ax4.set(xlabel="loom number", ylabel="escape (%)")
    return df_all
